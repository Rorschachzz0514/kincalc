# -*- coding: utf-8 -*-
"""
Step 3  把原文 + CBDB 人物/亲属边 拼成训练样本 (EpiNet v0), 划分 train/dev/test

输入  汇编文本_分块.jsonl, 汇编_CBDB映射.csv, CBDB (BIOG_MAIN, KIN_DATA, KINSHIP_CODES, POSTED_TO_OFFICE_DATA, OFFICE_CODES, BIOG_ADDR_DATA, ADDR_CODES)
输出  TRAIN_DIR/{train,dev,test}.jsonl, TRAIN_DIR/README.md

每篇样本
  nodes  (a) 本篇在 CBDB 登记的人物 in_doc=True;  (b) 它们亲属边的另一端若不在 (a) 中, 也加入, in_doc=False
         每个节点: id(CBDB personid), name(繁), name_simp, gender(1男/0女/None), birth, death, offices[], origin[], span, in_doc, path_from_ego
  edges  (a) 中每个人物的全部 KIN_DATA 出边: {head, rel, tail}, 语义 "tail 是 head 的 rel"
  ego    墓主: in_doc 节点中姓名(繁)等于我方表 subjects 里任一姓名者; 找不到则用志题里出现的人名; 再找不到 None
  path_from_ego  从 ego 出发 BFS ≤5 跳: 沿出边直接拼接 (F + F = FF); 出边不够时把入边反转 (X 的 S 是 ego → X 是 ego 的 F/M, 看 X 性别)
                 然后 normalize_path: FS→B, FFS→FB, FD→Z ...
  span   name_in_text 在 text(简体) 里的 [start, end]; 找不到 None
划分  按 code 的 md5 取模: 0-7 train, 8 dev, 9 test (同一编号永远在同一份里)
"""
import os, sys, json, hashlib, collections
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
from common import (HUIBIAN_DIR, TRAIN_DIR, cbdb, era_tables, code_from_cbdb_pages, name_in_text, clean_name, t2s, t2t,
                    base_steps, inverse_path, normalize_path, TEXTID_ZHENGBIAN, TEXTID_XUJI)


def ego_paths(ego, nodes, edges):
    gender = {n["id"]: n["gender"] for n in nodes}
    adj = collections.defaultdict(list)
    for h, rel, t in edges:
        adj[h].append((t, rel))
        st = base_steps(rel)
        if st:
            inv = inverse_path(st, gender.get(h), gender.get(t))
            if inv:
                adj[t].append((h, inv))
    paths, frontier = {ego: "EGO"}, [ego]
    for _ in range(5):
        nxt = []
        for u in frontier:
            for v, rel in adj[u]:
                if v not in paths:
                    paths[v] = rel if paths[u] == "EGO" else paths[u] + rel; nxt.append(v)
        frontier = nxt
    return {k: normalize_path(v) for k, v in paths.items()}


def gender_of(female):
    return None if pd.isna(female) else (0 if int(female) == 1 else 1)


def main():
    c = cbdb(); pin, _ = era_tables()
    docs = {d["rid"]: d for d in (json.loads(l) for l in open(os.path.join(HUIBIAN_DIR, "汇编文本_分块.jsonl"), encoding="utf-8"))}
    mp = pd.read_csv(os.path.join(HUIBIAN_DIR, "汇编_CBDB映射.csv"), encoding="utf-8-sig"); mp = mp[mp.matched]
    src = pd.read_sql(f"""select s.c_personid, s.c_textid, s.c_pages, b.c_name_chn, b.c_female, b.c_birthyear, b.c_deathyear
                          from BIOG_SOURCE_DATA s join BIOG_MAIN b on s.c_personid=b.c_personid where s.c_textid in ({TEXTID_ZHENGBIAN},{TEXTID_XUJI})""", c)
    src["code"] = src.c_pages.map(lambda p: code_from_cbdb_pages(p, pin)); src["book"] = src.c_textid.map({TEXTID_ZHENGBIAN: "正编", TEXTID_XUJI: "续集"})
    by = {k: g.drop_duplicates("c_personid") for k, g in src.dropna(subset=["code"]).groupby(["book", "code"])}
    pids = tuple(int(x) for x in src.c_personid.unique())
    kin = pd.read_sql(f"""select k.c_personid h, kc.c_kinrel rel, k.c_kin_id t, b.c_name_chn tname, b.c_female tfemale
                          from KIN_DATA k join KINSHIP_CODES kc on k.c_kin_code=kc.c_kincode join BIOG_MAIN b on k.c_kin_id=b.c_personid
                          where k.c_personid in {pids}""", c)
    kin_by_h = {h: g for h, g in kin.groupby("h")}
    off = pd.read_sql(f"select p.c_personid, oc.c_office_chn o from POSTED_TO_OFFICE_DATA p join OFFICE_CODES oc on p.c_office_id=oc.c_office_id where p.c_personid in {pids}", c).groupby("c_personid")["o"].apply(list).to_dict()
    addr = pd.read_sql(f"select a.c_personid, ac.c_name_chn a from BIOG_ADDR_DATA a join ADDR_CODES ac on a.c_addr_id=ac.c_addr_id where a.c_personid in {pids} and a.c_addr_type=1", c).groupby("c_personid")["a"].apply(list).to_dict()

    samples = []
    for m in mp.itertuples():
        d = docs[m.rid]; ts = t2s(d["text"]); g = by[(m.book, m.code)]
        nodes, seen = [], set()
        for p in g.itertuples():
            pid = int(p.c_personid); seen.add(pid)
            nodes.append(dict(id=pid, name=p.c_name_chn, name_simp=t2s(clean_name(p.c_name_chn)), gender=gender_of(p.c_female),
                              birth=None if pd.isna(p.c_birthyear) else int(p.c_birthyear), death=None if pd.isna(p.c_deathyear) else int(p.c_deathyear),
                              offices=off.get(pid, []), origin=addr.get(pid, []), span=name_in_text(p.c_name_chn, ts), in_doc=True))
        edges = []
        for n in list(nodes):
            kg = kin_by_h.get(n["id"])
            if kg is None:
                continue
            for k in kg.itertuples():
                t = int(k.t); edges.append((n["id"], k.rel, t))
                if t not in seen:
                    seen.add(t)
                    nodes.append(dict(id=t, name=k.tname, name_simp=t2s(clean_name(k.tname)), gender=gender_of(k.tfemale), birth=None, death=None,
                                      offices=off.get(t, []), origin=addr.get(t, []), span=name_in_text(k.tname, ts), in_doc=False))
        subj_names = {t2t(clean_name(s["name"])) for s in d["subjects"]}
        egos = [n["id"] for n in nodes if n["in_doc"] and t2t(clean_name(n["name"])) in subj_names]
        if not egos and d["title"]:
            tl = t2s(d["title"]); egos = [n["id"] for n in nodes if n["in_doc"] and len(n["name_simp"]) >= 2 and n["name_simp"] in tl][:1]
        ego = egos[0] if egos else None
        paths = ego_paths(ego, nodes, edges) if ego else {}
        for n in nodes:
            n["path_from_ego"] = paths.get(n["id"])
        samples.append(dict(id="%s_%s" % (d["code"], d["rid"]), rid=d["rid"], 编号=d["编号"], code=d["code"], book=m.book, cover=d["cover"], title=d["title"],
                            preface=d["preface"], inscription=d["inscription"], text=d["text"], subjects=d["subjects"], ego=ego, egos=egos,
                            nodes=nodes, edges=[dict(head=h, rel=r, tail=t) for h, r, t in edges]))

    os.makedirs(TRAIN_DIR, exist_ok=True)
    fh = {s: open(os.path.join(TRAIN_DIR, s + ".jsonl"), "w", encoding="utf-8") for s in ("train", "dev", "test")}
    cnt = collections.Counter()
    for s in samples:
        h = int(hashlib.md5(s["code"].encode()).hexdigest(), 16) % 10
        split = "train" if h < 8 else ("dev" if h == 8 else "test"); cnt[split] += 1
        fh[split].write(json.dumps(s, ensure_ascii=False) + "\n")
    for f in fh.values():
        f.close()
    open(os.path.join(TRAIN_DIR, "README.md"), "w", encoding="utf-8").write(README)
    print("样本 %d 篇; 划分 %s; 认出 ego %d; 节点 %d (in_doc %d); 边 %d"
          % (len(samples), dict(cnt), sum(s["ego"] is not None for s in samples), sum(len(s["nodes"]) for s in samples),
             sum(n["in_doc"] for s in samples for n in s["nodes"]), sum(len(s["edges"]) for s in samples)))


README = """# EpiNet v0 — 墓志原文 → 人物网络 训练集

由 pipeline/step1~3 生成。来源: 我方《唐代墓志汇编》原文 × CBDB 人工亲属库 (按汇编编号映射, 人名在文中核验)。
处理过程详见 pipeline/数据处理说明.md。

每行一篇墓志 (JSON):
- id, rid(我方表行号), 编号, code(汇编编号), book(正编/续集)
- cover(盖文) title(志题) preface(序文) inscription(铭文) text(全文, 无换行/空格; span 偏移基于 text)
- subjects [{rid,name,gender,death_year}] 我方表里的墓主 (合葬多人)
- ego 第一个认出的墓主 CBDB personid (None=未认出); egos 全部认出的
- nodes [{id, name(繁), name_simp, gender(1男0女), birth, death, offices[], origin[], span[start,end]|None, in_doc, path_from_ego}]
- edges [{head, rel, tail}]  "tail 是 head 的 rel" (CBDB 亲属码)

用法: 文档级抽取 (text → 有 span 的 nodes + path_from_ego), 三元组抽取 (text → 两端有 span 的 edges)。
注意: CBDB 未必录全文中所有人 → 查全率可信, 查准率需人工复核; CBDB 有重复人物 (异体字不同 id)。
"""

if __name__ == "__main__":
    main()
