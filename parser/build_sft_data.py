# -*- coding: utf-8 -*-
"""
把 EpiNet v0 转成微调用的指令数据, 三种输出格式:
  G  全局路径:      姓名 | 相对中心人物的路径码                      (基线: 模型自己算全局关系)
  P  父指针分解:    姓名 | 参照人 | 相对参照人的路径码                (模型只给局部关系, 全局路径由代数沿参照链组合)
  R  换根增强:      与 G 同格式, 但中心人物换成墓主以外的人, 标签由换根算子算出 (只用于训练)
另有组合泛化切分 comp: 训练只留"所有人物 <=2 跳"的文档, 测试看 >=3 跳。
输出: 实验/M_模型/sft/{G,P}_{train,dev,test}.jsonl, R_train.jsonl, stats.txt
每行: {"id","root","prompt","target","gold":[{"name","path"}], "g_root"}
"""
import os, sys, json, random, collections
sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "V_验证"))
from kin_algebra import core, normalize, reroot, GENDER_OF
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(ROOT, "data", "训练集", "epinet_v0"); DST = os.path.join(HERE, "sft"); os.makedirs(DST, exist_ok=True)
MAXC = 2400
CODES = "关系码用 8 个字母拼接, 从起点出发逐步读: F=父 M=母 S=子 D=女 H=夫 W=妻 B=兄弟 Z=姊妹。例: 祖父=FF, 曾祖=FFF, 孙=SS, 岳父=WF, 女婿=DH, 侄=BS。"
PG = "下面是一篇唐代墓志。请以「%s」为中心人物, 列出文中出现了名字、与其有亲属关系的人, 并给出每个人相对中心人物的关系码。\n" + CODES + "\n每人一行: 姓名 | 关系码\n\n墓志:\n%s"
PP = ("下面是一篇唐代墓志。请以「%s」为中心人物, 列出文中出现了名字、与其有亲属关系的人。对每个人给出一个参照人和他相对参照人的关系码: "
      "参照人写「中心」或本列表中另一个人的姓名, 尽量选离他最近的亲属。\n" + CODES + "\n每人一行: 姓名 | 参照人 | 关系码\n\n墓志:\n%s")
UNIQUE = set("FMH")


def clip(t):
    return t if len(t) <= MAXC else t[:MAXC * 2 // 3] + "……" + t[-MAXC // 3:]


def persons(d):
    """文中有名字、路径可分解的非墓主人物, 按出现位置排序。返回 [(name, path, id, gender, pos, located)]"""
    ego = next(n for n in d["nodes"] if n["id"] == d["ego"]); g = ego["gender"]; out = {}; text = d["text"]
    for n in d["nodes"]:
        c = core(n.get("path_from_ego"))
        if n["id"] == d["ego"] or not c or not n["in_doc"]: continue
        if n["span"]: pos = n["span"][0]
        else:
            pos = text.find(n["name_simp"])
            if pos < 0: continue
        if n["name_simp"] not in out:
            out[n["name_simp"]] = (n["name_simp"], normalize(c, g), n["id"], n["gender"], pos, bool(n["span"]))
    return ego, sorted(out.values(), key=lambda x: x[4])


def parent_pointer(ps, d, g_ego):
    """每个人 → (参照人名或 '中心', 局部路径)。取最长前缀槽位上"唯一或被库中边确认"的人作参照"""
    by_path = collections.defaultdict(list)
    for p in ps: by_path[p[1]].append(p)
    edge = {(e["head"], e["tail"]): core(e["rel"]) for e in d["edges"]}
    res = []
    for name, path, pid, g, pos, loc in ps:
        anchor, local = "中心", path
        for k in range(len(path) - 1, 0, -1):
            cands = [q for q in by_path.get(path[:k], []) if q[2] != pid]
            if not cands: continue
            suf = path[k:]
            conf = [q for q in cands if edge.get((q[2], pid)) and normalize(edge[(q[2], pid)], q[3]) == suf]
            if conf: anchor, local = conf[0][0], suf; break
            if len(cands) == 1 and not any(edge.get((q[2], pid)) for q in cands): anchor, local = cands[0][0], suf; break
            if len(cands) == 1: break                                   # 库中有边但与后缀不符 → 不可信, 退回中心
        res.append((name, anchor, local))
    return res


def ambiguous(py, px):
    """两条墓主路径的公共前缀里出现非唯一槽位 (子/女/兄弟/姊妹/妻) → 不知是否同一人, 换根结果不确定"""
    for a, b in zip(py, px):
        if a != b: return False
        if a not in UNIQUE: return True
    return False


def rerooted(d, ego, ps, rng):
    """随机选一个新中心, 用换根算子 (有库中直接边则优先用边) 生成 G 格式标签"""
    edge = {(e["head"], e["tail"]): core(e["rel"]) for e in d["edges"]}
    cands = [p for p in ps if p[5] and p[3] is not None]; rng.shuffle(cands)
    everyone = ps + [(ego["name_simp"], "", ego["id"], ego["gender"], d["text"].find(ego["name_simp"][1:]) if len(ego["name_simp"]) > 1 else 0, True)]
    for y in cands[:3]:
        lines = []
        for x in sorted(everyone, key=lambda z: z[4]):
            if x[2] == y[2]: continue
            r = edge.get((y[2], x[2]))
            if r: r = normalize(r, y[3])
            elif ambiguous(y[1], x[1]): continue
            else: r = reroot(y[1], x[1], ego["gender"], y[3])
            if r: lines.append((x[0], r))
        if len(lines) >= 3: return y, lines
    return None, None


rng = random.Random(0); S = []
for sp in ("train", "dev", "test"):
    fg = open(os.path.join(DST, "G_%s.jsonl" % sp), "w", encoding="utf-8"); fp = open(os.path.join(DST, "P_%s.jsonl" % sp), "w", encoding="utf-8")
    fr = open(os.path.join(DST, "R_train.jsonl"), "w", encoding="utf-8") if sp == "train" else None
    st = collections.Counter(); hop = collections.Counter(); anch = collections.Counter()
    for line in open(os.path.join(SRC, sp + ".jsonl"), encoding="utf-8"):
        d = json.loads(line)
        if d["ego"] is None: continue
        ego, ps = persons(d)
        if not ps: st["no_person"] += 1; continue
        text = clip(d["text"]); root = ego["name_simp"]; st["docs"] += 1
        gold = [dict(name=p[0], path=p[1]) for p in ps if p[5]]          # 评测只算文中定位到名字的人 (与 V3 一致)
        base = dict(id=d["id"], root=root, g_root=ego["gender"], gold=gold, maxhop=max(len(p[1]) for p in ps))
        fg.write(json.dumps(dict(base, prompt=PG % (root, text), target="\n".join("%s | %s" % (p[0], p[1]) for p in ps)), ensure_ascii=False) + "\n")
        pp = parent_pointer(ps, d, ego["gender"])
        fp.write(json.dumps(dict(base, prompt=PP % (root, text), target="\n".join("%s | %s | %s" % t for t in pp)), ensure_ascii=False) + "\n")
        for p in ps: hop[min(len(p[1]), 5)] += 1
        for t in pp: anch["中心" if t[1] == "中心" else "他人"] += 1; st["local_len_sum"] += len(t[2]); st["persons"] += 1
        if fr:
            y, lines = rerooted(d, ego, ps, rng)
            if y:
                st["reroot"] += 1
                fr.write(json.dumps(dict(id=d["id"] + "@" + y[0], root=y[0], g_root=y[3], gold=[], maxhop=0, prompt=PG % (y[0], text),
                                         target="\n".join("%s | %s" % l for l in lines)), ensure_ascii=False) + "\n")
    S.append("%s: 文档 %d, 人物 %d (每篇 %.1f); 跳数分布 %s; P 格式参照人: 中心 %d / 他人 %d (%.1f%%), 局部路径平均长度 %.2f (全局 %.2f)%s" % (
        sp, st["docs"], st["persons"], st["persons"] / st["docs"], dict(sorted(hop.items())), anch["中心"], anch["他人"], anch["他人"] / st["persons"] * 100,
        st["local_len_sum"] / st["persons"], sum(k * v for k, v in hop.items()) / st["persons"], "; 换根样本 %d" % st["reroot"] if fr else ""))
open(os.path.join(DST, "stats.txt"), "w", encoding="utf-8").write("\n".join(S)); print("\n".join(S))
