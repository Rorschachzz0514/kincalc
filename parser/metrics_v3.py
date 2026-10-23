# -*- coding: utf-8 -*-
"""v3 指标 (补充实验): 在 v2 核验参照上评测新增系统 (多种子 / Llama 底座 / 未微调底座 / 零样本大模型 / 自训练数据量),
   新增: 名字规范化后的精确率、未核验输出的处理 (乐观=跳过, 悲观=计错)、第二裁决模型一致率、多种子均值±标准差。
   参照集与 v2 完全相同: 由原 7 个系统 + 人工库的候选池经逐条核验得到, 新系统的输出若不在池内则无法核验, 分别按乐观/悲观两种口径报告。
输出: runs/verify_v2/metrics_v3.json
用法: python metrics_v3.py [JUDGE_TAG ...]   例: python metrics_v3.py _v2oss _v2ds
"""
import os, sys, json, csv, re, collections, statistics
sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__)); V = os.path.join(HERE, "runs", "verify_claude"); OUT = os.path.join(HERE, "runs", "verify_v2")
def same(pred, gold):
    if pred == gold: return True
    g = gold[1:] if len(gold) >= 2 else gold
    return len(g) >= 1 and (pred == g or (len(pred) >= 2 and pred[1:] == g) or (len(g) >= 2 and pred.endswith(g)))
def tsv(name): return {int(r[0]): (r[1], r[2] if len(r) > 2 else "", r[3] if len(r) > 3 else "") for r in (l.rstrip("\n").split("\t") for l in open(os.path.join(V, name), encoding="utf-8") if l.strip())}
T1 = tsv("t1.tsv"); rows1 = list(csv.DictReader(open(os.path.join(HERE, "runs", "verify", "1_测试集候选.csv"), encoding="utf-8-sig")))
sp = "test"; docs = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(HERE, "sft", "G_%s.jsonl" % sp), encoding="utf-8")}
gold = {i: dict(tuple(x.split(" | ")) for x in d["target"].splitlines()) for i, d in docs.items()}
POOL = ["G", "Aall", "Ref2", "JOINT", "STcons", "SYN_full", "U_apiall"]                       # 构成核验候选池的系统 (v2)
EXTRA = ["G_s1", "G_s2", "Aall_s1", "Aall_s2", "STcons_s1", "STcons_s2", "SYN_full_s1", "SYN_full_s2", "STc500", "STc1k", "STc2k", "G_llama", "Aall_llama", "QBASE", "APIZS"]
# 零样本大模型 (api_pseudo 输出) → pred 格式
api = os.path.join(HERE, "..", "U_无监督", "runs", "api_test_qwen397.jsonl")
if os.path.exists(api):
    os.makedirs(os.path.join(HERE, "runs", "APIZS"), exist_ok=True)
    with open(os.path.join(HERE, "runs", "APIZS", "pred_test.jsonl"), "w", encoding="utf-8") as f:
        for l in open(api, encoding="utf-8"):
            d = json.loads(l); f.write(json.dumps({"id": d["id"], "pred": d["direct"]}, ensure_ascii=False) + "\n")
preds = {}
for s in POOL + EXTRA:
    fp = os.path.join(HERE, "runs", s, "pred_%s.jsonl" % sp)
    if os.path.exists(fp): preds[s] = {json.loads(l)["id"]: json.loads(l)["pred"] for l in open(fp, encoding="utf-8")}
print("systems:", list(preds))
TERMS = set("父母祖考妣子女夫妻兄弟姊妹孙侄甥婿嗣君公氏某") | {"夫人", "府君", "墓主", "姓名", "祖父", "祖母", "曾祖", "高祖", "外祖", "嗣子", "长子", "次子", "季子", "长女", "次女", "伯父", "叔父"}
# ---------- 名字规范化 (去称谓 / 括注 / 郡望 / "夫人")
KIN = ["曾祖母", "曾祖", "高祖", "祖母", "祖父", "外祖", "夫人", "嗣子", "长子", "次子", "季子", "长女", "次女", "哀子", "孤子", "显考", "烈考", "先考", "皇考", "皇妣", "先妣"]
PAREN = re.compile(r"[（(][^）)]*[）)]")
def norm(n):
    n = PAREN.sub("", n).strip(); n = re.sub(r"^夫人", "", n); n = re.sub(r"夫人$", "", n)
    m = re.match(r"^(.{2,3})?(.)氏$", n)
    if m and m.group(1): n = m.group(2) + "氏"
    for k in sorted(KIN, key=len, reverse=True):
        if n.startswith(k) and len(n) > len(k): n = n[len(k):]; break
    return n
# ---------- 核验结论表: (文档, 名, 码) → 结论; 另建规范化名表与"应为 X"表
verd = {}; corr = {}
for r in rows1:
    c = T1.get(int(r["序号"]), ("未核对", "", ""))
    names = [r["候选人名"]] + [n for n in (r["名字其他写法"] or "").split("、") if n]
    for n in names: verd[(r["文档"], n, r["关系码"])] = c[0]
    if c[0] == "名字有误":
        m = re.search(r"应为[\"“]([^\"”]+)[\"”]", c[2] or "")
        if m: corr[(r["文档"], m.group(1), r["关系码"])] = "对"
vnorm = {(i, norm(n), p): c for (i, n, p), c in verd.items() if c == "对"}
judges = {tag: json.load(open(os.path.join(HERE, "runs", "judge_cache_%s%s.json" % (sp, tag)), encoding="utf-8")) for tag in (sys.argv[1:] or ["_v2oss"]) if os.path.exists(os.path.join(HERE, "runs", "judge_cache_%s%s.json" % (sp, tag)))}

def cluster(i, systems):
    cands = collections.defaultdict(set); srcs = collections.defaultdict(set)
    allc = [(n, p, "KB") for n, p in gold[i].items()] + [(n, p, s) for s in systems for n, p in preds[s].get(i, {}).items() if p]
    for n, p, src in allc:
        if not n or n in TERMS or len(n) > 6: continue
        key = next((k for k in cands if k[0] == p and any(same(n, m) or same(m, n) for m in cands[k])), None)
        if key is None: key = (p, n)
        cands[key].add(n); srcs[key].add(src)
    return cands, srcs

agree = {t: collections.Counter() for t in judges}; stat = {s: collections.Counter() for s in list(preds) + ["KB"]}; ref_n = 0
for i in docs:
    cands, srcs = cluster(i, list(preds)); status = {}
    for (p, _), names in cands.items():
        c = next((verd[(i, n, p)] for n in names if (i, n, p) in verd), None)
        if c is None:
            # 未抽查: 只有"人工库 + 池内多数系统同意"的条目视为对 (与 v2 相同); 其他 (新系统独有) 记为未核验
            c = "对(未抽查)" if "KB" in srcs[(p, _)] else "未核验"
        status[(p, frozenset(names))] = c
        if c in ("对", "错", "无此人", "名字有误"):
            for t, J in judges.items():
                jv = next((J["%s\t%s\t%s" % (i, n, p)] for n in names if "%s\t%s\t%s" % (i, n, p) in J), None)
                if jv is not None: agree[t]["n"] += 1; agree[t]["ok"] += (jv["ok"] == (c == "对"))
    ref = [k for k, c in status.items() if c in ("对", "对(未抽查)")]; ref_n += len(ref)
    for s in list(preds) + ["KB"]:
        pr = gold[i] if s == "KB" else preds[s].get(i, {})
        for n, p in pr.items():
            if not p or n in TERMS or len(n) > 6: continue
            k = next((k for k in status if k[0] == p and n in k[1]), None)
            c = status[k] if k else "未核验"
            if c in ("无法判断", "未核对"): continue
            if c == "未核验": stat[s]["unverified"] += 1; continue
            ok = c in ("对", "对(未抽查)"); okn = ok or (i, norm(n), p) in vnorm or (i, norm(n), p) in corr
            stat[s]["out"] += 1; stat[s]["ok"] += ok; stat[s]["ok_len"] += c in ("对", "对(未抽查)", "名字有误"); stat[s]["ok_norm"] += okn
        hit = {k for k in ref if any(k[0] == p and n in k[1] for n, p in pr.items())}; stat[s]["hit"] += len(hit)
R = {"test_docs": len(docs), "reference_persons_strict": ref_n, "judge_agreement": {t: [a["ok"], a["n"], round(a["ok"] / max(1, a["n"]) * 100, 1)] for t, a in agree.items()}, "systems": {}}
for s, c in stat.items():
    out, unv = c["out"], c["unverified"]; Rr = c["hit"] / max(1, ref_n)
    def f1(P): return round(2 * P * Rr / max(1e-9, P + Rr) * 100, 1)
    P = c["ok"] / max(1, out); Pp = c["ok"] / max(1, out + unv); Pn = c["ok_norm"] / max(1, out)
    R["systems"][s] = {"out": out, "unverified": unv, "P_strict": round(P * 100, 1), "P_strict_pessimistic": round(Pp * 100, 1), "P_norm": round(Pn * 100, 1), "P_lenient": round(c["ok_len"] / max(1, out) * 100, 1),
                       "R_strict": round(Rr * 100, 1), "F1_strict": f1(P), "F1_pessimistic": f1(Pp), "F1_norm": f1(Pn)}
# ---------- 多种子: 测试集 F1 与开发集召回 (eval_dev.txt 第一行 "召回 xx%")
def dev_recall(name):
    f = os.path.join(HERE, "runs", name, "eval_dev.txt")
    if not os.path.exists(f): return None
    m = re.search(r"召回 ([\d.]+)%", open(f, encoding="utf-8").read()); return float(m.group(1)) if m else None
R["seeds"] = {}
for base in ["G", "Aall", "STcons", "SYN_full"]:
    names = [n for n in [base, base + "_s1", base + "_s2"] if n in R["systems"]]
    f1s = [R["systems"][n]["F1_strict"] for n in names]; devs = [dev_recall(n) for n in names]; devs = [d for d in devs if d is not None]
    R["seeds"][base] = {"runs": names, "test_F1": f1s, "test_F1_mean": round(statistics.mean(f1s), 1) if f1s else None, "test_F1_std": round(statistics.pstdev(f1s), 2) if len(f1s) > 1 else None,
                        "dev_recall": devs, "dev_recall_mean": round(statistics.mean(devs), 1) if devs else None, "dev_recall_std": round(statistics.pstdev(devs), 2) if len(devs) > 1 else None}
R["dev_recall"] = {n: dev_recall(n) for n in list(preds)}
os.makedirs(OUT, exist_ok=True); json.dump(R, open(os.path.join(OUT, "metrics_v3.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1); print(json.dumps(R, ensure_ascii=False, indent=1))
