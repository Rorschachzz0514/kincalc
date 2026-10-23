# -*- coding: utf-8 -*-
"""v4: 核验参照上的文档级 bootstrap 置信区间 + 成对差异 p 值 (测试集 182 篇, 2,000 次重采样)。
参照与评分规则同 metrics_v3 (未核验输出跳过)。输出 runs/verify_v2/metrics_v4_ci.json
用法: python metrics_v4_ci.py [系统名...]   默认: 表 2 全部系统"""
import os, sys, json, csv, re, collections, random
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
POOL = ["G", "Aall", "Ref2", "JOINT", "STcons", "SYN_full", "U_apiall"]
SYS = sys.argv[1:] or POOL + ["G_s1", "G_s2", "Aall_s1", "Aall_s2", "STcons_s1", "STcons_s2", "SYN_full_s1", "SYN_full_s2", "APIZS", "APIFS", "QBASE", "G_llama", "Aall_llama"]
preds = {}
for s in POOL + [x for x in SYS if x not in POOL]:
    fp = os.path.join(HERE, "runs", s, "pred_%s.jsonl" % sp)
    if os.path.exists(fp): preds[s] = {json.loads(l)["id"]: json.loads(l)["pred"] for l in open(fp, encoding="utf-8")}
TERMS = set("父母祖考妣子女夫妻兄弟姊妹孙侄甥婿嗣君公氏某") | {"夫人", "府君", "墓主", "姓名", "祖父", "祖母", "曾祖", "高祖", "外祖", "嗣子", "长子", "次子", "季子", "长女", "次女", "伯父", "叔父"}
verd = {}
for r in rows1:
    c = T1.get(int(r["序号"]), ("未核对", "", ""))[0]
    for n in [r["候选人名"]] + [n for n in (r["名字其他写法"] or "").split("、") if n]: verd[(r["文档"], n, r["关系码"])] = c
def cluster(i, systems):
    cands = collections.defaultdict(set); srcs = collections.defaultdict(set)
    allc = [(n, p, "KB") for n, p in gold[i].items()] + [(n, p, s) for s in systems for n, p in preds[s].get(i, {}).items() if p]
    for n, p, src in allc:
        if not n or n in TERMS or len(n) > 6: continue
        key = next((k for k in cands if k[0] == p and any(same(n, m) or same(m, n) for m in cands[k])), None)
        if key is None: key = (p, n)
        cands[key].add(n); srcs[key].add(src)
    return cands, srcs
# 文档级计数: per[s][i] = (out, ok, hit); ref[i] = 参照人数
per = {s: {} for s in list(preds) + ["KB"]}; ref = {}
for i in docs:
    cands, srcs = cluster(i, list(preds)); status = {}
    for (p, _), names in cands.items():
        c = next((verd[(i, n, p)] for n in names if (i, n, p) in verd), None)
        if c is None: c = "对(未抽查)" if "KB" in srcs[(p, _)] else "未核验"
        status[(p, frozenset(names))] = c
    R = [k for k, c in status.items() if c in ("对", "对(未抽查)")]; ref[i] = len(R)
    for s in per:
        pr = gold[i] if s == "KB" else preds[s].get(i, {}); out = ok = 0
        for n, p in pr.items():
            if not p or n in TERMS or len(n) > 6: continue
            k = next((k for k in status if k[0] == p and n in k[1]), None); c = status[k] if k else "未核验"
            if c in ("无法判断", "未核对", "未核验"): continue
            out += 1; ok += c in ("对", "对(未抽查)")
        hit = sum(1 for k in R if any(k[0] == p and n in k[1] for n, p in pr.items())); per[s][i] = (out, ok, hit)
ids = list(docs); random.seed(0); B = 2000; samples = [[random.choice(ids) for _ in ids] for _ in range(B)]
def prf(s, sample):
    out = sum(per[s][i][0] for i in sample); ok = sum(per[s][i][1] for i in sample); hit = sum(per[s][i][2] for i in sample); rn = sum(ref[i] for i in sample)
    P = ok / max(1, out); R = hit / max(1, rn); return P * 100, R * 100, 2 * P * R / max(1e-9, P + R) * 100
res = {"n_docs": len(ids), "B": B, "systems": {}, "paired_vs_G": {}}
for s in per:
    pt = prf(s, ids); bs = [prf(s, smp) for smp in samples]
    ci = lambda k: (round(sorted(b[k] for b in bs)[int(0.025 * B)], 1), round(sorted(b[k] for b in bs)[int(0.975 * B)], 1))
    res["systems"][s] = {"P": round(pt[0], 1), "R": round(pt[1], 1), "F1": round(pt[2], 1), "P_ci": ci(0), "R_ci": ci(1), "F1_ci": ci(2)}
for s in per:
    if s == "G" or "G" not in per: continue
    d0 = prf(s, ids)[2] - prf("G", ids)[2]; ds = [prf(s, smp)[2] - prf("G", smp)[2] for smp in samples]
    lo, hi = sorted(ds)[int(0.025 * B)], sorted(ds)[int(0.975 * B)]; p = min(1.0, 2 * min(sum(d <= 0 for d in ds), sum(d >= 0 for d in ds)) / B)
    res["paired_vs_G"][s] = {"dF1": round(d0, 1), "ci": (round(lo, 1), round(hi, 1)), "p": round(p, 3)}
os.makedirs(OUT, exist_ok=True); json.dump(res, open(os.path.join(OUT, "metrics_v4_ci.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
for s, v in res["systems"].items(): print("%-12s F1 %5.1f [%5.1f, %5.1f]  P %5.1f  R %5.1f   %s" % (s, v["F1"], *v["F1_ci"], v["P"], v["R"], ("dF1 vs G %+.1f [%+.1f,%+.1f] p=%.3f" % (res["paired_vs_G"][s]["dF1"], *res["paired_vs_G"][s]["ci"], res["paired_vs_G"][s]["p"])) if s in res["paired_vs_G"] else ""))
