# -*- coding: utf-8 -*-
"""B1 —— 配对 bootstrap: 银标 F1 的差是否显著 (dev+test 共 395 篇, 按文档重采样 2000 次)"""
import os, sys, json, random, collections
sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
def same(pred, gold):
    if pred == gold: return True
    g = gold[1:] if len(gold) >= 2 else gold
    return len(g) >= 1 and (pred == g or (len(pred) >= 2 and pred[1:] == g) or (len(g) >= 2 and pred.endswith(g)))
SYS = ["G", "GR", "Aall", "Acons2", "Ref2", "JOINT", "STcons", "STrand"]
per = {s: [] for s in SYS + ["金标"]}
for sp in ("dev", "test"):
    cache = json.load(open(os.path.join(HERE, "runs", "judge_cache_%s_v2oss.json" % sp), encoding="utf-8"))
    docs = {}
    for l in open(os.path.join(HERE, "sft", "G_%s.jsonl" % sp), encoding="utf-8"):
        d = json.loads(l); docs[d["id"]] = d
    preds = {"金标": {i: dict(tuple(x.split(" | ")) for x in d["target"].splitlines()) for i, d in docs.items()}}
    for s in SYS: preds[s] = {json.loads(l)["id"]: json.loads(l)["pred"] for l in open(os.path.join(HERE, "runs", s, "pred_%s.jsonl" % sp), encoding="utf-8")}
    # 银标池只用这里列出的系统 + 金标, 保证各系统在同一池上比较
    for i in docs:
        ok = sorted({(n, p) for s in preds for n, p in preds[s].get(i, {}).items() if p and cache.get("%s\t%s\t%s" % (i, n, p), {}).get("ok")})
        cl = []
        for n, p in ok:
            for c in cl:
                if c[0] == p and any(same(n, m) or same(m, n) or set(n) <= set(m) or set(m) <= set(n) for m in c[1]): c[1].add(n); break
            else: cl.append((p, {n}))
        for s in preds:
            pr = {n: p for n, p in preds[s].get(i, {}).items() if p}
            n_ok = sum(1 for n, p in pr.items() if cache.get("%s\t%s\t%s" % (i, n, p), {}).get("ok"))
            hit = sum(1 for c in cl if any(p == c[0] and n in c[1] for n, p in pr.items()))
            per[s].append((len(pr), n_ok, hit, len(cl)))
N = len(per["G"])
def f1(rows):
    o = sum(r[0] for r in rows); k = sum(r[1] for r in rows); h = sum(r[2] for r in rows); t = sum(r[3] for r in rows)
    P = k / max(1, o); R = h / max(1, t); return P, R, 2 * P * R / max(1e-9, P + R)
S = ["B1 银标 F1 (dev+test 合并 %d 篇, 同一候选池) 与配对 bootstrap" % N, "系统      精确率   召回率   F1"]
for s in ["金标"] + SYS:
    P, R, F = f1(per[s]); S.append("%-8s  %.1f%%   %.1f%%   %.1f" % (s, P * 100, R * 100, F * 100))
rng = random.Random(0)
def boot(a, b, B=2000):
    ds = []
    for _ in range(B):
        idx = [rng.randrange(N) for _ in range(N)]
        ds.append(f1([per[a][i] for i in idx])[2] - f1([per[b][i] for i in idx])[2])
    ds.sort(); return (f1(per[a])[2] - f1(per[b])[2]) * 100, ds[int(B * .025)] * 100, ds[int(B * .975)] * 100, sum(d <= 0 for d in ds) / B
S.append("F1 之差 (95% 区间; p = 差值<=0 的比例)")
for a, b in (("Aall", "G"), ("Ref2", "G"), ("Acons2", "G"), ("STcons", "STrand"), ("STcons", "Acons2"), ("GR", "G"), ("JOINT", "G"), ("G", "金标"), ("Aall", "金标")):
    d, lo, hi, p = boot(a, b); S.append("  %s − %s: %+.1f  [%+.1f, %+.1f]  p=%.3f" % (a, b, d, lo, hi, p))
open(os.path.join(HERE, "runs", "B1_bootstrap.txt"), "w", encoding="utf-8").write("\n".join(S)); print("\n".join(S))
