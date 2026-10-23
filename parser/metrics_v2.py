# -*- coding: utf-8 -*-
"""v2 指标: 用 Claude 逐条核验结果 (数据校验/claude_verdicts) 重算测试集各系统 P/R/F1、人工库标签错误率、裁决模型可靠度、可信度等级正确率、纠错/补全准确率"""
import os, sys, json, csv, collections
sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__)); V = os.path.join(HERE, "runs", "verify_claude"); OUT = os.path.join(HERE, "runs", "verify_v2"); os.makedirs(OUT, exist_ok=True)
def same(pred, gold):
    if pred == gold: return True
    g = gold[1:] if len(gold) >= 2 else gold
    return len(g) >= 1 and (pred == g or (len(pred) >= 2 and pred[1:] == g) or (len(g) >= 2 and pred.endswith(g)))
def tsv(name): return {int(r[0]): (r[1], r[2] if len(r) > 2 else "", r[3] if len(r) > 3 else "") for r in (l.rstrip("\n").split("\t") for l in open(os.path.join(V, name), encoding="utf-8") if l.strip())}
T1 = tsv("t1.tsv"); rows1 = list(csv.DictReader(open(os.path.join(HERE, "runs", "verify", "1_测试集候选.csv"), encoding="utf-8-sig")))
# ---------- 测试集: 重建候选池 (含未抽查的"一致"条目)
sp = "test"; docs = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(HERE, "sft", "G_%s.jsonl" % sp), encoding="utf-8")}
gold = {i: dict(tuple(x.split(" | ")) for x in d["target"].splitlines()) for i, d in docs.items()}
SYS = ["G", "Aall", "Ref2", "JOINT", "STcons", "SYN_full", "U_apiall"]; preds = {}
for s in SYS:
    f = os.path.join(HERE, "runs", s, "pred_%s.jsonl" % sp)
    if os.path.exists(f): preds[s] = {json.loads(l)["id"]: json.loads(l)["pred"] for l in open(f, encoding="utf-8")}
TERMS = set("父母祖考妣子女夫妻兄弟姊妹孙侄甥婿嗣君公氏某") | {"夫人", "府君", "墓主", "姓名", "祖父", "祖母", "曾祖", "高祖", "外祖", "嗣子", "长子", "次子", "季子", "长女", "次女", "伯父", "叔父"}
verd = {}                       # (doc, 主名, path) -> 结论
for r in rows1:
    c = T1.get(int(r["序号"]), ("未核对", "", ""))[0]; verd[(r["文档"], r["候选人名"], r["关系码"])] = c
    for n in (r["名字其他写法"] or "").split("、"):
        if n: verd[(r["文档"], n, r["关系码"])] = c
judge = json.load(open(os.path.join(HERE, "runs", "judge_cache_%s_v2oss.json" % sp), encoding="utf-8"))
def cluster(i):
    cands = collections.defaultdict(set); srcs = collections.defaultdict(set)
    allc = [(n, p, "KB") for n, p in gold[i].items()] + [(n, p, s) for s in preds for n, p in preds[s].get(i, {}).items() if p]
    for n, p, src in allc:
        if not n or n in TERMS or len(n) > 6: continue
        key = next((k for k in cands if k[0] == p and any(same(n, m) or same(m, n) for m in cands[k])), None)
        if key is None: key = (p, n)
        cands[key].add(n); srcs[key].add(src)
    return cands, srcs
agree = collections.Counter(); stat = {s: collections.Counter() for s in list(preds) + ["KB"]}; ref_n = 0; kb_err = collections.Counter(); unanimous_assumed = 0
for i in docs:
    cands, srcs = cluster(i); status = {}
    for (p, _), names in cands.items():
        c = next((verd[(i, n, p)] for n in names if (i, n, p) in verd), None)
        if c is None:                                               # 未抽查的"人工库且多数模型同意"条目 → 视为对 (抽查 100 条的正确率另报)
            c = "对(未抽查)"; unanimous_assumed += 1
        status[(p, frozenset(names))] = c
        if c in ("对", "错", "无此人", "名字有误"):                     # 裁决模型一致率
            jv = next((judge["%s\t%s\t%s" % (i, n, p)] for n in names if "%s\t%s\t%s" % (i, n, p) in judge), None)
            if jv is not None: agree["n"] += 1; agree["ok"] += (jv["ok"] == (c in ("对",)))
    ref = [k for k, c in status.items() if c in ("对", "对(未抽查)")]; ref_len = [k for k, c in status.items() if c in ("对", "对(未抽查)", "名字有误")]
    ref_n += len(ref)
    for s in list(preds) + ["KB"]:
        pr = gold[i] if s == "KB" else preds[s].get(i, {})
        for n, p in pr.items():
            if not p or n in TERMS or len(n) > 6: continue
            k = next((k for k in status if k[0] == p and n in k[1]), None)
            if k is None: continue
            c = status[k]
            if c in ("无法判断", "未核对"): continue
            stat[s]["out"] += 1; stat[s]["ok"] += c in ("对", "对(未抽查)"); stat[s]["ok_len"] += c in ("对", "对(未抽查)", "名字有误")
            if s == "KB": kb_err[c] += 1
        hit = {k for k in ref if any(k[0] == p and n in k[1] for n, p in pr.items())}; stat[s]["hit"] += len(hit)
        hitl = {k for k in ref_len if any(k[0] == p and n in k[1] for n, p in pr.items())}; stat[s]["hit_len"] += len(hitl)
ref_len_n = sum(1 for i in docs for k, c in [(k, c) for k, c in cluster(i)[0].items()] if False)  # placeholder
R = {"test_docs": len(docs), "reference_persons_strict": ref_n, "unanimous_assumed_correct": unanimous_assumed, "judge_agreement": [agree["ok"], agree["n"], round(agree["ok"] / max(1, agree["n"]) * 100, 1)], "kb_label_verdicts": dict(kb_err), "systems": {}}
for s, c in stat.items():
    P = c["ok"] / max(1, c["out"]); Rr = c["hit"] / max(1, ref_n); Pl = c["ok_len"] / max(1, c["out"])
    R["systems"][s] = {"out": c["out"], "P_strict": round(P * 100, 1), "R_strict": round(Rr * 100, 1), "F1_strict": round(2 * P * Rr / max(1e-9, P + Rr) * 100, 1), "P_lenient": round(Pl * 100, 1)}
# ---------- 其他表
def simple(name, csvname, col="序号"):
    T = tsv(name); rows = list(csv.DictReader(open(os.path.join(HERE, "runs", "verify", csvname), encoding="utf-8-sig")))
    return T, rows
T5, rows5 = simple("t5.tsv", "5_网络边抽样.csv"); tier = collections.defaultdict(collections.Counter)
for r in rows5:
    c = T5[int(r["序号"])][0]; tier[r["可信度等级"]][c] += 1
R["tiers"] = {t: {"acc_strict": round(c["对"] / max(1, sum(v for k, v in c.items() if k != "无法判断")) * 100, 1), "acc_lenient": round((c["对"] + c["名字有误"]) / max(1, sum(v for k, v in c.items() if k != "无法判断")) * 100, 1), "counts": dict(c)} for t, c in sorted(tier.items())}
T3, _ = simple("t3.tsv", "3_纠错建议.csv"); c3 = collections.Counter(v[0] for v in T3.values()); R["flips"] = dict(c3)
T4, _ = simple("t4.tsv", "4_补全建议.csv"); c4 = collections.Counter(v[0] for v in T4.values()); R["additions"] = dict(c4)
T2, _ = simple("t2.tsv", "2_跨篇推断.csv"); c2 = collections.Counter(v[0] for v in T2.values()); R["crossdoc_contested"] = dict(c2)
c1 = collections.Counter(v[0] for v in T1.values()); R["t1_verdict_counts"] = dict(c1)
json.dump(R, open(os.path.join(OUT, "metrics_v2.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1); print(json.dumps(R, ensure_ascii=False, indent=1))
