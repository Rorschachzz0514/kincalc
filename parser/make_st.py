# -*- coding: utf-8 -*-
"""
自训练数据: 从无标注墓志的多视角伪标签 (runs/GR/pseudo_unl.jsonl) 里挑文档。
  STcons  换根一致性筛选: 至少一个其他视角、没有任何人被否定、至少 2 人被印证
  STrand  对照: 从"至少列出 2 人"的文档里随机取同样多篇 (不看一致性)
并用我方表里的 父亲/祖父 列 (独立于 CBDB 的弱标签) 检验: 一致的文档里, 模型给出的父名/祖父名是否更准。
输出: sft/STcons.jsonl, sft/STrand.jsonl, runs/make_st_stats.txt
"""
import os, sys, json, random, collections
sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
CAP = int(sys.argv[1]) if len(sys.argv) > 1 else 3000


def same(pred, gold):
    if pred == gold: return True
    g = gold[1:] if len(gold) >= 2 else gold
    return len(g) >= 1 and (pred == g or (len(pred) >= 2 and pred[1:] == g) or (len(g) >= 2 and pred.endswith(g)))


SRC = sys.argv[2] if len(sys.argv) > 2 else "runs/GR/pseudo_unl.jsonl"      # 多视角伪标签文件
PRE = sys.argv[3] if len(sys.argv) > 3 else "ST"                             # 输出前缀: sft/<PRE>cons.jsonl, sft/<PRE>rand.jsonl, 以及 <PRE>all.jsonl (全部 >=2 人的文档)
docs = [json.loads(l) for l in open(os.path.join(HERE, SRC), encoding="utf-8")]
def klass(d):
    if len(d["direct"]) < 2: return "少于2人"
    if d["n_views"] == 0: return "无其他视角"
    if d["conflict"]: return "有否定"
    return "一致(>=2人印证)" if sum(1 for n in d["direct"] if d["support"].get(n, 0) >= 1) >= 2 else "无否定但印证不足"
weak = collections.defaultdict(collections.Counter); cnt = collections.Counter()
for d in docs:
    k = klass(d); cnt[k] += 1
    for slot in ("F", "FF"):
        w = d["weak"].get(slot)
        if not w: continue
        pred = [n for n, p in d["direct"].items() if p == slot]
        weak[k][slot + "_n"] += 1; weak[k][slot + "_ok"] += any(same(n, w) or same(w, n) for n in pred); weak[k][slot + "_has"] += bool(pred)
gq = collections.defaultdict(collections.Counter)
for d in docs:
    if not d.get("gold"): continue
    k = klass(d); m = [(p, [g["path"] for g in d["gold"] if same(nm, g["name"])]) for nm, p in d["direct"].items()]; m = [(p, g) for p, g in m if g]
    gq[k]["n"] += len(m); gq[k]["ok"] += sum(p in g for p, g in m)
cons = [d for d in docs if klass(d) == "一致(>=2人印证)"]; pool = [d for d in docs if len(d["direct"]) >= 2]
rng = random.Random(0); rng.shuffle(cons); cons = cons[:CAP]; rng.shuffle(pool); rand = pool[:len(cons)]
for name, sel in ((PRE + "cons", cons), (PRE + "rand", rand), (PRE + "all", pool)):
    with open(os.path.join(HERE, "sft", name + ".jsonl"), "w", encoding="utf-8") as f:
        for d in sel:
            f.write(json.dumps(dict(id=d["id"], root=d["root"], g_root=d["g_root"], gold=[], maxhop=0, prompt=d["prompt"],
                                    target="\n".join("%s | %s" % kv for kv in d["direct"].items())), ensure_ascii=False) + "\n")
S = ["无标注墓志 %d 篇的多视角伪标签分类: %s" % (len(docs), dict(cnt)), "选入 %scons %d 篇; %srand %d 篇 (对照); %sall %d 篇" % (PRE, len(cons), PRE, len(rand), PRE, len(pool)),
     "用我方表的 父亲/祖父 列 (独立于 CBDB) 检验模型给出的父名/祖父名:"]
for k in sorted(weak):
    w = weak[k]
    S.append("  %-16s 父: %d 篇有弱标签, 模型一致 %.1f%% (给出父名的 %.1f%%);  祖父: %d 篇, 一致 %.1f%%" % (
        k, w["F_n"], w["F_ok"] / max(1, w["F_n"]) * 100, w["F_has"] / max(1, w["F_n"]) * 100, w["FF_n"], w["FF_ok"] / max(1, w["FF_n"]) * 100))
if gq: S.append("带金标的那部分文档 (隐藏标签当作无标注用), 各类里对上金标者的路径正确率: " + "; ".join("%s %.1f%% (%d)" % (k, c["ok"] / max(1, c["n"]) * 100, c["n"]) for k, c in sorted(gq.items())))
open(os.path.join(HERE, "runs", "make_st_stats%s.txt" % ("" if PRE == "ST" else "_" + PRE)), "w", encoding="utf-8").write("\n".join(S)); print("\n".join(S))
