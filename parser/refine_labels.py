# -*- coding: utf-8 -*-
"""
标签修订: 人工库 (CBDB) 的标签既不完整 (漏标亲属) 也有错 (辈分差一代等)。
用交叉拟合的模型给训练集每篇打多视角伪标签 (run_xfit.sh), 再按"换根一致性"决定:
  补全 add:  模型列出、金标没有的人
               all  全部加入            cons 只加入"至少一个其他视角印证、且没有视角否定"的
  纠错 flip: 模型与金标路径不同、且被其他视角印证 (无否定) → 采用模型路径
输出三套训练集 (G 格式):
  sft/Aall_train.jsonl   金标 + 全部补全
  sft/Acons_train.jsonl  金标 + 一致性筛过的补全
  sft/Ref_train.jsonl    金标 + 一致性补全 + 一致性纠错
"""
import os, sys, json, collections
sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__)); DST = os.path.join(HERE, "sft")


def same(pred, gold):
    if pred == gold: return True
    g = gold[1:] if len(gold) >= 2 else gold
    return len(g) >= 1 and (pred == g or (len(pred) >= 2 and pred[1:] == g) or (len(g) >= 2 and pred.endswith(g)))


pl = {}
for f in ("runs/X0/xfit.jsonl", "runs/X1/xfit.jsonl"):
    for l in open(os.path.join(HERE, f), encoding="utf-8"):
        d = json.loads(l); pl[d["id"]] = d
SUF = sys.argv[1] if len(sys.argv) > 1 else ""                       # 输出文件后缀 (区分不同版本的一致性规则)
outs = {k: open(os.path.join(DST, "%s%s_train.jsonl" % (k, SUF)), "w", encoding="utf-8") for k in ("Aall", "Acons", "Ref")}
st = collections.Counter(); ex_add = []; ex_flip = []
for l in open(os.path.join(DST, "G_train.jsonl"), encoding="utf-8"):
    d = json.loads(l); p = pl.get(d["id"]); st["docs"] += 1
    gold = [tuple(x.split(" | ")) for x in d["target"].splitlines()]; st["gold"] += len(gold)
    text = d["prompt"][d["prompt"].index("墓志:\n") + 4:]
    pos = lambda n: (lambda i: i if i >= 0 else 10 ** 6)(text.find(n if text.find(n) >= 0 else n[1:] if len(n) > 1 else n))
    var = {k: list(gold) for k in outs}
    if p:
        for name, path in p["direct"].items():
            if not path or same(name, d["root"]) or same(d["root"], name) or len(name) < 2 or len(name) > 6: continue
            m = [i for i, (gn, gp) in enumerate(gold) if same(name, gn) or same(gn, name)]
            sup, con = p["support"].get(name, 0), p["conflict"].get(name, 0); cons = sup >= 1 and con == 0
            if not m:
                st["add_all"] += 1; var["Aall"].append((name, path))
                if cons:
                    st["add_cons"] += 1; var["Acons"].append((name, path)); var["Ref"].append((name, path))
                    if len(ex_add) < 12 and st["add_cons"] % 40 == 1: ex_add.append("%s: +%s|%s" % (d["id"], name, path))
            elif gold[m[0]][1] != path:
                st["disagree"] += 1
                if cons:
                    st["flip_cons"] += 1; i = m[0]
                    var["Ref"] = [(gn, path) if j == i else (gn, gp) for j, (gn, gp) in enumerate(var["Ref"])]
                    if len(ex_flip) < 14 and st["flip_cons"] % 8 == 1: ex_flip.append("%s: %s %s→%s" % (d["id"], gold[i][0], gold[i][1], path))
    for k, f in outs.items():
        rows = sorted(var[k], key=lambda x: pos(x[0])) if len(var[k]) != len(gold) else var[k]
        f.write(json.dumps(dict(d, target="\n".join("%s | %s" % r for r in rows)), ensure_ascii=False) + "\n")
S = ["标签修订统计 (训练集 %d 篇, 金标人物 %d)" % (st["docs"], st["gold"]),
     "模型多列出的人: %d (相当于金标的 %.1f%%); 其中通过换根一致性的 %d (%.1f%%)" % (st["add_all"], st["add_all"] / st["gold"] * 100, st["add_cons"], st["add_cons"] / max(1, st["add_all"]) * 100),
     "模型与金标路径不同: %d; 其中模型路径被其他视角印证且无否定 → 纠错 %d" % (st["disagree"], st["flip_cons"]),
     "补全样例: " + "; ".join(ex_add), "纠错样例: " + "; ".join(ex_flip)]
open(os.path.join(HERE, "runs", "refine_stats%s.txt" % SUF), "w", encoding="utf-8").write("\n".join(S)); print("\n".join(S))
