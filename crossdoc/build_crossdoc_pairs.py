# -*- coding: utf-8 -*-
"""
构造"跨篇亲属关系推断"基准 (数据准备, 本地运行; 读 CBDB sqlite)。
问题: 甲篇墓志里的人 x 与 乙篇墓志里的人 y 是什么关系? 没有任何一篇单独说出这个关系, 必须把两篇的家谱接起来推。
金标: CBDB KIN_DATA 里 x→y 的直接关系 (历史学者录入), 且 x、y 分别只出现在各自的篇里 (不是同一个人出现在两篇)。
每条样本还给出"桥": 两篇共有的人物 z (CBDB 同一 id)。
同时用金标家谱树 + 代数先算一遍 (上界): 若 compose(x→z 在甲篇, z→y 在乙篇) 与 CBDB 直接关系一致, 说明跨篇组合在原理上成立。
输出: data/训练集/crossdoc_pairs.jsonl, 实验/X_跨篇推理/pairs_stats.txt
"""
import os, sys, json, sqlite3, collections, itertools
sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "实验", "V_验证")); sys.path.insert(0, os.path.join(ROOT, "pipeline"))
from kin_algebra import core, normalize, reroot, inverse
from common import CBDB_SQLITE
SRC = os.path.join(ROOT, "data", "训练集", "epinet_v0")

docs = {}; where = collections.defaultdict(set)
for sp in ("train", "dev", "test"):
    for l in open(os.path.join(SRC, sp + ".jsonl"), encoding="utf-8"):
        d = json.loads(l)
        if d["ego"] is None: continue
        ego = next(n for n in d["nodes"] if n["id"] == d["ego"])
        ppl = {}
        for n in d["nodes"]:
            c = "" if n["id"] == d["ego"] else core(n.get("path_from_ego"))
            if c is None or not n["in_doc"]: continue
            ppl[n["id"]] = dict(name=n["name_simp"], path=normalize(c, ego["gender"]), g=n["gender"])
        docs[d["id"]] = dict(split=sp, root=ego["name_simp"], g_root=ego["gender"], text=d["text"], people=ppl)
        for pid in ppl: where[pid].add(d["id"])

con = sqlite3.connect(CBDB_SQLITE)
kin = collections.defaultdict(dict)
for h, t, rel in con.execute("select k.c_personid, k.c_kin_id, kc.c_kinrel from KIN_DATA k join KINSHIP_CODES kc on k.c_kin_code=kc.c_kincode"):
    c = core(rel)
    if c: kin[h][t] = c

pairs = []; st = collections.Counter()
shared = {p: ds for p, ds in where.items() if len(ds) >= 2}
doc_pairs = set()
for z, ds in shared.items():
    for a, b in itertools.combinations(sorted(ds), 2): doc_pairs.add((a, b, z))
st["doc_pairs_with_bridge"] = len({(a, b) for a, b, z in doc_pairs})
for a, b, z in sorted(doc_pairs):
    A, B = docs[a], docs[b]
    for x in A["people"]:
        if x in B["people"]: continue                           # x 不能同时出现在乙篇
        for y, rel in kin.get(x, {}).items():
            if y not in B["people"] or y in A["people"]: continue
            gold = normalize(rel, A["people"][x]["g"]); st["pairs"] += 1
            # 代数上界: x→z (甲篇, 以甲墓主坐标换根) ∘ z→y (乙篇)
            xz = reroot(A["people"][x]["path"], A["people"][z]["path"], A["g_root"], A["people"][x]["g"])
            zy = reroot(B["people"][z]["path"], B["people"][y]["path"], B["g_root"], B["people"][z]["g"])
            alg = normalize(xz + zy, A["people"][x]["g"]) if xz is not None and zy is not None else None
            ok = alg == gold; st["alg_ok"] += ok; st["alg_none"] += alg is None
            st["len_%d" % min(len(gold), 5)] += 1
            pairs.append(dict(doc_a=a, doc_b=b, split=B["split"] if A["split"] == "train" else A["split"], x=A["people"][x]["name"], x_id=x, x_g=A["people"][x]["g"], x_path_a=A["people"][x]["path"],
                              y=B["people"][y]["name"], y_id=y, y_path_b=B["people"][y]["path"], bridge=A["people"][z]["name"], bridge_id=z, z_path_a=A["people"][z]["path"], z_path_b=B["people"][z]["path"],
                              gold=gold, algebra_from_gold_trees=alg, root_a=A["root"], root_b=B["root"], g_root_a=A["g_root"], g_root_b=B["g_root"], text_a=A["text"], text_b=B["text"]))
# 去重 (同一 x,y 可能经不同桥出现多次): 保留第一条
seen = set(); uniq = []
for p in pairs:
    k = (p["x_id"], p["y_id"], p["doc_a"], p["doc_b"])
    if k in seen: continue
    seen.add(k); uniq.append(p)
os.makedirs(os.path.join(ROOT, "data", "训练集"), exist_ok=True)
with open(os.path.join(ROOT, "data", "训练集", "crossdoc_pairs.jsonl"), "w", encoding="utf-8") as f:
    for p in uniq: f.write(json.dumps(p, ensure_ascii=False) + "\n")
by_split = collections.Counter(p["split"] for p in uniq); L = collections.Counter(min(len(p["gold"]), 5) for p in uniq)
ego_pairs = sum(1 for p in uniq if p["x_path_a"] == "" and p["y_path_b"] == "")
S = ["跨篇亲属关系推断基准",
     "有共同人物的文档对 %d; CBDB 直接相连的跨篇人物对 %d (去重后 %d); 其中两边都是墓主的 %d" % (st["doc_pairs_with_bridge"], st["pairs"], len(uniq), ego_pairs),
     "按 split (含 dev/test 文档的归入 dev/test): %s" % dict(by_split),
     "金标关系长度分布 (5=5+): %s" % dict(sorted(L.items())),
     "用金标家谱树 + 代数经桥组合, 与 CBDB 直接关系一致: %d / %d (%.1f%%); 无法组合 %d" % (st["alg_ok"], st["pairs"], st["alg_ok"] / max(1, st["pairs"]) * 100, st["alg_none"]),
     "样例: " + "; ".join("%s(%s)→%s(%s) 金标 %s 经桥 %s 代数 %s" % (p["x"], p["doc_a"], p["y"], p["doc_b"], p["gold"], p["bridge"], p["algebra_from_gold_trees"]) for p in uniq[:6])]
open(os.path.join(HERE, "pairs_stats.txt"), "w", encoding="utf-8").write("\n".join(S)); print("\n".join(S))
