# -*- coding: utf-8 -*-
"""
N1 —— 语料级建网: 把每篇墓志的家谱树拼成一张人物网络。
每篇的树:  训练集 = 补全后的标签 (Aall_train);  dev/test = 模型预测 (Aall);  无标注 6,296 篇 = 多视角伪标签 (GR)。
跨篇合并:  两篇里同名 (非"某氏"类) 的两个人, 若两篇还共有至少一个别的亲属姓名 → 视为同一人
           (L1 在金标树上测得: 精确率 94.0%, 召回率 76.1%)。
边:        墓主→亲属 的关系码; 另外把"父指针"展开成基本边 (如同时有 F 和 FF, 则 FF 是 F 的父)。
可信度:    无标注部分每条边带换根一致性等级 (A 两个以上视角印证 / B 一个视角印证 / C 无覆盖 / D 被否定)。
输出: runs/network/nodes.csv, edges.csv, N1_stats.txt
"""
import os, sys, json, csv, collections, itertools
sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "runs", "network"); os.makedirs(OUT, exist_ok=True)
generic = lambda n: len(n) < 2 or n.endswith("氏") or n.endswith("某") or "□" in n or n.endswith("公") or n.endswith("君") or n.endswith("夫人")
rows = lambda t: [tuple(x.split(" | ")) for x in t.splitlines() if " | " in x]

docs = []                                                      # {id, root, src, people:[(name, path, tier)]}
for l in open(os.path.join(HERE, "sft", "Aall_train.jsonl"), encoding="utf-8"):
    d = json.loads(l); docs.append(dict(id=d["id"], root=d["root"], src="标注(补全)", people=[(n, p, "标注") for n, p in rows(d["target"])]))
for sp in ("dev", "test"):
    for l in open(os.path.join(HERE, "runs", "Aall", "pred_%s.jsonl" % sp), encoding="utf-8"):
        d = json.loads(l); root = None
        docs.append(dict(id=d["id"], root=None, src="预测", people=[(n, p, "预测") for n, p in d["pred"].items()]))
roots = {}
for sp in ("dev", "test"):
    for l in open(os.path.join(HERE, "sft", "G_%s.jsonl" % sp), encoding="utf-8"):
        d = json.loads(l); roots[d["id"]] = d["root"]
for d in docs:
    if d["root"] is None: d["root"] = roots[d["id"]]
for l in open(os.path.join(HERE, "runs", "GR", "pseudo_unl.jsonl"), encoding="utf-8"):
    d = json.loads(l); ppl = []
    for n, p in d["direct"].items():
        s, c = d["support"].get(n, 0), d["conflict"].get(n, 0)
        ppl.append((n, p, "D" if c else "A" if s >= 2 else "B" if s == 1 else "C"))
    docs.append(dict(id=d["id"], root=d["root"], src="无标注(伪标签)", people=ppl))

# 提及 = (文档序号, 人名); 墓主也是一个提及
mentions = []; by_name = collections.defaultdict(list)
for di, d in enumerate(docs):
    names = [d["root"]] + [n for n, p, t in d["people"]]
    d["names"] = {n for n in names if not generic(n)}
    for n in dict.fromkeys(names):
        mid = len(mentions); mentions.append((di, n)); by_name[n].append(mid)
par = list(range(len(mentions)))
def find(x):
    while par[x] != x: par[x] = par[par[x]]; x = par[x]
    return x
n_pairs = n_link = 0
for n, ms in by_name.items():
    if generic(n) or len(ms) < 2 or len(ms) > 60: continue
    for a, b in itertools.combinations(ms, 2):
        da, db = mentions[a][0], mentions[b][0]
        if da == db: continue
        n_pairs += 1
        if (docs[da]["names"] & docs[db]["names"]) - {n}: par[find(a)] = find(b); n_link += 1
pid = {};
for m in range(len(mentions)): pid.setdefault(find(m), len(pid))
node_of = {mentions[m]: pid[find(m)] for m in range(len(mentions))}
ndocs = collections.defaultdict(set); nname = {}
for m, (di, n) in enumerate(mentions): ndocs[node_of[(di, n)]].add(di); nname.setdefault(node_of[(di, n)], n)

edges = collections.Counter(); tier_cnt = collections.Counter(); prim = collections.Counter()
RANK = {"标注": 0, "预测": 1, "A": 2, "B": 3, "C": 4, "D": 5}; ETIER = {"标注": "L", "预测": "P", "A": "A", "B": "B", "C": "C", "D": "D"}
etier = {}; edocs = collections.defaultdict(list); pdocs = collections.defaultdict(list)   # release: best tier + source documents per edge
for di, d in enumerate(docs):
    ego = node_of[(di, d["root"])]; by_path = collections.defaultdict(list)
    for n, p, t in d["people"]:
        if not p: continue
        x = node_of[(di, n)]; edges[(ego, p, x)] += 1; tier_cnt[t] += 1; by_path[p].append(x)
        if (ego, p, x) not in etier or RANK[t] < RANK[etier[(ego, p, x)]]: etier[(ego, p, x)] = t
        edocs[(ego, p, x)].append(d["id"])
    for p, xs in by_path.items():                                # 父指针展开: 路径 p = q + 一步, 且 q 槽位只有一个人
        if len(p) == 1:
            for x in xs: prim[(ego, p, x)] += 1; pdocs[(ego, p, x)].append(d["id"])
        elif len(by_path.get(p[:-1], [])) == 1:
            for x in xs: prim[(by_path[p[:-1]][0], p[-1], x)] += 1; pdocs[(by_path[p[:-1]][0], p[-1], x)].append(d["id"])
with open(os.path.join(OUT, "nodes.csv"), "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f); w.writerow(["node", "name", "n_docs"]); [w.writerow([k, nname[k], len(v)]) for k, v in sorted(ndocs.items())]
with open(os.path.join(OUT, "edges.csv"), "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f); w.writerow(["head", "rel", "tail", "n_docs", "kind"])
    for (h, r, t), c in edges.items(): w.writerow([h, r, t, c, "ego_path"])
    for (h, r, t), c in prim.items(): w.writerow([h, r, t, c, "primitive"])
# release export: edges with confidence tier (L curated/completed label, P parser prediction on labelled dev/test, A-D re-rooting tiers on unlabelled epitaphs) and source documents
with open(os.path.join(OUT, "edges_release.csv"), "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f); w.writerow(["head", "rel", "tail", "kind", "tier", "n_docs", "source_docs"])
    for (h, r, t), c in edges.items(): w.writerow([h, r, t, "ego_path", ETIER[etier[(h, r, t)]], c, ";".join(dict.fromkeys(edocs[(h, r, t)]))])
    for (h, r, t), c in prim.items(): w.writerow([h, r, t, "primitive", "", c, ";".join(dict.fromkeys(pdocs[(h, r, t)]))])
with open(os.path.join(OUT, "nodes_release.csv"), "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f); w.writerow(["node", "name", "n_docs", "docs"]); [w.writerow([k, nname[k], len(v), ";".join(docs[di]["id"] for di in sorted(v))]) for k, v in sorted(ndocs.items())]
# 连通分量 (按基本边)
p2 = list(range(len(pid)))
def f2(x):
    while p2[x] != x: p2[x] = p2[p2[x]]; x = p2[x]
    return x
for (h, r, t) in edges: p2[f2(h)] = f2(t)
comp = collections.Counter(f2(x) for x in range(len(pid))); sizes = sorted(comp.values(), reverse=True)
multi = sum(1 for v in ndocs.values() if len(v) >= 2)
big = max(comp, key=comp.get); big_docs = {di for k, v in ndocs.items() if f2(k) == big for di in v}
S = ["N1 语料级人物网络",
     "文档 %d 篇 (标注补全 %d / 模型预测 %d / 无标注伪标签 %d)" % (len(docs), sum(d["src"].startswith("标注") for d in docs), sum(d["src"] == "预测" for d in docs), sum(d["src"].startswith("无标注") for d in docs)),
     "人物提及 %d → 合并后人物节点 %d; 同名候选对 %d, 其中判为同一人并合并 %d (%.1f%%)" % (len(mentions), len(pid), n_pairs, n_link, n_link / max(1, n_pairs) * 100),
     "出现在 >=2 篇墓志中的人物 %d (%.1f%%)" % (multi, multi / len(pid) * 100),
     "亲属关系边 (墓主→亲属, 去重) %d; 展开成基本边 (父/母/子/女/夫/妻/兄弟/姊妹) %d" % (len(edges), len(prim)),
     "无标注部分的边按可信度: A(>=2 视角印证) %d, B(1 个视角) %d, C(无覆盖) %d, D(被否定) %d" % (tier_cnt["A"], tier_cnt["B"], tier_cnt["C"], tier_cnt["D"]),
     "连通分量 %d 个; 最大的 10 个 (人数): %s; 最大分量覆盖 %d 篇墓志; 只含 1 篇墓志的家族占文档 %.1f%%" % (
         len(sizes), sizes[:10], len(big_docs), 0.0)]
# 一个家族 (分量) 覆盖几篇墓志
comp_docs = collections.defaultdict(set)
for k, v in ndocs.items(): comp_docs[f2(k)] |= v
dist = collections.Counter(min(len(v), 6) for v in comp_docs.values()); single = sum(1 for v in comp_docs.values() if len(v) == 1)
S[-1] = S[-1].replace("只含 1 篇墓志的家族占文档 0.0%", "家族覆盖的墓志篇数分布 (6=6+): %s; 独立成族的墓志占 %.1f%%" % (dict(sorted(dist.items())), single / len(docs) * 100))
open(os.path.join(OUT, "N1_stats.txt"), "w", encoding="utf-8").write("\n".join(S)); print("\n".join(S))
