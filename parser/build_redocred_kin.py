# -*- coding: utf-8 -*-
"""
把 Re-DocRED (英文维基, 文档级) 的亲属子集转成与墓志相同的任务形式: 给定中心人物 → 列出亲属及其路径码。
亲属关系: P22 father, P25 mother, P40 child, P26 spouse, P3373 sibling。英文不分性别的用 C 子女 / E 配偶 / G 兄弟姊妹。
中心人物 = 亲属边最多的人物; 金标路径 = 有直接边用直接边, 否则沿亲属边最短路用代数组合
(路径中出现"子女→其父母"的拐弯时无法确定是本人还是配偶, 该人不计入金标)。
输出: sft/RD_{train,dev,test}.jsonl  (与 G 格式同结构; gold 带 aliases 供英文名匹配)
"""
import os, sys, json, collections
sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "V_验证"))
from kin_algebra import normalize, UP, DOWN
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(ROOT, "data", "external", "2_原文配结构化结果语料", "Re-DocRED", "data"); DST = os.path.join(HERE, "sft")
REL = {"P22": ("F", "C"), "P25": ("M", "C"), "P40": ("C", "P"), "P26": ("E", "E"), "P3373": ("G", "G")}   # (h→t, t→h)
PROMPT = ("Below is a document. Taking \"%s\" as the central person, list every person named in the document who is a family relative of the central person, "
          "with a relation code read step by step from the central person: F=father M=mother P=parent S=son D=daughter C=child H=husband W=wife E=spouse "
          "B=brother Z=sister G=sibling. Examples: grandfather=FF, parent's sibling=PG, child's spouse=CE, spouse's father=EF.\n"
          "One person per line: name | code\n\nDocument:\n%s")


def detok(sents):
    t = " ".join(" ".join(s) for s in sents)
    for a, b in ((" ,", ","), (" .", "."), (" 's", "'s"), ("( ", "("), (" )", ")"), (" ;", ";"), (" :", ":"), (" - ", "-")): t = t.replace(a, b)
    return t


S = []
for sp, fn in (("train", "train_revised.json"), ("dev", "dev_revised.json"), ("test", "test_revised.json")):
    docs = json.load(open(os.path.join(SRC, fn), encoding="utf-8")); st = collections.Counter(); hop = collections.Counter()
    with open(os.path.join(DST, "RD_%s.jsonl" % sp), "w", encoding="utf-8") as f:
        for di, d in enumerate(docs):
            adj = collections.defaultdict(dict)
            for l in d.get("labels", []):
                if l["r"] in REL and l["h"] != l["t"]:
                    a, b = REL[l["r"]]
                    # 有性别的码 (F/M) 优先于不分性别的 (P)
                    if adj[l["h"]].get(l["t"]) not in ("F", "M"): adj[l["h"]][l["t"]] = a
                    if adj[l["t"]].get(l["h"]) not in ("F", "M"): adj[l["t"]][l["h"]] = b
            if not adj: continue
            st["docs_with_kin"] += 1
            root = max(adj, key=lambda v: (len(adj[v]), -v))
            path = {root: ""}; q = [root]
            while q:
                u = q.pop(0)
                for v, r in adj[u].items():
                    if v in path: continue
                    pu = path[u]
                    if pu is None or (pu and pu[-1] in DOWN and r in UP): path[v] = None       # 子女→父母 拐弯: 不确定
                    else: path[v] = pu + r
                    q.append(v)
            for v, r in adj[root].items(): path[v] = r                                           # 直接边优先
            gold = []
            for v, p in path.items():
                if v == root or not p: continue
                p = normalize(p)
                if not p: continue
                names = [m["name"] for m in d["vertexSet"][v]]; main = max(set(names), key=lambda n: (names.count(n), len(n)))
                first = min((m["sent_id"], m["pos"][0]) for m in d["vertexSet"][v])
                gold.append(dict(name=main, aliases=sorted(set(names)), path=p, pos=first))
            if len(gold) < 2: st["too_small"] += 1; continue
            gold.sort(key=lambda g: g["pos"])
            rn = [m["name"] for m in d["vertexSet"][root]]; rname = max(set(rn), key=lambda n: (rn.count(n), len(n)))
            for g in gold: hop[min(len(g["path"]), 4)] += 1; g.pop("pos")
            st["docs"] += 1; st["persons"] += len(gold)
            f.write(json.dumps(dict(id="RD_%s_%d" % (sp, di), root=rname, g_root=None, gold=gold, maxhop=max(len(g["path"]) for g in gold),
                                    prompt=PROMPT % (rname, detok(d["sents"])), target="\n".join("%s | %s" % (g["name"], g["path"]) for g in gold)), ensure_ascii=False) + "\n")
    S.append("RD %s: 含亲属边的文档 %d, 保留(>=2 个亲属) %d, 人物 %d (每篇 %.1f), 跳数分布 %s" % (sp, st["docs_with_kin"], st["docs"], st["persons"], st["persons"] / max(1, st["docs"]), dict(sorted(hop.items()))))
open(os.path.join(DST, "stats_RD.txt"), "w", encoding="utf-8").write("\n".join(S)); print("\n".join(S))
