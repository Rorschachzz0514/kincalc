# -*- coding: utf-8 -*-
"""
导出人工核对表 (在服务器上运行, 输出 CSV 到 runs/verify/):
  1_测试集候选.csv     测试集 182 篇的候选池: 人工库有但多数模型不同意 / 模型多报 / 人工库且多数同意(抽 100)
  2_跨篇推断.csv       170 对中 "人工树+代数" 与人工库直接关系不一致的 (另附全部)
  3_纠错建议.csv       修订流程提出的 102 条纠错 (旧路径 → 新路径)
  4_补全建议.csv       修订流程补全的人物抽样 100 条
  5_网络边抽样.csv     无标注墓志上的伪标签边, A/B/C/D 四级各 50 条
每行都带原文片段, 以及留给核对人填写的三列: 核对结论 / 更正关系码 / 备注
"""
import os, sys, json, csv, random, collections
sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "runs", "verify"); os.makedirs(OUT, exist_ok=True)
GLOSS = {"F": "父", "M": "母", "S": "子", "D": "女", "H": "夫", "W": "妻", "B": "兄弟", "Z": "姊妹", "P": "父母", "C": "子女", "E": "配偶", "G": "兄弟姊妹"}
NAMED = {"F": "父亲", "M": "母亲", "S": "儿子", "D": "女儿", "H": "丈夫", "W": "妻子", "B": "兄弟", "Z": "姊妹", "FF": "祖父", "FM": "祖母", "FFF": "曾祖父", "FFFF": "高祖父", "MF": "外祖父", "MM": "外祖母",
         "SS": "孙子", "SD": "孙女", "SSS": "曾孙", "DS": "外孙", "DH": "女婿", "SW": "儿媳", "WF": "岳父", "WM": "岳母", "HF": "公公(夫之父)", "HM": "婆婆(夫之母)", "FB": "伯叔", "FZ": "姑", "MB": "舅", "MZ": "姨",
         "BS": "侄", "BD": "侄女", "ZS": "外甥", "FBS": "堂兄弟", "HFF": "夫之祖父", "HFFF": "夫之曾祖", "WFF": "妻之祖父", "SWF": "亲家(儿媳之父)", "DHF": "亲家(女婿之父)", "BW": "嫂/弟媳", "ZH": "姊妹之夫"}
gloss = lambda p: NAMED.get(p) or "的".join(GLOSS.get(c, c) for c in p)
def same(pred, gold):
    if pred == gold: return True
    g = gold[1:] if len(gold) >= 2 else gold
    return len(g) >= 1 and (pred == g or (len(pred) >= 2 and pred[1:] == g) or (len(g) >= 2 and pred.endswith(g)))
def snippet(text, name, w=45):
    for key in (name, name[1:] if len(name) > 1 else name):
        i = text.find(key)
        if i >= 0: return ("…" if i > w else "") + text[max(0, i - w):i] + "【" + text[i:i + len(key)] + "】" + text[i + len(key):i + len(key) + w] + "…"
    return "(文中未找到此名)"
COLS_TAIL = ["核对结论(对/错/无此人/名字有误)", "更正关系码", "备注"]
rng = random.Random(0)

# ---------- 1 测试集候选
sp = "test"; cache = json.load(open(os.path.join(HERE, "runs", "judge_cache_%s_v2oss.json" % sp), encoding="utf-8"))
docs = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(HERE, "sft", "G_%s.jsonl" % sp), encoding="utf-8")}
text_of = {i: d["prompt"][d["prompt"].index("墓志:\n") + 4:] for i, d in docs.items()}
gold = {i: dict(tuple(x.split(" | ")) for x in d["target"].splitlines()) for i, d in docs.items()}
SYS = ["G", "Aall", "Ref2", "JOINT", "STcons", "SYN_full", "U_apiall"]; preds = {}
for s in SYS:
    f = os.path.join(HERE, "runs", s, "pred_%s.jsonl" % sp)
    if os.path.exists(f): preds[s] = {json.loads(l)["id"]: json.loads(l)["pred"] for l in open(f, encoding="utf-8")}
TERMS = set("父母祖考妣子女夫妻兄弟姊妹孙侄甥婿嗣君公氏某") | {"夫人", "府君", "墓主", "姓名", "祖父", "祖母", "曾祖", "高祖", "外祖", "嗣子", "长子", "次子", "季子", "长女", "次女", "伯父", "叔父"}
rows = []; unanimous = []
for i in docs:
    cands = collections.defaultdict(set)                       # (path, 人) -> 名字写法集合; 同一人不同写法合并
    srcs = collections.defaultdict(set)
    allc = [(n, p, "人工库") for n, p in gold[i].items()] + [(n, p, s) for s in preds for n, p in preds[s].get(i, {}).items() if p]
    for n, p, src in allc:
        if not n or n in TERMS or len(n) > 6: continue
        key = next((k for k in cands if k[0] == p and any(same(n, m) or same(m, n) for m in cands[k])), None)
        if key is None: key = (p, n)
        cands[key].add(n); srcs[key].add(src)
    for (p, _), names in cands.items():
        main = max(names, key=len); ing = "人工库" in srcs[(p, _)]; hit = len(srcs[(p, _)] - {"人工库"})
        v = next((cache["%s	%s	%s" % (i, n, p)] for n in sorted(names, key=len, reverse=True) if "%s	%s	%s" % (i, n, p) in cache), {})
        row = dict(序号=0, 文档=i, 墓主=docs[i]["root"], 候选人名=main, 名字其他写法="、".join(sorted(names - {main})), 关系码=p, 关系释义=gloss(p), 来源="人工库+模型" if ing and hit else "仅人工库" if ing else "仅模型",
                   同意的模型数="%d/%d" % (hit, len(preds)), 模型裁决=("对" if v.get("ok") else "错") if v else "未裁决", 裁决备注=v.get("note", ""), 原文片段=snippet(text_of[i], main))
        if ing and hit >= len(preds) / 2: unanimous.append(row)
        else: rows.append(row)
rng.shuffle(unanimous); rows += unanimous[:100]
for r in unanimous[:100]: r["来源"] += "(抽查)"
rows.sort(key=lambda r: (r["文档"], r["来源"]))
with open(os.path.join(OUT, "1_测试集候选.csv"), "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) + COLS_TAIL); w.writeheader()
    for k, r in enumerate(rows, 1): r["序号"] = k; w.writerow(r)
print("1 测试集候选", len(rows), "(其中一致抽查 100)")

# ---------- 2 跨篇推断
pairs = [json.loads(l) for l in open(os.path.join(HERE, "..", "..", "data", "训练集", "crossdoc_pairs.jsonl"), encoding="utf-8")]
with open(os.path.join(OUT, "2_跨篇推断.csv"), "w", encoding="utf-8-sig", newline="") as f:
    cols = ["序号", "需核对", "甲篇", "甲篇人物x", "乙篇", "乙篇人物y", "桥(两篇共有的人)", "人工库说x→y", "释义", "人工树+代数算出", "释义2", "甲篇原文", "乙篇原文"] + ["核对结论(人工库对/代数对/都错/无法判断)", "更正关系码", "备注"]
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
    for k, p in enumerate(pairs, 1):
        w.writerow({"序号": k, "需核对": "是" if p["algebra_from_gold_trees"] != p["gold"] else "", "甲篇": p["doc_a"], "甲篇人物x": p["x"], "乙篇": p["doc_b"], "乙篇人物y": p["y"], "桥(两篇共有的人)": p["bridge"],
                    "人工库说x→y": p["gold"], "释义": gloss(p["gold"]), "人工树+代数算出": p["algebra_from_gold_trees"], "释义2": gloss(p["algebra_from_gold_trees"] or ""), "甲篇原文": p["text_a"], "乙篇原文": p["text_b"]})
print("2 跨篇", len(pairs), "需核对", sum(1 for p in pairs if p["algebra_from_gold_trees"] != p["gold"]))

# ---------- 3/4 纠错与补全
G = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(HERE, "sft", "G_train.jsonl"), encoding="utf-8")}
R = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(HERE, "sft", "Ref2_train.jsonl"), encoding="utf-8")}
tc = json.load(open(os.path.join(HERE, "runs", "judge_cache_train_v2oss.json"), encoding="utf-8"))
rows_f, rows_a = [], []
for i in G:
    g = dict(tuple(x.split(" | ")) for x in G[i]["target"].splitlines()); r = dict(tuple(x.split(" | ")) for x in R[i]["target"].splitlines()); t = G[i]["prompt"][G[i]["prompt"].index("墓志:\n") + 4:]
    for n, p in r.items():
        if n in g and g[n] != p:
            rows_f.append(dict(文档=i, 墓主=G[i]["root"], 人名=n, 人工库原路径=g[n], 原释义=gloss(g[n]), 建议改为=p, 新释义=gloss(p), 模型裁决旧=("对" if tc.get("%s\t%s\t%s" % (i, n, g[n]), {}).get("ok") else "错") if "%s\t%s\t%s" % (i, n, g[n]) in tc else "未裁决",
                               模型裁决新=("对" if tc.get("%s\t%s\t%s" % (i, n, p), {}).get("ok") else "错") if "%s\t%s\t%s" % (i, n, p) in tc else "未裁决", 原文片段=snippet(t, n)))
        elif n not in g:
            rows_a.append(dict(文档=i, 墓主=G[i]["root"], 人名=n, 建议关系码=p, 释义=gloss(p), 模型裁决=("对" if tc.get("%s\t%s\t%s" % (i, n, p), {}).get("ok") else "错") if "%s\t%s\t%s" % (i, n, p) in tc else "未裁决", 原文片段=snippet(t, n)))
rng.shuffle(rows_a); rows_a = rows_a[:100]
for name, rs, tail in (("3_纠错建议.csv", rows_f, ["核对结论(原路径对/建议对/都错)", "更正关系码", "备注"]), ("4_补全建议.csv", rows_a, COLS_TAIL)):
    with open(os.path.join(OUT, name), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["序号"] + list(rs[0].keys()) + tail); w.writeheader()
        for k, r in enumerate(rs, 1): w.writerow({"序号": k, **r})
print("3 纠错", len(rows_f), "4 补全抽样", len(rows_a))

# ---------- 5 网络边抽样 (无标注墓志伪标签, 按可信度等级)
by = collections.defaultdict(list)
for l in open(os.path.join(HERE, "runs", "GR", "pseudo_unl.jsonl"), encoding="utf-8"):
    d = json.loads(l); t = d["prompt"][d["prompt"].index("墓志:\n") + 4:]
    for n, p in d["direct"].items():
        s, c = d["support"].get(n, 0), d["conflict"].get(n, 0); tier = "D" if c else "A" if s >= 2 else "B" if s == 1 else "C"
        by[tier].append(dict(文档=d["id"], 墓主=d["root"], 人名=n, 关系码=p, 释义=gloss(p), 可信度等级=tier, 印证视角数=s, 否定视角数=c, 原文片段=snippet(t, n)))
rows5 = []
for tier in "ABCD": rng.shuffle(by[tier]); rows5 += by[tier][:50]
with open(os.path.join(OUT, "5_网络边抽样.csv"), "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["序号"] + list(rows5[0].keys()) + COLS_TAIL); w.writeheader()
    for k, r in enumerate(rows5, 1): w.writerow({"序号": k, **r})
print("5 网络边抽样", len(rows5), {t: len(v) for t, v in by.items()})
# 测试集全文 (供核对人通读)
with open(os.path.join(OUT, "0_测试集原文.csv"), "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f); w.writerow(["文档", "墓主", "原文"]); [w.writerow([i, d["root"], text_of[i]]) for i, d in docs.items()]
print("EXPORT_DONE")
