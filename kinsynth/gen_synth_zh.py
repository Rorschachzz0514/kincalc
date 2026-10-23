# -*- coding: utf-8 -*-
"""
KinSynth —— 由亲属代数驱动的合成墓志生成器: 随机生成一棵家谱, 再用墓志的惯用句式把它写成文言短文。
标签由构造保证正确 (包括"哪个儿子的孩子"这种分支信息), 任意根的标签都能精确算出 → 无限量、零人工的任意根训练数据。
用途: (1) 零人工标签的预训练; (2) 少标注场景先在合成数据上学"称谓→路径", 再用少量真实数据适应文风。
资源 (姓氏、名用字、官职、地名、年号) 从真实训练集里抽取, 只用来当词表, 不带任何标签。
用法: python gen_synth.py 6000   → 实验/M_模型/sft/SYN_train.jsonl (每篇 2 条: 墓主根 + 随机另一根)
"""
import os, sys, json, random, re, collections
sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "实验", "V_验证"))
from kin_algebra import normalize
import zhconv
N = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
rng = random.Random(42)
t2s = lambda s: zhconv.convert(s, "zh-cn")

# ---------- 词表 (来自真实数据, 仅作词表)
SRC = os.path.join(ROOT, "data", "训练集", "epinet_v0", "train.jsonl")
sur, given, office, place, eras = collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter()
for l in open(SRC, encoding="utf-8"):
    d = json.loads(l)
    for n in d["nodes"]:
        nm = n["name_simp"]
        if len(nm) >= 2 and not nm.endswith("氏"): sur[nm[0]] += 1; given[nm[1:]] += 1
        for o in n.get("offices", []): office[t2s(o)] += 1
        for o in n.get("origin", []): place[t2s(o)] += 1
    m = re.match(r"([一-鿿]{2})\d", d["code"] or "")
    if m: eras[m.group(1)] += 1
SUR = [s for s, c in sur.most_common(60)]; SURSET = {s for s, c in sur.most_common(200)}
GIVEN = [g for g, c in given.most_common(1500) if 1 <= len(g) <= 2 and "□" not in g and not (len(g) == 2 and g[0] in SURSET and c < 3)]   # 去掉复姓切错留下的"姓+名"
OFF = [o for o, c in office.most_common(200) if 2 <= len(o) <= 8]; PLACE = [p for p, c in place.most_common(80) if len(p) == 2]; ERA = [e for e, c in eras.most_common(30)]
PREFIX = ["唐故", "大唐故", "故"]; FILL = ["幼而岐嶷，长而温恭。", "器宇冲深，风神爽迈。", "克绍家声，不坠先业。", "门传诗礼，代有簪缨。", "廉慎自守，为时所称。", "好学不倦，博涉经史。",
                                        "履信思顺，立身以廉。", "居家以孝闻，莅官以清著。", "英华发外，令问昭彰。", "蕴德含章，卓然不群。"]
MING = ["郁郁佳城，岩岩玄室。", "松槚森森，泉扉寂寂。", "德音不朽，遗芳永传。", "哀哉令德，奄归泉壤。", "刻石纪德，用表幽宫。", "山川长在，日月不留。"]
NUM = "一二三四五六七八九十"


# ---------- 家谱
class P:
    def __init__(self, name, g): self.name, self.g, self.edges = name, g, []     # edges: (letter, other) 从自己出发一步
def link(a, L, b, Linv):
    a.edges.append((L, b)); b.edges.append((Linv, a))
def parent(child, father, mother=None):
    link(child, "F", father, "S" if child.g == 1 else "D")
    if mother: link(child, "M", mother, "S" if child.g == 1 else "D"); link(father, "W", mother, "H")
def sibling(a, b):
    link(a, "B" if b.g == 1 else "Z", b, "B" if a.g == 1 else "Z")
def path(src, dst):
    """树上最短路 (基本边字母串) → 规范形。按构造的树算, 分支信息精确。"""
    prev = {src: None}; q = [src]
    while q:
        u = q.pop(0)
        if u is dst: break
        for L, v in u.edges:
            if v not in prev: prev[v] = (u, L); q.append(v)
    if dst not in prev: return None
    s = []; v = dst
    while prev[v]: u, L = prev[v]; s.append(L); v = u
    return normalize("".join(reversed(s)), src.g)


def gen_doc(i):
    S = rng.choice(SUR); gn = lambda: rng.choice(GIVEN); off = lambda: rng.choice(OFF); female = rng.random() < 0.3
    ego = P(S + gn(), 0 if female else 1); people = [ego]; text = []; mentions = {}       # name -> (person, 显示名)
    def mention(p, shown=None): people.append(p) if p not in people else None; mentions[p.name] = (p, shown or p.name)
    hs = rng.choice([s for s in SUR if s != S]) if female else None                      # 女性墓主的夫家姓
    # 标题与开头
    if female:
        text.append("%s%s%s府君夫人%s氏墓志铭并序" % (rng.choice(PREFIX), off(), hs, S)); text.append("夫人%s氏，%s人也。" % (S, rng.choice(PLACE)))
    else:
        text.append("%s%s%s府君墓志铭并序" % (rng.choice(PREFIX), off(), S)); text.append("君讳%s，字%s，%s人也。" % (ego.name[1:], gn(), rng.choice(PLACE)))
    # 父系祖先
    f = P(S + gn(), 1); ff = P(S + gn(), 1); fff = P(S + gn(), 1); ffff = P(S + gn(), 1)
    parent(ego, f); parent(f, ff); parent(ff, fff); parent(fff, ffff)
    style = rng.random()
    if style < 0.45:
        parts = []
        if rng.random() < 0.25: parts.append("高祖%s，%s" % (ffff.name[1:], off())); mention(ffff, ffff.name[1:])
        parts.append("%s%s，%s" % (rng.choice(["曾祖", "曾祖讳", "皇曾祖讳", "曾王父"]), fff.name[1:], off())); mention(fff, fff.name[1:])
        parts.append("%s%s，%s" % (rng.choice(["祖", "祖讳", "皇祖讳", "大父", "王父"]), ff.name[1:], off())); mention(ff, ff.name[1:])
        parts.append("%s%s，%s" % (rng.choice(["父", "考", "皇考讳", "烈考", "先考讳"]), f.name[1:], off())); mention(f, f.name[1:])
        text.append("；".join(parts) + "。")
    elif style < 0.7:
        text.append("%s%s之曾孙，%s%s之孙，%s%s之%s也。" % (off(), fff.name[1:], off(), ff.name[1:], off(), f.name[1:], "第%s子" % rng.choice(NUM[:4]) if not female else "女"))
        mention(fff, fff.name[1:]); mention(ff, ff.name[1:]); mention(f, f.name[1:])
    else:
        text.append("祖%s，%s；父%s，%s。" % (ff.name[1:], off(), f.name[1:], off())); mention(ff, ff.name[1:]); mention(f, f.name[1:])
    if rng.random() < 0.3:                                                                 # 伯叔
        fb = P(S + gn(), 1); parent(fb, ff); sibling(fb, f); text.append("%s%s，%s。" % (rng.choice(["伯父", "叔父", "季父"]), fb.name[1:], off())); mention(fb, fb.name[1:])
    if rng.random() < 0.3:                                                                 # 母与外祖
        ms = rng.choice(SUR); m = P(ms + "氏", 0); mf = P(ms + gn(), 1); link(ego, "M", m, "S" if ego.g else "D"); link(f, "W", m, "H"); parent(m, mf)
        text.append("%s%s氏，%s%s之女也。" % (rng.choice(["母", "妣", "先妣"]), ms, off(), mf.name[1:])); mention(m); mention(mf, mf.name[1:])
    text.append(rng.choice(FILL) + rng.choice(FILL))
    # 配偶
    if female:
        h = P(hs + gn(), 1); link(ego, "H", h, "W")
        text.append("年十%s，归于%s氏%s，%s。" % (rng.choice("五六七八九"), hs, h.name[1:], off())); mention(h, h.name[1:])
        if rng.random() < 0.4:
            hf = P(hs + gn(), 1); parent(h, hf); text.append("%s讳%s，%s。" % (rng.choice(["舅", "夫之考", "君舅"]), hf.name[1:], off())); mention(hf, hf.name[1:])
        spouse = h
    else:
        ws = rng.choice(SUR); w = P(ws + "氏", 0); link(ego, "W", w, "H"); wf = P(ws + gn(), 1); parent(w, wf)
        if rng.random() < 0.8:
            text.append("夫人%s氏，%s%s之女也。" % (ws, off(), wf.name[1:])); mention(w); mention(wf, wf.name[1:])
            if rng.random() < 0.3:
                wff = P(ws + gn(), 1); parent(wf, wff); text.append("夫人之祖%s，%s。" % (wff.name[1:], off())); mention(wff, wff.name[1:])
        spouse = w
    # 子女
    cs = rng.choice([0, 1, 2, 3, 4]); kids = []
    for k in range(cs):
        c = P((hs if female else S) + gn(), 1 if rng.random() < 0.7 else 0); parent(c, spouse if female else ego, None)
        if female: link(ego, "S" if c.g else "D", c, "M")
        else: link(c, "M", spouse, "S" if c.g else "D")
        kids.append(c)
    for a_, b_ in zip(kids, kids[1:]): sibling(a_, b_)
    sons = [c for c in kids if c.g == 1]; daus = [c for c in kids if c.g == 0]
    if sons:
        if len(sons) == 1 and rng.random() < 0.5: text.append("%s%s，%s。" % (rng.choice(["嗣子", "有子", "子"]), sons[0].name[1:], off())); mention(sons[0], sons[0].name[1:])
        else:
            ords = ["长曰", "次曰", "季曰", "第四子"] if rng.random() < 0.6 else ["长子", "次子", "第三子", "第四子"]
            text.append("有子%s人：%s。" % (NUM[len(sons) - 1], "，".join("%s%s%s" % (ords[j], s_.name[1:], "，" + off() if rng.random() < 0.5 else "") for j, s_ in enumerate(sons))))
            for s_ in sons: mention(s_, s_.name[1:])
        if rng.random() < 0.3:
            ss = P(sons[0].name[0] + gn(), 1); parent(ss, sons[0]); text.append("孙%s，%s。" % (ss.name[1:], off())); mention(ss, ss.name[1:])
    if daus:
        segs = []
        for j, d_ in enumerate(daus):
            if rng.random() < 0.6:
                ds = rng.choice(SUR); dh = P(ds + gn(), 1); link(d_, "H", dh, "W"); segs.append("%s适%s氏%s，%s" % (["长", "次", "季"][j] if j < 3 else "幼", ds, dh.name[1:], off())); mention(dh, dh.name[1:])
            else: segs.append("%s曰%s" % (["长", "次", "季"][j] if j < 3 else "幼", d_.name[1:])); mention(d_, d_.name[1:])
        text.append("女%s人：%s。" % (NUM[len(daus) - 1], "；".join(segs)))
    # 兄弟
    if rng.random() < 0.35:
        b = P(S + gn(), 1); parent(b, f); sibling(b, ego); text.append("%s%s，%s。" % (rng.choice(["兄", "弟", "长兄", "季弟"]), b.name[1:], off())); mention(b, b.name[1:])
        if rng.random() < 0.3:
            bs = P(S + gn(), 1); parent(bs, b); text.append("%s%s，%s。" % (rng.choice(["侄", "从子", "犹子"]), bs.name[1:], off())); mention(bs, bs.name[1:])
    # 卒葬与铭
    text.append("以%s%s年%s月%s日终于%s之私第，春秋%s十有%s。" % (rng.choice(ERA), rng.choice(NUM[:9]), rng.choice(NUM[:9]), rng.choice(NUM[:9]), rng.choice(PLACE), rng.choice(NUM[3:8]), rng.choice(NUM[:9])))
    text.append("铭曰：" + rng.choice(MING) + rng.choice(MING))
    return ego, people, mentions, "".join(text)


tpl = json.loads(open(os.path.join(ROOT, "实验", "M_模型", "sft", "G_train.jsonl"), encoding="utf-8").readline())["prompt"]
head = tpl[:tpl.index("「")]; mid = tpl[tpl.index("」"):tpl.index("墓志:\n") + 4]
out = open(os.path.join(ROOT, "实验", "M_模型", "sft", "SYN_train.jsonl"), "w", encoding="utf-8"); stat = collections.Counter()
for i in range(N):
    ego, people, mentions, text = gen_doc(i)
    named = [(p, shown) for nm, (p, shown) in mentions.items() if p is not ego]
    def rows(root):
        rs = []
        for p, shown in named + ([(ego, ego.name)] if root is not ego else []):
            if p is root: continue
            r = path(root, p)
            if r: rs.append("%s | %s" % (p.name, r))
        return rs
    r0 = rows(ego); stat["docs"] += 1; stat["persons"] += len(r0)
    out.write(json.dumps(dict(id="SYN%d" % i, root=ego.name, g_root=ego.g, gold=[], maxhop=0, prompt=head + "「" + ego.name + mid + text, target="\n".join(r0)), ensure_ascii=False) + "\n")
    cands = [p for p, s in named if p.g is not None and not p.name.endswith("氏")]
    if cands:
        y = rng.choice(cands); r1 = rows(y)
        if len(r1) >= 2:
            stat["reroot"] += 1
            out.write(json.dumps(dict(id="SYN%d@%s" % (i, y.name), root=y.name, g_root=y.g, gold=[], maxhop=0, prompt=head + "「" + y.name + mid + text, target="\n".join(r1)), ensure_ascii=False) + "\n")
out.close()
print("合成文档 %d, 平均人物 %.1f, 另有换根样本 %d" % (stat["docs"], stat["persons"] / stat["docs"], stat["reroot"]))
for l in list(open(os.path.join(ROOT, "实验", "M_模型", "sft", "SYN_train.jsonl"), encoding="utf-8"))[:2]:
    d = json.loads(l); print(d["root"], "::", d["prompt"][d["prompt"].index("墓志:") + 4:][:300]); print(d["target"]); print()
