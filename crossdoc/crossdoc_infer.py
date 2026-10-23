# -*- coding: utf-8 -*-
"""
跨篇亲属关系推断: 甲篇的 x 和乙篇的 y 是什么关系?  (data/训练集/crossdoc_pairs.jsonl, 金标来自 CBDB 直接关系)
方法:
  A  墓主视角解析 + 代数:   两篇各以墓主为中心解析 → 找两篇共有的人 z 作桥 → 代数换根并拼接 x→z→y
  B  任意根解析 + 代数:     甲篇以 x 为中心解析得 x→z, 乙篇以 z 为中心解析得 z→y → 拼接 (需要会换中心的模型 GR)
  C  模型直接读两篇 (端到端): 把两篇接在一起, 以 x 为中心让微调模型列亲属, 取 y 的关系码
  D  零样本大模型读两篇:      API, 直接问 x 与 y 的关系码          (--api)
用法: python crossdoc_infer.py --name GR [--api]
输出: runs/crossdoc_<name>.txt
"""
import os, sys, json, argparse, collections
sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "实验", "V_验证")); M = os.path.join(ROOT, "实验", "M_模型")
from kin_algebra import normalize, reroot, BASE, GENDER_OF
ap = argparse.ArgumentParser(); ap.add_argument("--name", default="GR"); ap.add_argument("--api", action="store_true")
ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct"); ap.add_argument("--bs", type=int, default=6)
a = ap.parse_args()
pairs = [json.loads(l) for l in open(os.path.join(ROOT, "data", "训练集", "crossdoc_pairs.jsonl"), encoding="utf-8")]
tpl = json.loads(open(os.path.join(M, "sft", "G_train.jsonl"), encoding="utf-8").readline())["prompt"]
head = tpl[:tpl.index("「")]; mid = tpl[tpl.index("」"):tpl.index("墓志:\n") + 4]
prompt = lambda root, text: head + "「" + root + mid + text
clip = lambda t: t if len(t) <= 2400 else t[:1600] + "……" + t[-800:]


def same(p, g):
    if p == g: return True
    gg = g[1:] if len(g) >= 2 else g
    return len(gg) >= 1 and (p == gg or (len(p) >= 2 and p[1:] == gg) or (len(gg) >= 2 and p.endswith(gg)))


def parse_G(out, g):
    res = {}
    for ln in out.splitlines():
        p = [x.strip() for x in ln.split("|")]
        if len(p) == 2 and p[1] and set(p[1]) <= BASE and p[0] not in res: res[p[0]] = normalize(p[1], g)
    return res


def find(parse, name, root):
    if same(name, root) or same(root, name): return ""
    for k, p in parse.items():
        if same(k, name) or same(name, k): return p
    return None


res = {k: [] for k in "ABCD"}; cache = {}
if not a.api:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(a.base); tok.padding_side = "left"
    model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16, device_map={"": 0}), os.path.join(M, "runs", a.name)).eval()

    @torch.no_grad()
    def gen(prompts):
        need = [p for p in dict.fromkeys(prompts) if p not in cache]
        need.sort(key=len, reverse=True)
        for b in range(0, len(need), a.bs):
            ps = need[b:b + a.bs]
            txt = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in ps]
            x = tok(txt, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            y = model.generate(**x, max_new_tokens=400, do_sample=False)
            for p, o in zip(ps, y): cache[p] = tok.decode(o[x["input_ids"].shape[1]:], skip_special_tokens=True)
        return [cache[p] for p in prompts]

    # 第一轮: 两篇各以墓主为中心
    PA = [prompt(p["root_a"], clip(p["text_a"])) for p in pairs]; PB = [prompt(p["root_b"], clip(p["text_b"])) for p in pairs]
    OA = gen(PA); OB = gen(PB)
    parA = [parse_G(o, p["g_root_a"]) for o, p in zip(OA, pairs)]; parB = [parse_G(o, p["g_root_b"]) for o, p in zip(OB, pairs)]
    bridges = []
    for p, pa, pb in zip(pairs, parA, parB):
        namesA = {p["root_a"]: ""} | pa; namesB = {p["root_b"]: ""} | pb
        bs = [(na, namesA[na], nb, namesB[nb]) for na in namesA for nb in namesB if (same(na, nb) or same(nb, na)) and not na.endswith("氏") and len(na) >= 2 and na not in (p["x"],) and nb not in (p["y"],)]
        bridges.append(bs)
    # 方法 A
    for p, pa, pb, bs in zip(pairs, parA, parB, bridges):
        px, py = find(pa, p["x"], p["root_a"]), find(pb, p["y"], p["root_b"]); ans = None
        if px is not None and py is not None:
            votes = collections.Counter()
            for na, pza, nb, pzb in bs:
                gx = p["x_g"]; xz = reroot(px, pza, p["g_root_a"], gx); gz = GENDER_OF[pza[-1]] if pza else p["g_root_a"]
                zy = reroot(pzb, py, p["g_root_b"], gz)
                if xz is not None and zy is not None:
                    q = normalize(xz + zy, gx)
                    if q: votes[q] += 1
            if votes: ans = votes.most_common(1)[0][0]
        res["A"].append(ans)
    # 方法 B: 甲篇以 x 为中心, 乙篇以 z 为中心 (桥取方法 A 找到的第一个)
    PX = [prompt(p["x"], clip(p["text_a"])) for p in pairs]; OX = gen(PX)
    zs = [bs[0] if bs else None for bs in bridges]
    PZ = [prompt(z[2], clip(p["text_b"])) if z else "" for p, z in zip(pairs, zs)]; OZ = gen([q for q in PZ if q])
    oz_iter = iter(OZ)
    for p, ox, z in zip(pairs, OX, zs):
        ans = None
        if z:
            o = next(oz_iter); vx = parse_G(ox, p["x_g"]); gz = GENDER_OF[z[1][-1]] if z[1] else p["g_root_a"]; vz = parse_G(o, gz)
            pz = find(vx, z[0], p["x"]); py = find(vz, p["y"], z[2])
            if pz and py: ans = normalize(pz + py, p["x_g"]) or None
        res["B"].append(ans)
    # 方法 C: 两篇接在一起, 以 x 为中心
    PC = [prompt(p["x"], "〔甲篇〕" + clip(p["text_a"]) + "\n〔乙篇〕" + clip(p["text_b"])) for p in pairs]; OC = gen(PC)
    for p, o in zip(pairs, OC):
        v = parse_G(o, p["x_g"]); res["C"].append(find(v, p["y"], p["x"]) or None)
else:
    import time, threading
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["LLM_API_KEY"], base_url=os.environ.get("LLM_BASE_URL", "https://chat-ai.academiccloud.de/v1"), timeout=240, max_retries=1)
    MODEL = os.environ.get("LLM_MODEL", "openai-gpt-oss-120b"); lock = threading.Lock(); last = [0.0]
    Q = ("下面是两篇唐代墓志。甲篇里有一个人叫「%s」, 乙篇里有一个人叫「%s」。请根据两篇的内容推断「%s」相对「%s」是什么亲属关系, 用关系码回答。\n"
         "关系码从「%s」出发逐步读: F=父 M=母 S=子 D=女 H=夫 W=妻 B=兄弟 Z=姊妹。例: FF=祖父, SSS=曾孙, WF=岳父, DH=女婿, BS=侄。\n"
         "两篇可能有共同的人物, 需要你自己找出来并把关系接起来。只输出一行: 关系码 (若无法判断输出 无)。\n\n〔甲篇〕\n%s\n\n〔乙篇〕\n%s")
    def ask(q):
        for k in range(4):
            with lock:
                w = last[0] + 1.1 - time.time()
                if w > 0: time.sleep(w)
                last[0] = time.time()
            try:
                r = client.chat.completions.create(model=MODEL, temperature=0, max_tokens=4000, messages=[{"role": "user", "content": q}])
                if r.choices[0].message.content: return r.choices[0].message.content.strip()
            except Exception: time.sleep(5 * (k + 1))
        return ""
    import concurrent.futures as cf
    with cf.ThreadPoolExecutor(3) as ex:
        outs = list(ex.map(ask, [Q % (p["x"], p["y"], p["y"], p["x"], p["x"], clip(p["text_a"]), clip(p["text_b"])) for p in pairs]))
    for o in outs:
        code = "".join(ch for ch in o.splitlines()[-1] if ch in BASE) if o else ""
        res["D"].append(code or None)

ok_alg = sum(1 for p in pairs if p["algebra_from_gold_trees"] == p["gold"])
S = ["跨篇亲属关系推断 (%d 对; 金标 = CBDB 直接关系) —— 模型 %s%s" % (len(pairs), a.name, " + API 零样本" if a.api else ""),
     "金标家谱树 + 代数 (上界): %.1f%%" % (ok_alg / len(pairs) * 100), "方法            全部  准确率   (回答了的  其中正确)   dev+test 子集   按金标长度 1/2/3+"]
for k, name in (("A", "A 墓主解析+代数"), ("B", "B 任意根解析+代数"), ("C", "C 微调模型读两篇"), ("D", "D 零样本大模型读两篇")):
    r = res[k]
    if not r: continue
    ok = sum(1 for x, p in zip(r, pairs) if x == p["gold"]); ans = sum(1 for x in r if x)
    dt = [(x, p) for x, p in zip(r, pairs) if p["split"] != "train"]; okdt = sum(1 for x, p in dt if x == p["gold"])
    byL = []
    for L in (1, 2, 3):
        sub = [(x, p) for x, p in zip(r, pairs) if (min(len(p["gold"]), 3) == L)]
        byL.append("%.0f%%" % (sum(1 for x, p in sub if x == p["gold"]) / max(1, len(sub)) * 100))
    S.append("%-14s %4d  %5.1f%%   (%4d  %5.1f%%)   %5.1f%% (%d)   %s" % (name, len(r), ok / len(r) * 100, ans, ok / max(1, ans) * 100, okdt / max(1, len(dt)) * 100, len(dt), " / ".join(byL)))
os.makedirs(os.path.join(HERE, "runs"), exist_ok=True)
json.dump(res, open(os.path.join(HERE, "runs", "crossdoc_%s%s.json" % (a.name, "_api" if a.api else "")), "w", encoding="utf-8"), ensure_ascii=False)
os.makedirs(os.path.join(HERE, "runs"), exist_ok=True)
open(os.path.join(HERE, "runs", "crossdoc_%s%s.txt" % (a.name, "_api" if a.api else "")), "w", encoding="utf-8").write("\n".join(S)); print("\n".join(S))
