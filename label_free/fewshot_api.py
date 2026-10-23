# -*- coding: utf-8 -*-
"""3-shot 大模型基线: 在零样本提示前加 3 篇训练集墓志及其标准答案, 对测试集逐篇提问, 输出 pred 格式供 metrics 评测。
用法: LLM_MODEL=deepseek-v4-flash-0731 python fewshot_api.py --out ../M_模型/runs/APIFS/pred_test.jsonl [--k 3] [--threads 6]
"""
import os, sys, json, re, time, argparse, threading, concurrent.futures as cf
sys.stdout.reconfigure(encoding="utf-8")
from openai import OpenAI
HERE = os.path.dirname(os.path.abspath(__file__)); M = os.path.join(HERE, "..", "M_模型")
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--k", type=int, default=3); ap.add_argument("--threads", type=int, default=6); a = ap.parse_args()
MODEL = os.environ.get("LLM_MODEL", "deepseek-v4-flash-0731")
client = OpenAI(api_key=os.environ["LLM_API_KEY"], base_url=os.environ.get("LLM_BASE_URL", "https://chat-ai.academiccloud.de/v1"), timeout=180, max_retries=1)
train = [json.loads(l) for l in open(os.path.join(M, "sft", "G_train.jsonl"), encoding="utf-8")]
test = [json.loads(l) for l in open(os.path.join(M, "sft", "G_test.jsonl"), encoding="utf-8")]
# 示例: 取前 k 篇正文较短、至少 3 个亲属的训练文档
shots = [d for d in train if 600 < len(d["prompt"]) < 1400 and d["target"].count("\n") >= 2][:a.k]
head = shots[0]["prompt"][:shots[0]["prompt"].index("下面是一篇唐代墓志")] if "下面是一篇唐代墓志" in shots[0]["prompt"] else ""
def body(d): return d["prompt"]
examples = "\n\n".join("【示例 %d】\n%s\n答案:\n%s" % (i + 1, body(d), d["target"]) for i, d in enumerate(shots))
pat = re.compile(r"([一-鿿□]{1,6})\s*[|｜]\s*([A-Z]{1,8})")
_lock = threading.Lock(); _last = [0.0]
def ask(prompt):
    for k in range(5):
        with _lock:
            w = _last[0] + 1.1 - time.time()
            if w > 0: time.sleep(w)
            _last[0] = time.time()
        try:
            r = client.chat.completions.create(model=MODEL, temperature=0, max_tokens=1500, messages=[{"role": "user", "content": prompt}]); return r.choices[0].message.content or ""
        except Exception as e: time.sleep(5 * (k + 1)); err = str(e)
    return ""
done = {}
if os.path.exists(a.out):
    for l in open(a.out, encoding="utf-8"): d = json.loads(l); done[d["id"]] = d
todo = [d for d in test if d["id"] not in done]; print("test %d, done %d, todo %d, shots %d" % (len(test), len(done), len(todo), len(shots)), flush=True)
def one(d):
    prompt = "下面先给出 %d 个已标注的示例, 然后是需要你处理的墓志。请按示例的格式作答, 每行一个人: 姓名 | 关系码。\n\n%s\n\n【待处理】\n%s" % (len(shots), examples, body(d))
    raw = ask(prompt); pred = {}
    for n, c in pat.findall(raw):
        n = n.strip("*")
        if n != d["root"] and c: pred.setdefault(n, c)
    return {"id": d["id"], "pred": pred, "raw": raw}
os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, "a", encoding="utf-8") as f, cf.ThreadPoolExecutor(a.threads) as ex:
    for i, r in enumerate(ex.map(one, todo), 1):
        f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
        if i % 20 == 0: print("done %d/%d" % (i, len(todo)), flush=True)
print("FEWSHOT_DONE", flush=True)
