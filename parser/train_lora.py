# -*- coding: utf-8 -*-
"""
LoRA 微调 (单卡, bf16)。只在回答部分算损失。
用法: python train_lora.py --name G --train sft/G_train.jsonl [sft/R_train.jsonl ...] [--maxhop 2] [--frac 0.1] [--epochs 3]
  --maxhop k  只用"所有人物 <=k 跳"的文档 (组合泛化实验)
  --frac f    只用前 f 比例的文档 (少样本实验; 换根样本按同一批文档筛)
输出: runs/<name>/ (LoRA 权重 + train.log)
"""
import os, sys, json, math, random, argparse, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True); ap.add_argument("--train", nargs="+", required=True)
ap.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
ap.add_argument("--epochs", type=float, default=3); ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--accum", type=int, default=8); ap.add_argument("--maxlen", type=int, default=3072)
ap.add_argument("--init", default="")                                 # 从已有 LoRA 继续训练 (如先在合成数据上预训练)
ap.add_argument("--maxhop", type=int, default=0); ap.add_argument("--frac", type=float, default=1.0); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "runs", a.name); os.makedirs(OUT, exist_ok=True)
random.seed(a.seed); torch.manual_seed(a.seed)
log = open(os.path.join(OUT, "train.log"), "a", encoding="utf-8")
def P(*x):
    s = " ".join(str(i) for i in x); print(s, flush=True); log.write(s + "\n"); log.flush()

first = [json.loads(l) for l in open(os.path.join(HERE, a.train[0]), encoding="utf-8")]
if a.maxhop: first = [d for d in first if d["maxhop"] <= a.maxhop]
if a.frac < 1: first = first[:int(len(first) * a.frac)]
keep = {d["id"] for d in first}; data = list(first)
for f in a.train[1:]:
    rows = [json.loads(l) for l in open(os.path.join(HERE, f), encoding="utf-8")]
    if "R_train" in f: rows = [d for d in rows if d["id"].split("@")[0] in keep]      # 金标换根样本: 只保留对应文档在训练子集里的
    data += rows
P("args", vars(a)); P("train examples", len(data), "(first file %d)" % len(first))

tok = AutoTokenizer.from_pretrained(a.base)
model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16, device_map=("auto" if os.environ.get("DEVMAP") == "auto" else {"": 0}), max_memory=(eval(os.environ["MAXMEM"]) if os.environ.get("MAXMEM") else None))
model.gradient_checkpointing_enable(); model.enable_input_require_grads()
if a.init:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, os.path.join(HERE, a.init), is_trainable=True); P("init from", a.init)
else:
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))

def enc(d):
    p = tok.apply_chat_template([{"role": "user", "content": d["prompt"]}], tokenize=False, add_generation_prompt=True)
    pi = tok(p, add_special_tokens=False)["input_ids"]; ti = tok(d["target"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
    pi = pi[:a.maxlen - len(ti)]
    return torch.tensor([pi + ti]), torch.tensor([[-100] * len(pi) + ti])

params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
total = math.ceil(len(data) * a.epochs / a.accum); sch = get_cosine_schedule_with_warmup(opt, int(total * 0.05), total)
model.train(); step = 0; t0 = time.time(); run = 0.0; n_seen = 0; n_target = int(len(data) * a.epochs)
while n_seen < n_target:
    random.shuffle(data)
    for d in data:
        if n_seen >= n_target: break
        x, y = enc(d)
        dev0 = next(model.parameters()).device
        loss = model(input_ids=x.to(dev0), labels=y.to(dev0)).loss
        (loss / a.accum).backward(); run += loss.item(); n_seen += 1
        if n_seen % a.accum == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sch.step(); opt.zero_grad(); step += 1
            if step % 20 == 0:
                P("step %d/%d loss %.4f  %.0fs" % (step, total, run / (20 * a.accum), time.time() - t0)); run = 0.0
model.save_pretrained(OUT); P("TRAIN_DONE", "%.0fs" % (time.time() - t0))
