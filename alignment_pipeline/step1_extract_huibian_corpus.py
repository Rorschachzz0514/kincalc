# -*- coding: utf-8 -*-
"""
Step 1  从我方原始表里抽出《唐代墓志汇编》格式的原文, 拆块, 去重

输入  RAW_XLSX (10,201 行, 每行一个墓主; 合葬墓一人一行, 原文重复)
输出  HUIBIAN_DIR/汇编文本_分块.jsonl   每行一篇:
        code        汇编编号 (简体年号 + 序号, 如 开元76)
        cover       盖文 (【盖】后面的内容; "失。" 表示盖佚)
        title       志题 (【志文】后第一行, 若 ≤40 字且含 志/铭/墓/记/碣/塔 且无标点)
        preface     序文 (志题之后到最后一个 "铭曰/词曰/其词曰..." 为止)
        inscription 铭文 (其后的韵文; 无则 None)
        text        序文+铭文 拼成的一整段 (去掉排版换行与空格; 后续所有 span 偏移都基于它)
        subjects    [{rid, name, gender, death_year}]  该篇的墓主 (合葬墓多人)
        rid, 编号   我方表的行号 / 编号列 (取第一个墓主的)

规则
  1. 只保留原文第一行是 "年号+数字" 的篇 (如 "开元０７６"); 年号集合取自 CBDB NIAN_HAO 表
  2. 排版: 原文是按书页硬换行的, 每行前有缩进空格 → 把 【志文】之后所有行 strip 后直接拼接
  3. 去重键 = (code, text 前 200 字): 同一篇原文在表里出现多次时只保留一篇, 墓主并入 subjects
"""
import os, re, sys, json
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
from common import RAW_XLSX, HUIBIAN_DIR, era_tables, code_from_text_head

INS_MARK = re.compile(r"(铭曰|其词曰|其辞曰|词曰|辞曰|铭云|乃为铭曰|为铭曰|系之铭曰)[：:]?")


def parse_blocks(raw, eras):
    lines = [l.rstrip() for l in str(raw).split("\n")]
    code = code_from_text_head(lines[0], eras) if lines else None
    if not code:
        return None
    body = "\n".join(lines[1:])
    cover = None
    mc = re.search(r"【盖】\s*(.*?)\s*(?=【志文】|$)", body, re.S)
    if mc:
        cover = re.sub(r"\s+", "", mc.group(1)) or None
    mz = re.search(r"【志文】\s*(.*)$", body, re.S)
    zl = [l.strip() for l in (mz.group(1) if mz else body).split("\n") if l.strip()]
    title = None
    if zl and len(zl[0]) <= 40 and re.search(r"志|铭|墓|记|碣|塔", zl[0]) and not re.search(r"[，。]", zl[0]):
        title = zl.pop(0)
    text = re.sub(r"[ 　]+", "", "".join(zl))
    ms = list(INS_MARK.finditer(text))
    preface, inscription = (text[:ms[-1].end()], text[ms[-1].end():]) if ms else (text, None)
    return dict(code=code, cover=cover, title=title, preface=preface, inscription=inscription, text=text)


def main():
    _, eras = era_tables()
    e = pd.read_excel(RAW_XLSX); e = e.loc[:, ~e.columns.duplicated()]; e["rid"] = range(len(e))
    docs, seen = [], {}
    for r in e.itertuples():
        b = parse_blocks(r.原文, eras)
        if not b:
            continue
        subj = dict(rid=r.rid, name=str(r.姓名), gender=str(r.性别), death_year=None if pd.isna(r.去世年) else float(r.去世年))
        k = (b["code"], b["text"][:200])
        if k in seen:
            seen[k]["subjects"].append(subj); continue
        b.update(rid=r.rid, 编号=str(r.编号), subjects=[subj]); seen[k] = b; docs.append(b)
    os.makedirs(HUIBIAN_DIR, exist_ok=True)
    with open(os.path.join(HUIBIAN_DIR, "汇编文本_分块.jsonl"), "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print("原始表 %d 行 → 带汇编编号 %d 行 → 去重后 %d 篇 (合葬/多墓主 %d 篇); 有志题 %d, 有铭文 %d"
          % (len(e), sum(len(d["subjects"]) for d in docs), len(docs), sum(len(d["subjects"]) > 1 for d in docs),
             sum(d["title"] is not None for d in docs), sum(d["inscription"] is not None for d in docs)))


if __name__ == "__main__":
    main()
