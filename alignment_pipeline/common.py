# -*- coding: utf-8 -*-
"""
公共配置与工具函数 —— 所有步骤脚本都 import 这里。

路径约定
  RAW_XLSX      我方原始表 (10,201 篇唐墓志, 每行一个墓主, 含"原文"列)
  EXT           外部数据根目录 test2graph/data/external
  CBDB_SQLITE   CBDB 全库 SQLite
  TBDB_DIR      Tackett 库目录 (accdb + 导出 csv)
  HUIBIAN_DIR   《汇编》语料与映射输出
  TRAIN_DIR     最终训练集
"""
import os, re, sqlite3
import pandas as pd
import zhconv

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))          # test2graph/
EXT = os.path.join(ROOT, "data", "external")
RAW_DIR = os.path.join(EXT, "0_墓志数据")                                       # 所有原始墓志/维基数据以这个目录为准
RAW_XLSX = os.path.join(RAW_DIR, "tang_0915_epitaphs.xlsx")                    # 唐墓志原始表 (10,201 行)
RAW_WIKI = os.path.join(RAW_DIR, "tang_0928_wiki_属性.xlsx")                    # 唐人物维基表 (2,027 行, 含人物原文)
CBDB_DIR = os.path.join(EXT, "1_史学家人工关系库", "CBDB")
TBDB_DIR = os.path.join(EXT, "1_史学家人工关系库", "Tackett_TBDB")
HUIBIAN_DIR = os.path.join(EXT, "4_唐代墓志汇编原文")
TRAIN_DIR = os.path.join(ROOT, "data", "训练集", "epinet_v0")
CBDB_SQLITE = os.path.join(CBDB_DIR, "cbdb_20260926.sqlite3")

# CBDB 里《汇编》两本书的 textid (TEXT_CODES 表)
TEXTID_ZHENGBIAN = 32038   # 唐代墓誌匯編:二卷
TEXTID_XUJI = 32039        # 唐代墓誌彙編續集
DY_TANG = 6                # DYNASTIES.c_dy = 6 → 唐

# ------------------------------------------------------------------ 字形/人名
t2s = lambda s: zhconv.convert(str(s), "zh-cn")     # 繁 → 简
t2t = lambda s: zhconv.convert(str(s), "zh-hant")   # 简 → 繁
FULLWIDTH = str.maketrans("０１２３４５６７８９", "0123456789")


def clean_name(s):
    """去空白/间隔号, 去掉 CBDB 人名后的括注: '李氏(崔夐妻)' → '李氏'"""
    return re.sub(r"[()（）].*$", "", re.sub(r"[\s·]", "", str(s)))


# ------------------------------------------------------------------ 年号与汇编编号
_conn = None


def cbdb():
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(CBDB_SQLITE)
    return _conn


def era_tables():
    """返回 (拼音→简体年号 dict, 简体年号集合), 来自 CBDB NIAN_HAO 表"""
    nh = pd.read_sql("select c_nianhao_chn, c_nianhao_pin from NIAN_HAO", cbdb())
    pin, eras = {}, set()
    for a, p in zip(nh.c_nianhao_chn, nh.c_nianhao_pin):
        if a:
            eras.add(t2s(a))
        if a and p:
            pin[str(p).replace(" ", "").lower()] = t2s(a)      # 同一拼音多次出现时取最后一个 (与 CBDB 唐代年号一致)
    return pin, eras


def code_from_cbdb_pages(pages, pin):
    """CBDB 出处页码 'Kaiyuan 383' / '(XJ)Kaiyuan140' → '开元383'; 解析失败返回 None"""
    m = re.match(r"\(?(XJ)?\)?\s*([A-Za-z]+)\s*(\d+)", str(pages))
    if not m:
        return None
    era = pin.get(m.group(2).lower())
    return era + str(int(m.group(3))) if era else None


def code_from_text_head(first_line, eras):
    """我方原文首行 '开元０７６' → '开元76'; 不是年号式编号返回 None"""
    m = re.match(r"\s*([\u4e00-\u9fff]{2,4})\s*([０-９0-9]{1,4})\s*$", str(first_line))
    if not m:
        return None
    era = t2s(m.group(1))
    return era + str(int(m.group(2).translate(FULLWIDTH))) if era in eras else None


# ------------------------------------------------------------------ 亲属路径
GENDER_OF = {"F": 1, "M": 0, "S": 1, "D": 0, "H": 1, "W": 0, "B": 1, "Z": 0}


def base_steps(rel):
    """把 CBDB 亲属码拆成基本步 (只含 F M S D H W B Z); 含 G-n / K+n / U 等无法拆的返回 None"""
    s = re.sub(r"[^FMSDHWBZ]", "", rel)
    return list(s) if s and len(s) == len(re.sub(r"[eyn+\-0-9()]", "", rel)) else None


def inverse_path(steps, g_head, g_tail):
    """head→tail 的基本步 → tail→head 的路径。中间节点性别由字母决定, 端点性别由参数给出 (1 男 / 0 女 / None 未知)"""
    genders = [g_head] + [GENDER_OF[s] for s in steps]
    if g_tail is not None:
        genders[-1] = g_tail
    out = []
    for i in range(len(steps) - 1, -1, -1):
        L, gprev = steps[i], genders[i]
        if gprev is None and L in "FMSDBZ":
            return None
        if L in "FM":   inv = "S" if gprev == 1 else "D"
        elif L in "SD": inv = "F" if gprev == 1 else "M"
        elif L == "H":  inv = "W"
        elif L == "W":  inv = "H"
        else:           inv = "B" if gprev == 1 else "Z"
        out.append(inv)
    return "".join(out)


def normalize_path(p):
    """FS → B, FFS → FB, FD → Z, MS → B ... 循环化简到不变"""
    if not p or p == "EGO":
        return p
    q = None
    while q != p:
        q = p
        p = re.sub(r"[FM]S(?![0-9+\-])", "B", p, count=1)
        p = re.sub(r"[FM]D(?![0-9+\-])", "Z", p, count=1)
    return p


# ------------------------------------------------------------------ 人名在文中定位
def name_in_text(name, text_simp):
    """name(繁体, 可带括注) 在简体文本中的位置 (start, end); 找不到返回 None。
    顺序: 全名 → 名(≥2字) → 单字名 + 称谓上下文 (讳/父/祖/考/妣/子/女/兄/弟/妻/夫人/字 后 4 字内)。
    'X氏' / 'X某' 这类无名者不定位。"""
    n = t2s(clean_name(name))
    if len(n) < 2 or n.endswith("氏") or n.endswith("某"):
        return None
    i = text_simp.find(n)
    if i >= 0:
        return (i, i + len(n))
    g = n[1:]
    if len(g) >= 2:
        i = text_simp.find(g)
        return (i, i + len(g)) if i >= 0 else None
    m = re.search(r"(?:讳|諱|父|祖|考|妣|子|女|兄|弟|妻|夫人|字)[^，。；、]{0,4}" + re.escape(g), text_simp)
    if m:
        i = m.end() - len(g)
        return (i, i + len(g))
    return None
