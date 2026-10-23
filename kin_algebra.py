# -*- coding: utf-8 -*-
"""
亲属代数 (kinship calculus) —— 8 个生成元: F 父 M 母 S 子 D 女 H 夫 W 妻 B 兄弟 Z 姊妹。
一条路径 = 从起点出发逐步走的生成元串, 'WF' = 起点的妻的父。
提供: core(去修饰) / normalize(规范形) / inverse(逆) / compose(组合) / reroot(换根)。
规范形取"主读法": 例如 父的子 → 兄弟 (另一读法是本人), 妻的子 → 子 (另一读法是继子)。
"""
import re

GENDER_OF = {"F": 1, "M": 0, "S": 1, "D": 0, "H": 1, "W": 0, "B": 1, "Z": 0,
             "P": None, "C": None, "E": None, "G": None}      # 性别未知的生成元: P 父母 C 子女 E 配偶 G 兄弟姊妹 (英文 child/spouse/sibling)
BASE = set(GENDER_OF)
UP, DOWN, SP, SIB = "FMP", "SDC", "HWE", "BZG"
_SIB_OF = {"S": "B", "D": "Z", "C": "G"}; _CHILD_OF = {"B": "S", "Z": "D", "G": "C"}


def core(rel):
    """CBDB 亲属码 → 生成元串; 去掉长幼/排行修饰 (+ - e y 数字); 含 G-n/K/U 等不可分解的返回 None; 'EGO' → ''"""
    if rel is None:
        return None
    if rel == "EGO":
        return ""
    c = re.sub(r"[+\-0-9ey()]", "", rel)
    return c if c and set(c) <= set("FMSDHWBZ") else None


def _reduce_pair(x, y, gprev):
    """相邻两步 x y 的化简。返回 None(不化简) 或 替换串('' 表示回到原节点)。gprev = x 之前那个节点的性别"""
    if x in UP:
        if y in DOWN: return _SIB_OF[y]                         # 父母的子女 → 兄弟姊妹
        if y in SP:                                             # 父之妻 → 母, 母之夫 → 父
            if x == "F" and y in "WE": return "M"
            if x == "M" and y in "HE": return "F"
            if x == "P": return {"W": "M", "H": "F", "E": "P"}[y]
    elif x in DOWN:
        if y in UP:                                             # 子女的父/母 → 本人或配偶
            if gprev is None or y == "P": return None
            if (gprev == 1) == (y == "F"): return ""
            return "W" if y == "M" else "H"
        if y in SIB: return _CHILD_OF[y]                        # 子女的兄弟姊妹 → 子女
    elif x in SIB:
        if y in UP: return y                                    # 兄弟的父 → 父
        if y in SIB: return y                                   # 兄弟的兄弟 → 兄弟
    elif x in SP:
        if y in SP and not (x == y and x != "E"): return ""     # 夫之妻 → 本人
        if y in DOWN: return y                                  # 配偶的子女 → 子女
    return None


def normalize(path, g0=None):
    """化简到规范形。g0 = 起点性别 (1 男 / 0 女 / None 未知)"""
    p = list(path)
    changed = True
    while changed:
        changed = False
        g = [g0] + [GENDER_OF[s] for s in p]                    # g[i] = 第 i 步之前的节点性别
        for i in range(len(p) - 1):
            r = _reduce_pair(p[i], p[i + 1], g[i])
            if r is not None:
                p[i:i + 2] = list(r); changed = True
                break
    return "".join(p)


def inverse(path, g_head, g_tail=None):
    """head→tail 的路径 → tail→head 的路径; 中间节点性别由字母决定, 性别未知时用不分性别的生成元 (P/C/G)"""
    genders = [g_head] + [GENDER_OF[s] for s in path]
    out = []
    for i in range(len(path) - 1, -1, -1):
        L, gp = path[i], genders[i]
        if L in SP: out.append({"H": "W", "W": "H", "E": "E"}[L] if gp is None else ("H" if gp == 1 else "W"))
        elif L in UP:   out.append("C" if gp is None else ("S" if gp == 1 else "D"))
        elif L in DOWN: out.append("P" if gp is None else ("F" if gp == 1 else "M"))
        else:           out.append("G" if gp is None else ("B" if gp == 1 else "Z"))
    return "".join(out)


def compose(p1, p2, g0=None):
    """a→b 为 p1, b→c 为 p2 ⇒ a→c"""
    return normalize(p1 + p2, g0)


def reroot(p_new_root, p_x, g_ego, g_new_root=None):
    """已知 ego→y 为 p_new_root, ego→x 为 p_x ⇒ y→x 的路径 (规范形); 不可求返回 None"""
    inv = inverse(p_new_root, g_ego, g_new_root)
    if inv is None:
        return None
    g_y = g_new_root if g_new_root is not None else (GENDER_OF[p_new_root[-1]] if p_new_root else g_ego)
    return normalize(inv + p_x, g_y)


def view_consistent(ypath, q_direct, p_view, g_ego, g_y=None):
    """多视角一致性: 墓主视角说 ego→x = q_direct, 以 y (ego→y = ypath) 为中心的视角说 y→x = p_view。
    把墓主视角的结论换算到 y 的视角再比 (这个方向是确定的; 反方向会遇到"父之子 = 本人还是兄弟"的歧义)。"""
    g_y = GENDER_OF[ypath[-1]] if g_y is None and ypath else g_y
    return reroot(ypath, q_direct, g_ego, g_y) == normalize(p_view, g_y)


def view_to_ego(ypath, p_view, g_ego, g_y=None):
    """把 y 视角里的 y→x 换算成 ego→x 的候选 (按可能性排序)。若 p_view 以 "y→ego" 开头, 优先读作"经过墓主本人"。"""
    g_y = GENDER_OF[ypath[-1]] if g_y is None and ypath else g_y
    back = normalize(inverse(ypath, g_ego, g_y), g_y); p = normalize(p_view, g_y); out = []
    if back and p.startswith(back) and len(p) > len(back): out.append(normalize(p[len(back):], g_ego))
    q = normalize(ypath + p, g_ego)
    if q and q not in out: out.append(q)
    return out


if __name__ == "__main__":
    assert normalize("SFF", 1) == "F" and normalize("HS") == "S" and normalize("FD") == "Z"
    assert inverse("FF", 1) == "SS" and inverse("WF", 1) == "DH" and inverse("DH", 1) == "WF"
    assert reroot("F", "FF", 1) == "F" and reroot("W", "S", 1) == "S" and reroot("FF", "F", 1) == "S"
    assert reroot("S", "W", 1) == "M" and reroot("B", "S", 1) == "BS"
    assert inverse("F", None) == "C" and inverse("C", None) == "P" and normalize("PC") == "G" and normalize("EC") == "C" and normalize("GF") == "F"
    assert view_consistent("F", "S", "SS", 1) and view_consistent("F", "BS", "SS", 1) and not view_consistent("F", "S", "S", 1)
    assert view_to_ego("F", "SS", 1) == ["S", "BS"] and view_to_ego("W", "HF", 1)[0] == "F" and view_to_ego("F", "F", 1) == ["FF"]
    print("ok")
