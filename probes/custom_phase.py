# -*- coding: utf-8 -*-
"""custom_phase.py：支持直接导入 .D（pyFAI d-spacing 列表）与 .cif 晶体文件并注册到相库。"""
import os
import re

import numpy as np

# 常见元素原子序数（用于纯 numpy 结构因子近似相对强度权重）
_ELEM_Z = {
    "H": 1, "HE": 2, "LI": 3, "BE": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "NE": 10,
    "NA": 11, "MG": 12, "AL": 13, "SI": 14, "P": 15, "S": 16, "CL": 17, "AR": 18, "K": 19, "CA": 20,
    "SC": 21, "TI": 22, "V": 23, "CR": 24, "MN": 25, "FE": 26, "CO": 27, "NI": 28, "CU": 29, "ZN": 30,
    "GA": 31, "GE": 32, "AS": 33, "SE": 34, "BR": 35, "KR": 36, "RB": 37, "SR": 38, "Y": 39, "ZR": 40,
    "NB": 41, "MO": 42, "TC": 43, "RU": 44, "RH": 45, "PD": 46, "AG": 47, "CD": 48, "IN": 49, "SN": 50,
    "SB": 51, "TE": 52, "I": 53, "XE": 54, "CS": 55, "BA": 56, "LA": 57, "CE": 58, "PR": 59, "ND": 60,
    "PM": 61, "SM": 62, "EU": 63, "GD": 64, "TB": 65, "DY": 66, "HO": 67, "ER": 68, "TM": 69, "YB": 70,
    "LU": 71, "HF": 72, "TA": 73, "W": 74, "RE": 75, "OS": 76, "IR": 77, "PT": 78, "AU": 79, "HG": 80,
    "TL": 81, "PB": 82, "BI": 83, "TH": 90, "PA": 91, "U": 92,
}

_NUM_RE = re.compile(r"^([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)")


def _parse_cif_float(tok):
    m = _NUM_RE.match(str(tok).strip())
    if not m:
        raise ValueError(f"无法解析浮点数: {tok!r}")
    return float(m.group(1))


def _strip_elem(sym):
    s = re.sub(r"[^A-Za-z]", "", str(sym or "")).upper()
    if not s:
        return "SI"
    if s[:2] in _ELEM_Z:
        return s[:2]
    if s[:1] in _ELEM_Z:
        return s[:1]
    return "SI"


def load_d_file(path, name=None):
    """读取 pyFAI .D 文件（首列 d_A，可选第二列 I_rel），返回 (name, d_arr, i_arr, desc)。"""
    ds, ints = [], []
    has_i = True
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            try:
                d_val = _parse_cif_float(parts[0])
            except Exception:
                continue
            if d_val <= 0.0:
                continue
            ds.append(d_val)
            if len(parts) >= 2:
                try:
                    ints.append(max(float(_parse_cif_float(parts[1])), 0.0))
                except Exception:
                    has_i = False
            else:
                has_i = False
    if not ds:
        raise ValueError(f".D 文件未解析到有效 d-spacing: {path}")
    d_arr = np.asarray(ds, dtype=np.float64)
    order = np.argsort(-d_arr)
    d_arr = d_arr[order]
    if has_i and len(ints) == len(ds) and max(ints) > 0:
        i_arr = np.asarray(ints, dtype=np.float64)[order]
        i_arr = 100.0 * i_arr / float(i_arr.max())
    else:
        i_arr = 100.0 / (1.0 + 0.3 * np.arange(len(d_arr), dtype=np.float64))
    ph_name = (name or os.path.splitext(os.path.basename(path))[0]).strip()
    return ph_name, d_arr, i_arr, f"自定义 .D 标样 ({os.path.basename(path)})"


def _tokenize_cif(text):
    tokens = []
    n = len(text)
    i = 0
    while i < n:
        ch = text[i]
        if ch in " \t\r\n":
            i += 1
            continue
        if ch == "#":
            while i < n and text[i] != "\n":
                i += 1
            continue
        if ch == ";" and (i == 0 or text[i - 1] in "\r\n"):
            end = text.find("\n;", i + 1)
            if end < 0:
                tokens.append(text[i + 1:].strip())
                break
            tokens.append(text[i + 1:end].strip())
            i = end + 2
            continue
        if ch in ("'", '"'):
            q = ch
            i += 1
            start = i
            while i < n:
                if text[i] == q and (i + 1 == n or text[i + 1] in " \t\r\n"):
                    break
                i += 1
            tokens.append(text[start:i])
            i += 1
            continue
        start = i
        while i < n and text[i] not in " \t\r\n":
            i += 1
        tokens.append(text[start:i])
    return tokens


def _parse_symop_expr(expr):
    """把 'x+1/2, -y, 1/4+z' 解析成 (3,3) 矩阵与 (3,) 平移向量。"""
    parts = [p.strip().lower().replace(" ", "") for p in expr.split(",")]
    if len(parts) != 3:
        raise ValueError(f"无效对称操作: {expr!r}")
    mat = np.zeros((3, 3), dtype=np.float64)
    vec = np.zeros(3, dtype=np.float64)
    term_re = re.compile(r"([+-]?(?:(?:\d+/\d+|\d*\.?\d*)?\*?[xyz]|(?:\d+/\d+|\d+\.?\d*)))")
    axis_map = {"x": 0, "y": 1, "z": 2}
    for row, p in enumerate(parts):
        idx = 0
        for m in term_re.finditer(p):
            t = m.group(1)
            if not t:
                continue
            idx = m.end()
            if t[-1] in axis_map:
                col = axis_map[t[-1]]
                coef_s = t[:-1].rstrip("*")
                if coef_s in ("", "+"):
                    c = 1.0
                elif coef_s == "-":
                    c = -1.0
                elif "/" in coef_s:
                    num, den = coef_s.split("/", 1)
                    c = float(num) / float(den)
                else:
                    c = float(coef_s)
                mat[row, col] += c
            else:
                if "/" in t:
                    num, den = t.split("/", 1)
                    vec[row] += float(num) / float(den)
                else:
                    vec[row] += float(t)
        if idx == 0 and p:
            raise ValueError(f"无法解析对称项: {p!r}")
    return mat, vec


def load_cif_file(path, name=None, d_min=1.0, max_peaks=30):
    """纯 stdlib+numpy 解析 .cif 并生成衍射粉晶 (d_arr, i_arr)。"""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        tokens = _tokenize_cif(f.read())

    kv = {}
    loops = []
    i = 0
    nt = len(tokens)
    while i < nt:
        tok = tokens[i]
        tl = tok.lower()
        if tl == "loop_":
            i += 1
            cols = []
            while i < nt and tokens[i].startswith("_"):
                cols.append(tokens[i].lower())
                i += 1
            rows = []
            nc = len(cols)
            if nc > 0:
                while i + nc <= nt and not tokens[i].startswith("_") and tokens[i].lower() != "loop_" and not tokens[i].lower().startswith("data_"):
                    rows.append(tokens[i:i + nc])
                    i += nc
            loops.append((cols, rows))
        elif tl.startswith("_"):
            if i + 1 < nt and not tokens[i + 1].startswith("_"):
                kv[tl] = tokens[i + 1]
                i += 2
            else:
                i += 1
        else:
            i += 1

    a = _parse_cif_float(kv["_cell_length_a"])
    b = _parse_cif_float(kv["_cell_length_b"])
    c = _parse_cif_float(kv["_cell_length_c"])
    alpha = np.deg2rad(_parse_cif_float(kv.get("_cell_angle_alpha", 90.0)))
    beta = np.deg2rad(_parse_cif_float(kv.get("_cell_angle_beta", 90.0)))
    gamma = np.deg2rad(_parse_cif_float(kv.get("_cell_angle_gamma", 90.0)))

    symops = []
    atoms_raw = []
    for cols, rows in loops:
        sym_col = None
        for key in ("_space_group_symop_operation_xyz", "_symmetry_equiv_pos_as_xyz"):
            if key in cols:
                sym_col = cols.index(key)
                break
        if sym_col is not None:
            for r in rows:
                try:
                    symops.append(_parse_symop_expr(r[sym_col]))
                except Exception:
                    pass
        if ("_atom_site_fract_x" in cols and "_atom_site_fract_y" in cols
                and "_atom_site_fract_z" in cols):
            ix = cols.index("_atom_site_fract_x")
            iy = cols.index("_atom_site_fract_y")
            iz = cols.index("_atom_site_fract_z")
            isym = cols.index("_atom_site_type_symbol") if "_atom_site_type_symbol" in cols else None
            ilbl = cols.index("_atom_site_label") if "_atom_site_label" in cols else None
            iocc = cols.index("_atom_site_occupancy") if "_atom_site_occupancy" in cols else None
            for r in rows:
                try:
                    fx = _parse_cif_float(r[ix])
                    fy = _parse_cif_float(r[iy])
                    fz = _parse_cif_float(r[iz])
                    occ = _parse_cif_float(r[iocc]) if iocc is not None else 1.0
                    el = _strip_elem(r[isym] if isym is not None else (r[ilbl] if ilbl is not None else "Si"))
                    atoms_raw.append((np.array([fx, fy, fz], dtype=np.float64), float(_ELEM_Z.get(el, 14)) * occ))
                except Exception:
                    pass

    if not symops:
        symops = [(np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64))]
    if not atoms_raw:
        atoms_raw = [(np.zeros(3, dtype=np.float64), 14.0)]

    # 展开对称等效原子坐标并按模 1 去重
    atoms = []
    for xyz, zw in atoms_raw:
        seen = []
        for mat, vec in symops:
            p = np.mod(mat @ xyz + vec, 1.0)
            p[np.isclose(p, 1.0, atol=1e-4)] = 0.0
            is_dup = False
            for sp in seen:
                diff = np.abs(p - sp)
                diff = np.minimum(diff, 1.0 - diff)
                if np.all(diff < 1e-3):
                    is_dup = True
                    break
            if not is_dup:
                seen.append(p)
                atoms.append((p, zw))

    ca, cb, cg = np.cos(alpha), np.cos(beta), np.cos(gamma)
    G = np.array([
        [a * a, a * b * cg, a * c * cb],
        [a * b * cg, b * b, b * c * ca],
        [a * c * cb, b * c * ca, c * c],
    ], dtype=np.float64)
    G_star = np.linalg.inv(G)

    d_use = float(d_min)
    hm = int(np.floor(a / d_use)) + 1
    km = int(np.floor(b / d_use)) + 1
    lm = int(np.floor(c / d_use)) + 1
    while (2 * hm + 1) * (2 * km + 1) * (2 * lm + 1) > 800_000:
        d_use *= 1.15
        hm = int(np.floor(a / d_use)) + 1
        km = int(np.floor(b / d_use)) + 1
        lm = int(np.floor(c / d_use)) + 1

    hh, kk, ll = np.mgrid[-hm:hm + 1, -km:km + 1, -lm:lm + 1]
    hkl = np.stack([hh.ravel(), kk.ravel(), ll.ravel()], axis=1).astype(np.float64)
    nonzero = np.any(hkl != 0, axis=1)
    hkl = hkl[nonzero]
    s2 = np.sum((hkl @ G_star) * hkl, axis=1)
    s2_max = 1.0 / (d_use * d_use)
    keep = (s2 > 1e-12) & (s2 <= s2_max)
    hkl = hkl[keep]
    d_vals = 1.0 / np.sqrt(s2[keep])

    coords = np.array([p for p, _ in atoms], dtype=np.float64)   # (N, 3)
    weights = np.array([zw for _, zw in atoms], dtype=np.float64)  # (N,)
    phase = 2.0 * np.pi * (hkl @ coords.T)
    re_f = np.cos(phase) @ weights
    im_f = np.sin(phase) @ weights
    i_vals = re_f * re_f + im_f * im_f

    max_i = float(i_vals.max()) if len(i_vals) else 0.0
    if max_i <= 0.0:
        raise ValueError(f"CIF 计算不出有效衍射峰: {path}")
    pos = i_vals > 1e-6 * max_i
    d_vals = d_vals[pos]
    i_vals = i_vals[pos]

    order = np.argsort(-d_vals)
    d_sorted = d_vals[order]
    i_sorted = i_vals[order]

    # 按 0.15% 相对 d-spacing 容差合并等效晶面
    merged_d, merged_i = [], []
    cur_d_sum, cur_w_sum, cur_i_sum, cur_d0 = 0.0, 0.0, 0.0, None
    for dv, iv in zip(d_sorted, i_sorted):
        if cur_d0 is None or abs(dv - cur_d0) / cur_d0 <= 0.0015:
            if cur_d0 is None:
                cur_d0 = dv
            cur_d_sum += dv * iv
            cur_w_sum += iv
            cur_i_sum += iv
        else:
            merged_d.append(cur_d_sum / max(cur_w_sum, 1e-12))
            merged_i.append(cur_i_sum)
            cur_d0 = dv
            cur_d_sum = dv * iv
            cur_w_sum = iv
            cur_i_sum = iv
    if cur_d0 is not None:
        merged_d.append(cur_d_sum / max(cur_w_sum, 1e-12))
        merged_i.append(cur_i_sum)

    d_out = np.asarray(merged_d, dtype=np.float64)
    i_out = np.asarray(merged_i, dtype=np.float64)
    i_out = 100.0 * i_out / float(i_out.max())
    keep_strong = i_out >= 0.5
    if keep_strong.sum() >= 3:
        d_out = d_out[keep_strong]
        i_out = i_out[keep_strong]
    if len(d_out) > int(max_peaks):
        d_out = d_out[:int(max_peaks)]
        i_out = i_out[:int(max_peaks)]

    raw_name = (name
                or kv.get("_pd_phase_name")
                or kv.get("_chemical_name_common")
                or kv.get("_chemical_formula_sum")
                or os.path.splitext(os.path.basename(path))[0])
    ph_name = str(raw_name).strip().replace(" ", "") or os.path.splitext(os.path.basename(path))[0]
    return ph_name, d_out, i_out, f"自定义 .cif ({os.path.basename(path)})"


def register_custom_phase(name, d_arr, i_arr=None, desc="", harmonic=False):
    """注册自定义标样到 torchfai.phases.PHASES 与 PHASE_NAMES，并返回名称。"""
    import torchfai.phases as _tp
    import solver_core as _sc

    d_clean = np.asarray(d_arr, dtype=np.float64).ravel()
    if len(d_clean) == 0:
        raise ValueError("d_arr 不能为空")
    order = np.argsort(-d_clean)
    d_clean = d_clean[order]
    if i_arr is None:
        i_clean = 100.0 / (1.0 + 0.3 * np.arange(len(d_clean), dtype=np.float64))
    else:
        i_clean = np.asarray(i_arr, dtype=np.float64).ravel()[order]
        if i_clean.max() > 0:
            i_clean = 100.0 * i_clean / float(i_clean.max())

    ph_name = str(name).strip()
    _tp.PHASES[ph_name] = (d_clean, i_clean, desc or f"自定义标样 {ph_name}")
    if ph_name not in _tp.PHASE_NAMES:
        _tp.PHASE_NAMES.append(ph_name)
    if ph_name not in _sc.PHASE_NAMES:
        _sc.PHASE_NAMES.append(ph_name)
    if harmonic:
        _tp.HARMONIC_PHASES = _tp.HARMONIC_PHASES | frozenset({ph_name})
    return ph_name


def load_and_register_phase(path, name=None, d_min=1.0, max_peaks=30):
    """自动识别 .D / .cif 文件并注册到相库，返回相名称。"""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".cif":
        ph_name, d_arr, i_arr, desc = load_cif_file(
            path, name=name, d_min=d_min, max_peaks=max_peaks)
    else:
        ph_name, d_arr, i_arr, desc = load_d_file(path, name=name)
    return register_custom_phase(ph_name, d_arr, i_arr, desc=desc)
