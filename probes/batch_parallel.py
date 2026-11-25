# -*- coding: utf-8 -*-
"""batch_parallel.py：Windows spawn 安全的批量多进程衍射图并行求解。"""
from concurrent.futures import ProcessPoolExecutor
import os

import numpy as np

from solver_core import auto_offset, solve_image


def _load_img_headless(path):
    try:
        import fabio
        return np.asarray(fabio.open(path).data, dtype=np.float64)
    except Exception:
        from PIL import Image
        return np.asarray(Image.open(path), dtype=np.float64)


def _solve_one_worker(args):
    """顶层可 pickle worker（兼容 Windows multiprocessing spawn）。"""
    path, px_m, lam, px2_m, phases, solve_kw = args
    name = os.path.basename(path)
    try:
        img = auto_offset(_load_img_headless(path))
        res = solve_image(
            img, px_m, lam=lam, px2_m=px2_m, phases=phases, solve_kw=solve_kw)
        if not res:
            return dict(path=path, name=name, ok=False, error="无有效解")
        sol = res["sol"]
        s = float(res["s"])
        tilt_ref = res.get("tilt_ref")
        return dict(
            path=path,
            name=name,
            ok=True,
            phase=res.get("phase"),
            lam_A=float(res["lam"]),
            D_mm=float(sol[2]) * float(res["px_m"]) * 1000.0,
            cx=float(sol[0]) * s,
            cy=float(sol[1]) * s,
            m=int(res["m"]),
            rms=float(res["rms"]),
            ps=float(res["ps"]),
            native_2d=bool(res.get("native_2d", False)),
            tilt_applied=bool(isinstance(tilt_ref, dict) and tilt_ref.get("applied")),
            error=None,
        )
    except Exception as exc:
        return dict(path=path, name=name, ok=False, error=str(exc))


def solve_batch_parallel(paths, px_m, lam=None, px2_m=None, phases=None,
                         n_workers=None, solve_kw=None):
    """多帧并行批处理求解；单文件或 n_workers=1 时退化为串行以省去进程启动开销。"""
    path_list = list(paths)
    if not path_list:
        return []
    tasks = [(p, float(px_m), lam, px2_m, phases, solve_kw) for p in path_list]
    if n_workers is None:
        nw = min(os.cpu_count() or 2, 4, len(tasks))
    else:
        nw = max(1, int(n_workers))
    if nw == 1 or len(tasks) < 2:
        return [_solve_one_worker(t) for t in tasks]
    with ProcessPoolExecutor(max_workers=nw) as ex:
        return list(ex.map(_solve_one_worker, tasks))
