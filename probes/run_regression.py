# -*- coding: utf-8 -*-
"""run_regression.py：一键验收回归脚本（--fast 快速功能回归 / --full 含全量真实数据集）。"""
import argparse
import glob
import os
import py_compile
import subprocess
import sys
import tempfile
import time

import numpy as np

PROBES_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PROBES_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if PROBES_DIR not in sys.path:
    sys.path.insert(0, PROBES_DIR)


def test_compile():
    files = [
        "geomfit.py", "refine5.py", "weak_ring.py", "poni_out.py",
        "solver_core.py", "phase_fit.py", "auto_mask.py", "custom_phase.py",
        "cake_report.py", "batch_parallel.py", "multiframe_calib.py",
        "gui_solver.py", "run_regression.py",
    ]
    for fn in files:
        p = os.path.join(PROBES_DIR, fn)
        py_compile.compile(p, doraise=True)
    print(f"  [PASS] py_compile: {len(files)} 个文件编译通过")


def test_auto_mask():
    from auto_mask import build_auto_mask

    H, W = 256, 256
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.full((H, W), 120.0, dtype=np.float64)
    # 中心束流挡盘暗影 (r <= 14) + 孤立热像素
    img[((yy - 128) ** 2 + (xx - 128) ** 2) <= 14 ** 2] = 2.0
    hot_pts = [(40, 40), (60, 190), (200, 50), (210, 210), (90, 90)]
    for hy, hx in hot_pts:
        img[hy, hx] = 50000.0

    m, st = build_auto_mask(img, cx=128.0, cy=128.0)
    assert m.shape == (H, W) and m.dtype == bool
    assert st["n_bad"] >= len(hot_pts), f"热像素检出不足: {st}"
    assert st["n_beamstop"] >= 400, f"束挡检出不足: {st}"
    assert bool(m[128, 128]) and all(bool(m[hy, hx]) for hy, hx in hot_pts)
    print(f"  [PASS] auto_mask: n_bad={st['n_bad']}, n_beamstop={st['n_beamstop']}, n_total={st['n_total']}")


def test_custom_phase():
    import torchfai.phases as tp
    import solver_core as sc
    from custom_phase import load_and_register_phase, load_cif_file, load_d_file

    with tempfile.TemporaryDirectory() as td:
        d_path = os.path.join(td, "test_d_sample.D")
        with open(d_path, "w", encoding="utf-8") as f:
            f.write("# mock pyFAI .D\n3.1242 100\n2.7059 30\n1.9133 90\n1.6316 35\n")
        nm_d, d_arr, i_arr, _ = load_d_file(d_path, name="_TMP_D_TEST")
        assert nm_d == "_TMP_D_TEST" and len(d_arr) == 4
        assert abs(d_arr[0] - 3.1242) < 1e-4 and abs(i_arr[0] - 100.0) < 1e-4

        cif_path = os.path.join(td, "tmp_lab6.cif")
        with open(cif_path, "w", encoding="utf-8") as f:
            f.write(
                "data_lab6\n"
                "_cell_length_a 4.15682(5)\n"
                "_cell_length_b 4.15682(5)\n"
                "_cell_length_c 4.15682(5)\n"
                "_cell_angle_alpha 90\n"
                "_cell_angle_beta 90\n"
                "_cell_angle_gamma 90\n"
                "loop_\n"
                "_atom_site_label\n"
                "_atom_site_type_symbol\n"
                "_atom_site_fract_x\n"
                "_atom_site_fract_y\n"
                "_atom_site_fract_z\n"
                "La1 La 0.0 0.0 0.0\n"
            )
        _, dc, ic, _ = load_cif_file(cif_path, name="_TMP_LAB6")
        ref_d = tp.PHASES["LaB6"][0][:5]
        for x, y in zip(dc[:5], ref_d):
            assert abs(x - y) < 0.01, f"CIF d-spacing mismatch: {x} vs {y}"

        reg = load_and_register_phase(cif_path, name="_TMP_LAB6")
        try:
            assert reg == "_TMP_LAB6"
            assert "_TMP_LAB6" in tp.PHASES and "_TMP_LAB6" in tp.PHASE_NAMES
            qs = tp.phase_qs("_TMP_LAB6")
            assert len(qs) >= 5 and abs(qs[0] - 2.0 * np.pi / 4.15682) < 1e-3
        finally:
            tp.PHASES.pop("_TMP_LAB6", None)
            if "_TMP_LAB6" in tp.PHASE_NAMES:
                tp.PHASE_NAMES.remove("_TMP_LAB6")
            if "_TMP_LAB6" in sc.PHASE_NAMES:
                sc.PHASE_NAMES.remove("_TMP_LAB6")
    print("  [PASS] custom_phase: .D 解析 + .cif (LaB6 a=4.15682) 匹配误差 <0.01Å 与注册通过")


def test_cake_and_pilatus():
    import pyFAI.integrator.azimuthal as pyai_az
    from zoo_run import load_img
    from poni_out import tth_of
    from solver_core import solve_image
    from cake_report import compute_cake, export_diagnostic_figure

    # 1. 合成带倾角环验证 compute_cake 展平能力
    H, W = 384, 384
    px = 100e-6
    dist_m, poni1_m, poni2_m = 0.12, 192.5 * px, 192.5 * px
    rot1, rot2, lam = 0.015, -0.010, 0.50
    yy, xx = np.mgrid[0:H, 0:W]
    tth = tth_of((yy + 0.5) * px, (xx + 0.5) * px, dist_m, poni1_m, poni2_m, rot1, rot2)
    q_map = 4.0 * np.pi * np.sin(0.5 * tth) / lam
    syn = 20.0 + 400.0 * np.exp(-0.5 * ((q_map - 1.2) / 0.03) ** 2)
    cake = compute_cake(syn, dist_m, poni1_m, poni2_m, rot1=rot1, rot2=rot2,
                        px1=px, lam_A=lam, n_q=300, n_chi=90)
    pk_idx = np.nanargmax(cake["cake"], axis=1)
    q_peaks = cake["q_axis"][pk_idx]
    assert np.std(q_peaks) < 0.03, f"Cake 环展平不够直: std={np.std(q_peaks):.4f}"

    # 2. Pilatus2M.cbf + 2x2 diagnostic PNG export
    zdir = os.path.join(ROOT, "data", "real", "silx_zoo")
    p_img = load_img(os.path.join(zdir, "Pilatus2M.cbf"))
    ai = pyai_az.AzimuthalIntegrator.sload(os.path.join(zdir, "Pilatus2M.poni"))
    lam_A = ai.wavelength * 1e10
    px_m = float(ai.detector.pixel1)
    res = solve_image(p_img, px_m, lam=lam_A)
    assert res and res["phase"] == "CeO2" and res["m"] == 8
    D_mm = res["sol"][2] * res["px_m"] * 1000.0
    assert abs(D_mm - 799.942) < 0.5, f"Pilatus2M D_mm 异常: {D_mm}"
    assert res["confidence"]["tier"] == "HIGH"
    with tempfile.TemporaryDirectory() as td:
        png = os.path.join(td, "p2m_diag.png")
        export_diagnostic_figure(png, p_img, res, px_m)
        assert os.path.isfile(png) and os.path.getsize(png) > 10_000
    print(f"  [PASS] cake_report + Pilatus2M: CeO2 m={res['m']} D={D_mm:.3f}mm HIGH, cake_std={np.std(q_peaks):.4f}")


def test_batch_and_multiframe():
    from batch_parallel import solve_batch_parallel
    from multiframe_calib import calibrate_multiframe

    poni_path = glob.glob(os.path.join(ROOT, "data", "real", "alumina", "*.poni"))[0]
    txt = {k.strip(): v.strip() for k, v in (
        l.split(":", 1) for l in open(poni_path, encoding="utf-8")
        if ":" in l and not l.startswith("#")
    )}
    gt = float(txt["Distance"]) * 1000.0
    lam = float(txt["Wavelength"]) * 1e10
    px1, px2 = float(txt["PixelSize1"]), float(txt["PixelSize2"])

    frames = sorted(glob.glob(os.path.join(ROOT, "data", "real", "alumina", "al2o3_*.edf")))[:4]
    kw = dict(cascade=True, auto_center=False, refine_2d=False)
    bp_res = solve_batch_parallel(frames, px1, lam=lam, px2_m=px2,
                                  phases=["Al2O3"], n_workers=2, solve_kw=kw)
    assert len(bp_res) == 4 and all(r["ok"] and 110.0 <= r["D_mm"] <= 122.0 for r in bp_res), bp_res

    dark = os.path.join(ROOT, "data", "real", "alumina", "dark_0001.edf")
    frames5 = sorted(glob.glob(os.path.join(ROOT, "data", "real", "alumina", "al2o3_*.edf")))[:5]
    with tempfile.TemporaryDirectory() as td:
        poni_out_p = os.path.join(td, "mf_test.poni")
        mf = calibrate_multiframe(
            frames5 + [dark], px1, lam=lam, px2_m=px2,
            phases=["Al2O3"], out_poni=poni_out_p, n_workers=1, solve_kw=kw)
        assert mf["ok"], mf
        assert mf["n_dark"] == 1 and os.path.basename(mf["dark_files"][0]) == "dark_0001.edf"
        assert mf["n_kept"] >= 4
        assert abs(mf["dist_mm"] - gt) <= 0.5, f"多帧联合偏大: {mf['dist_mm']:.2f} vs GT {gt:.2f}"
        assert os.path.isfile(poni_out_p)
    print(f"  [PASS] batch_parallel (4/4, workers=2) + multiframe_calib: "
          f"dark_rejected={mf['n_dark']}, kept={mf['n_kept']}, D={mf['dist_mm']:.2f}mm (GT {gt:.2f}mm)")


def test_gui_headless():
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    code = (
        "import sys; sys.path.insert(0, r'" + ROOT + "'); sys.path.insert(0, r'" + PROBES_DIR + "')\n"
        "from PyQt5.QtWidgets import QApplication\n"
        "import gui_solver\n"
        "app = QApplication([])\n"
        "win = gui_solver.Win()\n"
        "for attr in ('b_automask', 'b_brush', 'b_add_phase', 'b_multiframe', 'b_diag_fig', 'ax_cake', 'cv_cake'):\n"
        "    assert hasattr(win, attr), f'Missing GUI attr: {attr}'\n"
        "assert win.tab_plot.count() >= 3\n"
        "win.close()\n"
        "print('GUI_OFFSCREEN_OK tabs=' + str(win.tab_plot.count()))\n"
    )
    cp = subprocess.run([sys.executable, "-X", "utf8", "-c", code],
                        env=env, capture_output=True, text=True, check=False)
    assert cp.returncode == 0, f"GUI offscreen failed:\n{cp.stdout}\n{cp.stderr}"
    print("  [PASS] gui_solver headless:", cp.stdout.strip().splitlines()[-1])


def run_full_benchmarks():
    cmds = [
        [sys.executable, "-X", "utf8", os.path.join(PROBES_DIR, "phase_fit.py"), "ceria"],
        [sys.executable, "-X", "utf8", os.path.join(PROBES_DIR, "phase_fit.py"), "alumina"],
        [sys.executable, "-X", "utf8", os.path.join(PROBES_DIR, "_cls_all.py")],
        [sys.executable, "-X", "utf8", os.path.join(PROBES_DIR, "_test_refine5_solver.py")],
    ]
    for c in cmds:
        print(f"\n>>> 运行完整基准: {' '.join(c[2:])}")
        cp = subprocess.run(c, cwd=ROOT, check=False)
        assert cp.returncode == 0, f"基准脚本返回非零: {c}"


def main():
    ap = argparse.ArgumentParser(description="pyfai-zero-shot 一键回归测试台")
    ap.add_argument("--fast", action="store_true", help="运行快速模块单元测试与核心冒烟检查（默认）")
    ap.add_argument("--full", action="store_true", help="加跑全量真实数据集回归（ceria / alumina / CLS / refine5）")
    args = ap.parse_args()

    t0 = time.perf_counter()
    print("=== pyfai-zero-shot 回归测试 ===")
    test_compile()
    test_auto_mask()
    test_custom_phase()
    test_cake_and_pilatus()
    test_batch_and_multiframe()
    test_gui_headless()
    if args.full:
        run_full_benchmarks()
    print(f"=== 全部通过 ({time.perf_counter() - t0:.1f}s) ===")


if __name__ == "__main__":
    main()
