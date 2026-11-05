# -*- coding: utf-8 -*-
"""R1 前置侦察：各数据集尺寸与默认档/升级档的池化倍率 s。"""
import sys
import glob

import numpy as np
import fabio

ROOT = r"D:\pyfai-project"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + r"\probes")
from torchfai.phases import phase_qs, PHASES  # noqa: E402

print("CeO2 rings:", len(phase_qs("CeO2")), "I_rel:", PHASES["CeO2"][1])
print("Al2O3 rings:", len(phase_qs("Al2O3")))


def s_of(shape):
    md = max(shape)
    sd = max(1, int(np.ceil(md / 520)))
    sl = max(1, int(np.ceil(md / np.ceil(md / 2.0))))
    return sd, sl


img = fabio.open(ROOT + r"\data\real\MX_ceria\ceria_150_1_0001.cbf").data
print(f"ceria_150: {img.shape} s_def,s_large={s_of(img.shape)}")

f = sorted(glob.glob(ROOT + r"\data\real\alumina\*.edf"))[0]
img = fabio.open(f).data
print(f"alumina {f.split(chr(92))[-1]}: {img.shape} s_def,s_large={s_of(img.shape)}")

Z = ROOT + r"\data\real\silx_zoo"
for name in ["Pilatus1M.edf", "Pilatus2M.cbf", "Pilatus6M.cbf", "Eiger4M.edf",
             "Eiger9M.h5", "Frelon2k.edf", "halfccd.edf", "Fairchild.edf"]:
    try:
        if name.endswith(".h5"):
            import h5py
            with h5py.File(Z + "\\" + name, "r") as fh:
                arr = [None]

                def _w(n, o):
                    if arr[0] is None and o.ndim >= 2 \
                            and n.split("/")[-1] == "data":
                        arr[0] = o[()]
                fh.visititems(_w)
                a = arr[0]
                a = a[0] if a.ndim == 3 else a
        else:
            a = fabio.open(Z + "\\" + name).data
        print(f"{name}: {a.shape} s_def,s_large={s_of(a.shape)}")
    except Exception as e:
        print(name, "ERR", repr(e)[:80])
a = fabio.open(Z + r"\LaB6_260210.mar3450").data
print(f"mar3450: {a.shape} s_def,s_large={s_of(a.shape)}")
