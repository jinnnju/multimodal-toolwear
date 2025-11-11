#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import numpy as np
import pandas as pd

TARGET_POINTS = 25 * 200        # 5000 포인트
POINTS_PER_CHUNK = 25
NUM_CHUNKS = TARGET_POINTS // POINTS_PER_CHUNK  # 200
HEADER_NAME = "Acoustic Emission in dB"

def read_acoustic_csv(path: str) -> np.ndarray:
    df = pd.read_csv(path)
    if df.shape[1] == 1:
        y = df.iloc[:, 0].to_numpy(dtype=float)
    else:
        y = df.iloc[:, -1].to_numpy(dtype=float)
    if np.isnan(y).any():
        y = pd.Series(y).fillna(method="ffill").fillna(0.0).to_numpy()
    return y

def linear_upsample_to_at_least(y: np.ndarray, target_points: int) -> np.ndarray:
    m = len(y)
    if m <= 1:
        return np.full(target_points, y[0] if m == 1 else 0.0)

    n_user = math.ceil(target_points / m)
    n_interp = math.ceil((target_points - 1) / (m - 1))
    n = max(n_user, n_interp)

    t_old = np.arange(m, dtype=float)
    t_new = np.linspace(0.0, m - 1.0, (m - 1) * n + 1)
    y_new = np.interp(t_new, t_old, y)

    excess = len(y_new) - target_points
    if excess > 0:
        keep_mask = np.ones(len(y_new), dtype=bool)
        removed = 0
        for idx in range(len(y_new)):
            if removed >= excess:
                break
            if (idx + 1) % 2 == 0:  # 짝수 번째 제거
                keep_mask[idx] = False
                removed += 1
        y_new = y_new[keep_mask]

    if len(y_new) < target_points:
        y_new = np.concatenate([y_new, np.full(target_points - len(y_new), y_new[-1])])

    return y_new[:target_points]

def write_chunks(y: np.ndarray, out_dir: str, type_name: str, exp_idx: int):
    os.makedirs(out_dir, exist_ok=True)
    assert len(y) == TARGET_POINTS
    for j in range(1, NUM_CHUNKS + 1):
        seg = y[(j - 1) * POINTS_PER_CHUNK : j * POINTS_PER_CHUNK]
        global_idx = (exp_idx - 1) * NUM_CHUNKS + j
        fname = f"{type_name}_{global_idx:03d}_Expt_{exp_idx}_{j}.csv"
        out_path = os.path.join(out_dir, fname)
        pd.DataFrame({HEADER_NAME: seg}).to_csv(out_path, index=False)

def process_all(input_dir: str, output_root: str, type_name: str = "T4", acoustic_subdir: str = "Acoustic"):
    out_dir = os.path.join(output_root, type_name, acoustic_subdir)
    files = []
    for k in range(1, 13):
        path = os.path.join(input_dir, f"Expt_{k}.csv")
        if os.path.isfile(path):
            files.append((k, path))

    if not files:
        raise FileNotFoundError(f"No Expt_k.csv found in {input_dir}")

    for exp_idx, path in files:
        print(f"[INFO] Processing Expt_{exp_idx}")
        y = read_acoustic_csv(path)
        y_up = linear_upsample_to_at_least(y, TARGET_POINTS)
        write_chunks(y_up, out_dir, type_name, exp_idx)

    print(f"[DONE] Saved {len(files) * NUM_CHUNKS} CSV files to {out_dir}")

if __name__ == "__main__":
    INPUT_DIR = "projects/tool_wear_havard/data/dataverse_files/Dataset/T4/Acoustic_Emission_Data"
    OUTPUT_ROOT = "projects/tool_wear_havard/data/splitted_100ms"
    TYPE_NAME = "T4"
    ACOUSTIC_DIRNAME = "Acoustic"

    process_all(INPUT_DIR, OUTPUT_ROOT, TYPE_NAME, ACOUSTIC_DIRNAME)
