#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import glob
import numpy as np
import pandas as pd

TARGET_POINTS = 13500
POINTS_PER_CHUNK = 25
NUM_CHUNKS = TARGET_POINTS // POINTS_PER_CHUNK  # 540
HEADER_NAME = "Acoustic Emission in dB"         # 출력 CSV의 칼럼명

def read_acoustic_csv(path: str) -> np.ndarray:
    """
    Expt_k.csv를 읽어 값의 1차원 numpy 배열로 반환.
    - 파일 형식이 (인덱스, 값) 2컬럼이든, 값 1컬럼이든 모두 처리
    - 첫 행에 헤더가 있다고 가정
    """
    df = pd.read_csv(path)
    if df.shape[1] == 1:
        # 단일 컬럼(헤더가 값의 이름) 형태
        col = df.columns[0]
        y = df[col].to_numpy(dtype=float)
    else:
        # (인덱스, 값) 형태일 가능성 높음 → 마지막 컬럼을 값으로 사용
        y = df.iloc[:, -1].to_numpy(dtype=float)
    # NaN 제거/보정(필요 시 앞값으로 채움, 이후 남은 NaN은 0으로)
    if np.isnan(y).any():
        s = pd.Series(y)
        y = s.fillna(method="ffill").fillna(0.0).to_numpy()
    return y

def linear_upsample_to_at_least(y: np.ndarray, target_points: int) -> np.ndarray:
    """
    사용자 규칙:
    - 최소 정수 n을 찾아 (원본 길이)*n >= target_points
    - 각 구간을 n등분(선형보간)하여 확장
    - 표준 선형보간 길이는 (m-1)*n + 1 이므로 실제 길이가 target_points 미만이면 n을 늘려 보정
    """
    m = len(y)
    if m <= 1:
        # 데이터가 1개 이하이면 필요한 길이만큼 복제
        return np.full(target_points, y[0] if m == 1 else 0.0, dtype=float)

   
    n_user = math.ceil(target_points / m)
    # 표준 선형보간으로도 target_points를 보장하도록 보정
    n_interp = math.ceil((target_points - 1) / (m - 1))
    n = max(n_user, n_interp)

    t_old = np.arange(m, dtype=float)
    t_new = np.linspace(0.0, m - 1.0, (m - 1) * n + 1)
    y_new = np.interp(t_new, t_old, y)  # 1D 선형 보간

    # 길이가 target_points보다 크면 초과분을 앞에서부터 "짝수번째"를 버리며 줄이기
    excess = len(y_new) - target_points
    if excess > 0:
        keep_mask = np.ones(len(y_new), dtype=bool)
        removed = 0
        # "맨 앞에서부터 짝수번째 수" → 1,2,3,... 번호라고 보면 2,4,6,... → 0-based 인덱스에서는 1,3,5,...
        for idx in range(len(y_new)):
            if removed >= excess:
                break
            # 0-based idx를 1-based로 보면 (idx+1); 짝수번째는 (idx+1) % 2 == 0
            if (idx + 1) % 2 == 0:
                keep_mask[idx] = False
                removed += 1
        y_new = y_new[keep_mask]

    # 길이가 모자라면(극히 드물지만 방어) 마지막 값 반복으로 채우기
    if len(y_new) < target_points:
        pad = np.full(target_points - len(y_new), y_new[-1], dtype=float)
        y_new = np.concatenate([y_new, pad], axis=0)

    # 최종 길이 고정
    return y_new[:target_points]

def write_chunks(y: np.ndarray, out_dir: str, type_name: str, exp_idx: int):
    """
    길이 13,500의 시퀀스를 25포인트 × 540 조각으로 나눠서
    파일명 규칙: T1_전체인덱스_Expt_{exp_idx}_{조각내부인덱스}.csv
    - 전체인덱스 = (exp_idx-1)*540 + 조각내부인덱스
    - out_dir는 '.../splitted_100ms/T1/Acoustic' 수준이어야 함
    """
    os.makedirs(out_dir, exist_ok=True)
    assert len(y) == TARGET_POINTS
    for j in range(1, NUM_CHUNKS + 1):
        seg = y[(j - 1) * POINTS_PER_CHUNK : j * POINTS_PER_CHUNK]
        global_idx = (exp_idx - 1) * NUM_CHUNKS + j  # 1..6480
        fname = f"{type_name}_{global_idx:04d}_Expt_{exp_idx}_{j}.csv" if NUM_CHUNKS * 12 >= 10000 else f"{type_name}_{global_idx:03d}_Expt_{exp_idx}_{j}.csv"
        out_path = os.path.join(out_dir, fname)
        # 단일 컬럼, 헤더 포함, 인덱스 미포함
        pd.DataFrame({HEADER_NAME: seg}).to_csv(out_path, index=False)

def process_all(input_dir: str,
                output_root: str,
                type_name: str = "T1",
                acoustic_subdir: str = "Acoustic"):
    """
    input_dir 예:
      projects/tool_wear_havard/data/dataverse_files/Dataset/T1/Acoustic_Emission_Data
      └─ Expt_1.csv ... Expt_12.csv

    output_root 예:
      projects/tool_wear_havard/data/splitted_100ms
      → 실제 저장 경로: {output_root}/{type_name}/{acoustic_subdir}/T1_001_Expt_1_1.csv ...
    """
    out_dir = os.path.join(output_root, type_name, acoustic_subdir)
    files = []
    # Expt_1..Expt_12 순서 보장
    for k in range(1, 13):
        path = os.path.join(input_dir, f"Expt_{k}.csv")
        if os.path.isfile(path):
            files.append((k, path))

    if not files:
        raise FileNotFoundError(f"No Expt_k.csv files found under: {input_dir}")

    for exp_idx, path in files:
        print(f"[INFO] Processing Expt_{exp_idx}: {path}")
        y = read_acoustic_csv(path)
        y_up = linear_upsample_to_at_least(y, TARGET_POINTS)
        assert len(y_up) == TARGET_POINTS, f"Length mismatch: {len(y_up)}"
        write_chunks(y_up, out_dir, type_name, exp_idx)

    print(f"[DONE] Saved {len(files) * NUM_CHUNKS} CSV files to {out_dir}")

if __name__ == "__main__":
    
    INPUT_DIR = "projects/tool_wear_havard/data/dataverse_files/Dataset/T1/Acoustic_Emission_Data"
    OUTPUT_ROOT = "projects/tool_wear_havard/data/splitted_100ms"
    TYPE_NAME = "T1"         
    ACOUSTIC_DIRNAME = "Acoustic"

    process_all(INPUT_DIR, OUTPUT_ROOT, TYPE_NAME, ACOUSTIC_DIRNAME)
