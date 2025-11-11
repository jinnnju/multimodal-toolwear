from torch.utils.data import Dataset, DataLoader, random_split
import os, re
import numpy as np
import pandas as pd
import torch

# ─────────────────────────────────────────────────────────────
# 공통 유틸
# ─────────────────────────────────────────────────────────────
_NUM_DTYPES = (np.float64, np.float32, np.int64, np.int32, float, int)

def _list_csv_map(folder: str) -> dict:
    """폴더 내 .csv 파일을 {basename_without_ext: fullpath} dict로 반환 (정렬 포함)."""
    if not os.path.isdir(folder):
        return {}
    files = [f for f in os.listdir(folder) if f.endswith('.csv')]
    files.sort()
    return {os.path.splitext(f)[0]: os.path.join(folder, f) for f in files}

def _normalize_modalities(modalities):
    """['acc','force','acoustic'] 등 대소문자 무시, 중복 제거. 존재하는 키만 반환."""
    valid = {'acc':'Accelerometer', 'force':'Force', 'acoustic':'Acoustic'}
    out = []
    for m in modalities:
        key = m.strip().lower()
        if key in valid and valid[key] not in out:
            out.append(valid[key])
    if not out:
        raise ValueError("modalities must contain at least one of: 'Acc', 'Force', 'Acoustic'")
    return out  # 예: ['Accelerometer','Force']

def _pick_common_keys(mod_maps: list[dict]) -> list[str]:
    """여러 modality 파일 맵의 교집합 키를 사전식 정렬로 반환. 하나만 있으면 그 키들."""
    present = [set(m.keys()) for m in mod_maps if len(m) > 0]
    if not present:
        return []
    keys = set.intersection(*present) if len(present) > 1 else list(present[0])
    keys = sorted(list(keys))
    return keys

# ─────────────────────────────────────────────────────────────
# FeatureExtractor (기존과 동일)
# ─────────────────────────────────────────────────────────────
class FeatureExtractor:
    def calculate_features(self, data):
        Z1 = np.mean(np.abs(data))  # (Mean Value)
        Z2 = np.sqrt(np.mean(data ** 2))  # RMS
        Z3 = np.std(data)  # Standard Deviation
        Z4 = Z2 / Z1  # Shape Factor
        Z5 = np.mean(((np.abs(data - Z1)) / Z3) ** 3)  # Skewness
        Z6 = np.mean(((np.abs(data - Z1)) / Z3) ** 4)  # Kurtosis
        Z7 = np.max(np.abs(data))  # Peak Value
        Z8 = Z7 / Z2  # Crest Factor
        Z9 = Z7 / Z1  # Impulse Factor
        spec = np.abs(np.fft.fft(data)) ** 2
        freqs = np.arange(len(spec))
        Z10 = np.sum((freqs ** 2) * spec)  # (MSF)
        Z11 = np.mean(spec)  # (MPS)
        Z12 = np.sum(freqs * spec) / np.sum(spec)  # (FC)
        return [Z1, Z2, Z3, Z4, Z5, Z6, Z7, Z8, Z9, Z10, Z11, Z12]

# ─────────────────────────────────────────────────────────────
# Feature 기반 Dataset: modalities로 조합 선택
# ─────────────────────────────────────────────────────────────
class Makeloader(Dataset, FeatureExtractor):
    """
    modalities 예시:
      ['Acc'], ['Force'], ['Acoustic'],
      ['Acc','Force'], ['Acc','Acoustic'], ['Force','Acoustic'], ['Acc','Force','Acoustic']
    """
    def __init__(self, directories, modalities):
        super().__init__()
        self.directories = directories
        self.modalities = _normalize_modalities(modalities)  # 폴더명 수준으로 통일
        self.data, self.labels = self._load_data()

    def _scan_and_feat(self, path):
        """CSV를 읽어 각 numeric 컬럼에 대해 Feature 12개를 산출해서 concat."""
        df = pd.read_csv(path)
        feats = []
        for col in df.columns:
            if df[col].dtype in _NUM_DTYPES:
                feats.extend(self.calculate_features(df[col].to_numpy()))
        return feats

    def _load_data(self):
        all_data, all_labels = [], []

        for directory in self.directories:
            # 각 modality 폴더의 파일맵
            maps = []
            for m in self.modalities:
                maps.append(_list_csv_map(os.path.join(directory, m)))

            # 교집합 키(동일 인덱스/이름만 사용)
            keys = _pick_common_keys(maps) if len(maps) > 1 else sorted(list(maps[0].keys()))
            if not keys:
                continue

            # 레이블 로드 (파일 수만큼 슬라이스)
            label_file = os.path.join(directory, f"{os.path.basename(directory.strip('/'))}_all_labels.csv")
            if not os.path.exists(label_file):
                raise FileNotFoundError(f"Label file not found: {label_file}")
            label_df = pd.read_csv(label_file)
            label_col = next(c for c in label_df.columns if "Tool Wear" in c)
            labels = label_df[label_col].to_numpy().tolist()

            n = min(len(keys), len(labels))
            keys = keys[:n]
            labels = labels[:n]

            for i, k in enumerate(keys):
                feat_concat = []
                for mp in maps:
                    if k in mp:
                        feat_concat.extend(self._scan_and_feat(mp[k]))
                all_data.append(feat_concat)
                all_labels.append(labels[i])

        if len(all_data) != len(all_labels):
            raise ValueError(f"Data/labels length mismatch: {len(all_data)} vs {len(all_labels)}")

        return pd.DataFrame(all_data), np.array(all_labels, dtype=np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.iloc[idx].values.astype(np.float32), self.labels[idx]

def create_loaders(directories, modalities, batch_size, train_val_ratio=None, shuffle=True):
    
    dataset = Makeloader(directories, modalities)

    if train_val_ratio:
        train_size = int(len(dataset) * train_val_ratio)
        val_size = len(dataset) - train_size
        train_subset, val_subset = torch.utils.data.random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=shuffle)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader
    else:
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return train_loader

from torch.utils.data import Dataset, DataLoader, random_split
import os
import numpy as np
import pandas as pd
import torch
# ─────────────────────────────────────────────────────────────
# Raw loader: Makeloader와 동일한 모달리티 매칭 로직 사용
# ─────────────────────────────────────────────────────────────

# 모달리티별 기대 채널 수 (고정)
EXPECTED_COLS = {'Accelerometer': 3, 'Force': 3, 'Acoustic': 1}

def _downsample_known_to_25(x: np.ndarray) -> np.ndarray:
    """
    x: (T, C)
    - T == 625 → (25,25,C).mean(axis=1)
    - T == 100 → (25, 4,C).mean(axis=1)
    - T == 25  → 그대로
    - 그 외 길이는 없다는 전제 (있으면 에러)
    """
    T, C = x.shape
    if T == 25:
        return x.astype(np.float32)
    if T == 625:
        return x.reshape(25, 25, C).mean(axis=1).astype(np.float32)
    if T == 100:
        return x.reshape(25, 4, C).mean(axis=1).astype(np.float32)
    raise ValueError(f"Unexpected T={T}. Allowed lengths: 25, 100, 625.")

def _fix_channel_count(x25: np.ndarray, expected_c: int) -> np.ndarray:
    """
    x25: (25, Cin) -> (25, expected_c)
    - Cin > expected_c → 앞에서 expected_c개만 사용
    - Cin < expected_c → 0 padding
    """
    T, Cin = x25.shape
    if Cin == expected_c:
        return x25
    if Cin > expected_c:
        return x25[:, :expected_c]
    pad = np.zeros((T, expected_c - Cin), dtype=x25.dtype)
    return np.concatenate([x25, pad], axis=1)

class RawMakeloader(Dataset):
    """
    - 모달리티 매칭: Makeloader와 동일 (파일맵 교집합 키 사용)
    - 각 CSV는 숫자만 있다고 가정 (전처리 끝)
    - 길이 정규화: 625→25평균, 100→4평균, 25→그대로
    - 모달리티별 기대 채널 수(3/3/1)로 pad/trim 후 concat → (25, C_total)
    - 라벨: {basename_dir}_all_labels.csv의 'Tool Wear' 포함 컬럼에서 앞 n개 슬라이스
    """
    def __init__(self, directories, modalities):
        self.samples = []
        self.labels  = []
        self.modalities = _normalize_modalities(modalities)  # ✅ Makeloader와 동일 함수 사용

        def _read_matrix(path: str) -> np.ndarray:
            arr = pd.read_csv(path).to_numpy(dtype=np.float32)  # 숫자만 있다고 가정
            if arr.ndim == 1:
                arr = arr[:, None]
            return arr  # (T, C)

        for d in directories:
            # 1) 모달리티별 파일맵 구성 (basename → fullpath)
            mod_maps = [ _list_csv_map(os.path.join(d, m)) for m in self.modalities ]

            # 2) 교집합 키(정렬) – Makeloader와 동일
            keys = _pick_common_keys(mod_maps) if len(mod_maps) > 1 else sorted(list(mod_maps[0].keys()))
            if not keys:
                # 해당 디렉토리에서 매칭되는 샘플 없음 → 다음 디렉토리로
                continue

            # 3) 라벨 로드 및 슬라이스
            label_file = os.path.join(d, f"{os.path.basename(d.strip('/'))}_all_labels.csv")
            if not os.path.exists(label_file):
                raise FileNotFoundError(f"Label file not found: {label_file}")
            label_df = pd.read_csv(label_file)
            label_col = next(c for c in label_df.columns if "Tool Wear" in c)
            labels = label_df[label_col].to_numpy().tolist()

            n = min(len(keys), len(labels))
            keys   = keys[:n]
            labels = labels[:n]

            # 4) 샘플 생성: 각 key에 대해 선택 모달리티를 순서대로 concat
            for i, k in enumerate(keys):
                parts = []
                for mp, modality_name in zip(mod_maps, self.modalities):
                    if k not in mp:
                        # 교집합에서 뽑았으므로 보통 발생 X, 혹시 모를 방어
                        continue
                    x  = _read_matrix(mp[k])                      # (T, C)
                    x25 = _downsample_known_to_25(x)              # (25, C)
                    x25 = _fix_channel_count(x25, EXPECTED_COLS[modality_name])  # (25, expected_C)
                    parts.append(x25)
                if not parts:
                    continue
                x_cat = np.concatenate(parts, axis=1)             # (25, sum_expected_C)
                self.samples.append(x_cat.astype(np.float32))
                self.labels.append(np.float32(labels[i]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.samples[idx])      # (25, C_total)
        y = torch.tensor([self.labels[idx]], dtype=torch.float32)
        return x, y

def create_raw_loaders(
    directories, modalities, batch_size,
    train_val_ratio=None, shuffle=True, num_workers=0, drop_last=False
):
    dataset = RawMakeloader(directories, modalities)
    N = len(dataset)
    if N == 0:
        raise ValueError("Raw dataset is empty (N=0). Check directories and file alignment.")

    if train_val_ratio is None:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, drop_last=drop_last
        )
        return loader

    # 안전 분할 (최소 1개 보장)
    train_size = int(N * float(train_val_ratio))
    train_size = max(1, min(train_size, N - 1))
    val_size   = N - train_size

    train_subset, val_subset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, drop_last=drop_last
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=False
    )
    return train_loader, val_loader


# clasification output 이용 regressor 반환 데이터 셋
def create_loaders_cls(
    directories,
    modalities,
    batch_size,
    aux_df_or_series,            # pred_prob가 들어있는 DataFrame/Series/ndarray/리스트
    prob_col: str = "pred_prob", # DataFrame일 경우 사용할 열 이름
    train_val_ratio=None,
    shuffle=True
):
    """
    Makeloader로 구성한 X(기존 84차원)에 aux의 pred_prob를 마지막 변수로 붙여
    85차원으로 만드는 로더 생성.

    Parameters
    ----------
    directories : list[str]
        기존과 동일.
    modalities : list[str]
        기존과 동일.
    batch_size : int
        배치 크기.
    aux_df_or_series : (pd.DataFrame | pd.Series | np.ndarray | list)
        pred_prob가 담긴 자료. DataFrame이면 prob_col을 사용해서 값 추출.
        길이는 dataset 길이와 동일해야 함.
    prob_col : str
        DataFrame에서 사용할 확률 열 이름(기본: 'pred_prob').
    train_val_ratio : float | None
        기존과 동일. None이면 단일 train_loader만 반환.
    shuffle : bool
        기존과 동일.
    """
    # 1) 기존 데이터셋 생성
    dataset = Makeloader(directories, modalities)

    # 2) aux에서 pred_prob 벡터 추출
    if isinstance(aux_df_or_series, pd.DataFrame):
        if prob_col not in aux_df_or_series.columns:
            raise ValueError(f"'{prob_col}' column not found in aux DataFrame.")
        probs = aux_df_or_series[prob_col].to_numpy()
    elif isinstance(aux_df_or_series, pd.Series):
        probs = aux_df_or_series.to_numpy()
    elif isinstance(aux_df_or_series, (np.ndarray, list, tuple)):
        probs = np.asarray(aux_df_or_series)
    else:
        raise TypeError("aux_df_or_series must be DataFrame/Series/ndarray/list.")

    # 3) 길이 체크 및 잘라쓰기(과잉 길이 방지)
    N = len(dataset)
    if len(probs) < N:
        raise ValueError(f"aux length ({len(probs)}) < dataset length ({N}).")
    if len(probs) > N:
        probs = probs[:N]

    # 4) dtype 정리 + 마지막 열로 병합 (85차원)
    probs = probs.astype(np.float32)
    # dataset.data는 DataFrame (samples x features)
    dataset.data = pd.concat(
        [dataset.data.reset_index(drop=True),
         pd.Series(probs, name=prob_col)],
        axis=1
    )

    # 5) 로더 생성 (기존 로직 유지)
    if train_val_ratio:
        train_size = int(N * float(train_val_ratio))
        val_size   = N - train_size
        train_subset, val_subset = torch.utils.data.random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=shuffle)
        val_loader   = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader
    else:
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return train_loader

