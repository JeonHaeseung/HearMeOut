import json
import os
import random
import glob
from pathlib import Path
from typing import List
from tqdm import tqdm

import numpy as np
import librosa
import soundfile as sf
# import colorednoise as cn

import hydra
from omegaconf import DictConfig


RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
METHOD = "NONE" # WN, TS, CS, CP


def add_white_noise(y: np.ndarray) -> np.ndarray:
    # https://www.kaggle.com/code/kaerunantoka/birdclef2022-use-2nd-label-f0
    white_noise = np.random.randn(*y.shape) * 0.0005
    return y + white_noise


def time_shift(y: np.ndarray) -> np.ndarray:
    # https://www.kaggle.com/code/CVxTz/audio-data-augmentation/notebook
    shift = np.random.randint(0, 1000 + 1)
    return np.roll(y, shift, axis=1)


def change_speed(y: np.ndarray) -> np.ndarray:
    rate = np.random.uniform(0.995, 1.005)
    chs = [librosa.effects.time_stretch(y[c], rate=rate) for c in range(y.shape[0])]
    L = min(map(len, chs))
    return np.stack([ch[:L] for ch in chs], axis=0)


def change_pitch(y: np.ndarray, sr: int) -> np.ndarray:
    # https://www.kaggle.com/code/CVxTz/audio-data-augmentation/notebook
    n_steps = np.random.choice([-1, 1])
    chs = [librosa.effects.pitch_shift(y[c], sr=sr, n_steps=n_steps) for c in range(y.shape[0])]
    L = min(map(len, chs))
    return np.stack([ch[:L] for ch in chs], axis=0)


def augment_random(y: np.ndarray, sr: int) -> np.ndarray:
    """증강 연산 1~2개를 랜덤 조합으로 적용"""
    ops = [
        lambda x: add_white_noise(x),
        lambda x: time_shift(x),
        lambda x: change_speed(x),
        lambda x: change_pitch(x, sr),
    ]
    k = np.random.choice([1, 2])  # 1개 또는 2개 랜덤 선택
    chosen = np.random.choice(ops, size=k, replace=False)
    y_aug = y.copy()
    for op in chosen:
        y_aug = op(y_aug)
    y_aug = np.clip(y_aug, -1.0, 1.0)
    return y_aug


def augment_fix(y: np.ndarray, sr: int) -> np.ndarray:
    """증강 연산 1개를 정해서 적용"""
    aug_dict = {
        "WN": add_white_noise,
        "TS": time_shift,
        "CS": change_speed,
        "CP": change_pitch,
    }
    y_aug = y.copy()
    aug_func = aug_dict.get(METHOD)
    if METHOD == "CP":
        y_aug = aug_func(y_aug, sr)
    else:
        y_aug = aug_func(y_aug)
    y_aug = np.clip(y_aug, -1.0, 1.0)
    return y_aug


def next_aug_index(out_dir: str, stem: str) -> int:
    # 이미 존재하는 <stem>_augN.wav가 있다면 가장 큰 N 다음 번호를 반환.
    pattern = os.path.join(out_dir, f"{stem}_aug*.wav")
    max_idx = 0
    for p in glob.glob(pattern):
        base = os.path.basename(p)
        try:
            idx = int(os.path.splitext(base)[0].split("_aug")[-1])
            max_idx = max(max_idx, idx)
        except Exception:
            continue
    return max_idx + 1


@hydra.main(config_path="..", config_name="config", version_base=None)
def main(cfg: DictConfig):
    METADATA_FILE = cfg.train_metadata
    ORIG_DIR = os.path.join(cfg.train_dataset, cfg.train_signals)
    AUG_DIR  = os.path.join(f"{cfg.aug_dataset}_{METHOD}", cfg.aug_signals)
    os.makedirs(AUG_DIR, exist_ok=True)

    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # correctness in (0, 100) 인 것만 처리하기 (0 or 100은 포함 X)
    targets: List[str] = []
    for item in meta:
        correctness = float(item.get("correctness", -1))
        if 0 < correctness < 100:
            sig = item["signal"]
            targets.append(sig)
    print(f"Target files: {len(targets)}")


    for sig in tqdm(targets, desc="Augmenting", unit="file"):
        in_path = os.path.join(ORIG_DIR, f"{sig}.wav")
        y, sr = librosa.load(in_path, sr=cfg.sample_rate, mono=False) # 양이음 채널 유지해야 함
        # print(f"Sample rate: {sr}")
        start_n = next_aug_index(AUG_DIR, sig)

        # 각 파일당 2개씩 augmentation 생성
        for j in range(2):
            if METHOD == "NONE":
                y_aug = y.copy()
            else:
                # y_aug = augment_once(y, sr)
                y_aug = augment_fix(y, sr)
            out_path = os.path.join(AUG_DIR, f"{sig}_aug{start_n + j}.wav")
            sf.write(out_path, y_aug.T, sr, subtype="PCM_16")


if __name__ == "__main__":
    main()