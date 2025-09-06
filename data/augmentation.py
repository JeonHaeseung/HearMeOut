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


TARGET_PER_BIN = 500
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def add_white_noise(original_audio):
    # https://www.kaggle.com/code/kaerunantoka/birdclef2022-use-2nd-label-f0
    white_noise = np.random.randn(len(original_audio)) * 0.005
    augmented_audio = original_audio + white_noise
    return augmented_audio


# def color_noise(original_audio):
#     # https://www.kaggle.com/code/kaerunantoka/birdclef2022-use-2nd-label-f0
#     pink_noise = cn.powerlaw_psd_gaussian(1, len(original_audio))
#     augmented_audio = original_audio + pink_noise
#     return augmented_audio


# def background_noise(original_audio):
#     # Load background noise from another audio file
#     background_noise, sample_rate = librosa.load("background_noise.wav")
#     augmented_audio = original_audio + background_noise
#     return augmented_audio


def time_shift(original_audio):
    # https://www.kaggle.com/code/CVxTz/audio-data-augmentation/notebook
    shift = np.random.randint(3000, 5000 + 1)
    augmented_audio = np.roll(original_audio, shift)
    return augmented_audio


def change_speed(original_audio):
    rate = np.random.uniform(0.95, 1.05)
    augmented_audio = librosa.effects.time_stretch(original_audio, rate=rate)
    return augmented_audio


def change_pitch(original_audio, sr):
    # https://www.kaggle.com/code/CVxTz/audio-data-augmentation/notebook
    n_steps = np.random.randint(1, 4 + 1)
    augmented_audio = librosa.effects.pitch_shift(original_audio, sr=sr, n_steps=n_steps)
    return augmented_audio


def augment_once(y: np.ndarray, sr: int) -> np.ndarray:
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


def next_aug_index(out_dir: Path, stem: str) -> int:
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
    AUG_DIR  = os.path.join(cfg.aug_dataset, cfg.aug_signals)

    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # correctness in (0, 100) 인 것만 처리하기 (0 or 100은 포함 X)
    targets: List[str] = []
    for item in meta:
        c = float(item.get("correctness", -1))
        if 0 < c < 100:
            sig = item["signal"]
            targets.append(sig)
    print(f"Target files: {len(targets)}")


    for sig in tqdm(targets, desc="Augmenting", unit="file"):
        in_path = os.path.join(ORIG_DIR, f"{sig}.wav")
        y, sr = librosa.load(in_path, sr=None, mono=True)
        start_n = next_aug_index(AUG_DIR, sig)

        # 각 파일당 2개씩 augmentation 생성
        for j in range(2):
            y_aug = augment_once(y, sr)
            out_path = os.path.join(AUG_DIR, f"{sig}_aug{start_n + j}.wav")
            sf.write(out_path, y_aug, sr, subtype="PCM_16")


if __name__ == "__main__":
    main()