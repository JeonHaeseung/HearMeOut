from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import os
import json
import glob
from tqdm import tqdm

import logging
logger = logging.getLogger(__name__)
import hydra
from omegaconf import DictConfig

import torch
import torch.nn as nn

import numpy as np
import pandas as pd
from model_ete_transformer import WhisperEncoderRegressor
from transformers import WhisperFeatureExtractor
from utils.file_io import read_signal

os.environ["HYDRA_FULL_ERROR"] = "7"


def set_device():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dataset(cfg, file_path, device):
    signal_array = read_signal(file_path, sample_rate=16000)
    L_signal_array = signal_array[:, 0]   # 왼쪽 채널
    R_signal_array = signal_array[:, 1]   # 오른쪽 채널

    L_feature_extractor = WhisperFeatureExtractor.from_pretrained(cfg.model_L_whisper_root)
    R_feature_extractor = WhisperFeatureExtractor.from_pretrained(cfg.model_R_whisper_root)

    L_input_features = L_feature_extractor(L_signal_array, sampling_rate=16000).input_features[0]  # (80, T)
    L_input_tensor = torch.tensor(L_input_features, dtype=torch.float).unsqueeze(0).to(device)

    R_input_features = R_feature_extractor(R_signal_array, sampling_rate=16000).input_features[0]  # (80, T)
    R_input_tensor = torch.tensor(R_input_features, dtype=torch.float).unsqueeze(0).to(device)
    return L_input_tensor, R_input_tensor


def load_dataset_batch(cfg, file_paths, device):
    L_feature_extractor = WhisperFeatureExtractor.from_pretrained(cfg.model_L_whisper_root)
    R_feature_extractor = WhisperFeatureExtractor.from_pretrained(cfg.model_R_whisper_root)

    L_batch = []
    R_batch = []

    for file_path in file_paths:
        signal_array = read_signal(file_path, sample_rate=16000)
        L = signal_array[:, 0]
        R = signal_array[:, 1]

        L_feat = L_feature_extractor(L, sampling_rate=16000).input_features[0]
        R_feat = R_feature_extractor(R, sampling_rate=16000).input_features[0]

        L_batch.append(torch.tensor(L_feat, dtype=torch.float))
        R_batch.append(torch.tensor(R_feat, dtype=torch.float))

    L_tensor = torch.stack(L_batch).to(device)  # (B, 80, T)
    R_tensor = torch.stack(R_batch).to(device)
    return L_tensor, R_tensor


def load_model(cfg):
    # load the pre-trained checkpoints
    model = WhisperEncoderRegressor(cfg.model_L_whisper_root, cfg.model_R_whisper_root)  # 반드시 같은 구조여야 함
    model.load_state_dict(torch.load(os.path.join(cfg.model_final_LR_root, "model_final_e5.pth")))
    return model


def predict_score(model, L_input_tensor, R_input_tensor):
    with torch.no_grad():
        prediction = model(L_input_tensor, R_input_tensor)
        return prediction.item()


@hydra.main(config_path=".", config_name="config", version_base=None)
def predict_dev(cfg: DictConfig):
    device = set_device()

    # load the model
    logger.info("Loading model")
    model = load_model(cfg).to(device)
    model.eval()

    # Load the data
    logger.info("Loading dataset")
    wavs = glob.glob(os.path.join(cfg.dev_dataset, cfg.dev_signals, "*.wav"), recursive=True)
    # wavs = wavs[:10]

    # Make predictions for all items in the dev data
    logger.info("Starting predictions")
    prediction_records = []

    # for wav in tqdm(wavs):
    #     L_input_tensor, R_input_tensor = load_dataset(cfg, wav, device)
    #     score = predict_score(model, L_input_tensor, R_input_tensor)
    #     signal_id = Path(wav).stem

    #     prediction_records.append({
    #         "signal": signal_id,
    #         "predicted": score * 100
    #     })

    # 수정: 배치 처리로 변경
    BATCH_SIZE = 64
    for i in tqdm(range(0, len(wavs), BATCH_SIZE)):
        batch_paths = wavs[i:i + BATCH_SIZE]
        L_input, R_input = load_dataset_batch(cfg, batch_paths, device)

        with torch.no_grad():
            scores = model(L_input, R_input).cpu().numpy()  # (B,)

        for wav_path, score in zip(batch_paths, scores):
            signal_id = Path(wav_path).stem
            prediction_records.append({
                "signal": signal_id,
                "predicted": float(score * 100)
            })

    records_dev_df = pd.DataFrame(prediction_records)

    # Save results to CSV file
    records_dev_df[["signal", "predicted"]].to_csv(
        cfg.dev_predict_file,
        index=False,
        header=["signal", "predicted"],
        mode="w",
    )
    logger.info(f"Predictions saved to {cfg.dev_predict_file}")


if __name__ == "__main__":
    predict_dev()
