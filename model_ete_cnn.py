import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import os
import json
import glob

from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WhisperForConditionalGeneration, WhisperFeatureExtractor
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict
from tqdm import tqdm

import parmap
import multiprocessing
from multiprocessing import Manager
from functools import partial

import logging
logger = logging.getLogger(__name__)
import hydra
from omegaconf import DictConfig

import numpy as np
from sklearn.model_selection import train_test_split
from utils.file_io import read_signal

N_SAMPLES = None         # None for total

manager = Manager()
signal_manager = manager.list()

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ["HYDRA_FULL_ERROR"] = "1"


class WhisperRegressionDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        L_input = sample["L_input_features"]
        R_input = sample["R_input_features"]
        score = sample["correctness"]  # float (0.0~1.0)

        return {
            "L_input_features": torch.tensor(L_input, dtype=torch.float),
            "R_input_features": torch.tensor(R_input, dtype=torch.float),
            "label": torch.tensor(score, dtype=torch.float),
        }
    

class WhisperEncoderRegressor(nn.Module):
    def __init__(self, L_model_path: str, R_model_path: str, cnn_out_dim=256):
        super().__init__()

        self.L_whisper = WhisperForConditionalGeneration.from_pretrained(L_model_path)
        self.R_whisper = WhisperForConditionalGeneration.from_pretrained(R_model_path)

        # 디코더 사용 안함
        self.L_whisper.model.decoder.requires_grad_(False)
        self.R_whisper.model.decoder.requires_grad_(False)

        # 인코더 freeze
        self.L_whisper.model.encoder.requires_grad_(False)
        self.R_whisper.model.encoder.requires_grad_(False)

        L_hidden_size = self.L_whisper.config.d_model # (B, T, H_L)
        R_hidden_size = self.R_whisper.config.d_model # (B, T, H_R)
        self.hidden_size = L_hidden_size + R_hidden_size

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=self.hidden_size, out_channels=cnn_out_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        # attention pooling
        self.attn = nn.MultiheadAttention(embed_dim=cnn_out_dim, num_heads=4, batch_first=True)

        # regression
        self.regressor = nn.Sequential(
            nn.LayerNorm(cnn_out_dim),
            nn.Linear(cnn_out_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, L_input_features, R_input_features, attention_mask=None):
        L_enc_out = self.L_whisper.model.encoder(input_features=L_input_features, attention_mask=attention_mask)
        R_enc_out = self.R_whisper.model.encoder(input_features=R_input_features, attention_mask=attention_mask)
        L_hidden = L_enc_out.last_hidden_state  # (B, T, H_L)
        R_hidden = R_enc_out.last_hidden_state  # (B, T, H_R)
        hidden = torch.cat([L_hidden, R_hidden], dim=-1)

        x = hidden.transpose(1, 2)
        x = self.cnn(x).squeeze(-1)
        x = x.unsqueeze(1)
        x, _ = self.attn(x, x, x)
        x = x.squeeze(1)
        score = self.regressor(x)
        return score


def train(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Training"):
        L_inputs = batch["L_input_features"].to(device)
        R_inputs = batch["R_input_features"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(L_inputs, R_inputs)
        outputs = outputs.squeeze(1)
        loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    preds, targets = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            L_inputs = batch["L_input_features"].to(device)
            R_inputs = batch["R_input_features"].to(device)
            labels = batch["label"].to(device)

            outputs = model(L_inputs, R_inputs)
            outputs = outputs.squeeze(1)
            loss = loss_fn(outputs, labels)

            total_loss += loss.item()
            preds.extend(outputs.cpu().tolist())
            targets.extend(labels.cpu().tolist())

    return total_loss / len(dataloader), preds, targets


def rmse_loss(predictions, targets):
    return torch.sqrt(torch.mean((predictions - targets) ** 2))


def run_training(train_dataset, test_dataset, L_model_path, R_model_path, save_path, num_epochs=5, batch_size=8, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_set = WhisperRegressionDataset(train_dataset)
    test_set = WhisperRegressionDataset(test_dataset)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    model = WhisperEncoderRegressor(L_model_path, R_model_path).to(device)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    loss_fn = rmse_loss

    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
        train_loss = train(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_preds, val_targets = evaluate(model, test_loader, loss_fn, device)

        logger.info(f"Train Loss: {train_loss:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}")

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(save_path, f"model_final_e{epoch+1}.pth"))
    return model


def read_response(metadata_path, signal_name):
    with open(metadata_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    correctness = None
    for record in data:
        if record["signal"] == signal_name:
            correctness = (record["correctness"] / 100)
            break
    return correctness


def process_file(wav, metadata_path, sample_rate):
    signal_array = read_signal(filename=wav, sample_rate=sample_rate)

    L_signal_array = signal_array[:, 0]   # 왼쪽 채널
    R_signal_array = signal_array[:, 1]   # 오른쪽 채널

    signal_name = Path(wav).stem
    signal_correctness = read_response(metadata_path=metadata_path, signal_name=signal_name)
    signal_point = {
        "audio": {
            "path": wav,
            "L_array": L_signal_array,
            "R_array": R_signal_array,
            "sampling_rate": sample_rate,
        },
        "correctness": signal_correctness,
    }
    signal_manager.append(signal_point)



def load_dataset(cfg):
    wavs = glob.glob(os.path.join(cfg.train_dataset, cfg.train_signals, "*.wav"), recursive=True)
    if N_SAMPLES is not None:
        wavs = wavs[:N_SAMPLES]
    logger.info(f"Total WAV files: {len(wavs)}")

    num_processes = multiprocessing.cpu_count()
    process_file_args = partial(
                    process_file, 
                    metadata_path=cfg.train_metadata, 
                    sample_rate=cfg.sample_rate)
    
    with multiprocessing.Pool(num_processes) as pool:
        parmap.map(
                    process_file_args,
                    wavs,
                    pm_pbar=True,
                    pm_processes=num_processes)

    signal_list = list(signal_manager)
    logger.info(f"Total created dataset: {len(signal_list)}")
    logger.info(f"One sample: {signal_list[0]}")
    logger.info(f"One vector length: {len(signal_list[0]['audio']['L_array'])}")

    train_data, test_data = train_test_split(signal_list, test_size=0.2, random_state=42)

    train_dataset = Dataset.from_list(train_data)
    test_dataset = Dataset.from_list(test_data)
    logger.info(f"Train Dataset format: {train_dataset}")
    logger.info(f"Test Dataset format: {test_dataset}")

    signal_datadict = DatasetDict({
        "train": train_dataset,
        "test": test_dataset,
    })
    logger.info(f"Dataset format: {signal_datadict}")
    
    return signal_datadict


def prepare_dataset(batch, L_feature_extractor, R_feature_extractor):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]
    L_array = audio["L_array"]
    R_array = audio["R_array"]
    sr = audio["sampling_rate"]

    # compute log-Mel input features from input audio array 
    batch["L_input_features"] = L_feature_extractor(L_array, sampling_rate=sr).input_features[0]
    batch["R_input_features"] = R_feature_extractor(R_array, sampling_rate=sr).input_features[0]
    return batch


def extract_dataset(signal_datadict, L_feature_extractor, R_feature_extractor):
    logger.info(f"Mapping dataset")

    signal_feature = signal_datadict.map(
        prepare_dataset,
        fn_kwargs={
            "L_feature_extractor": L_feature_extractor,
            "R_feature_extractor": R_feature_extractor
        },
        # remove_columns=signal_datadict.column_names["train"],
        num_proc=1 # cannot use more than 1
        )

    logger.info(f"Signal feature: {signal_feature}")
    return signal_feature


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # L_feature_extractor = WhisperFeatureExtractor.from_pretrained(cfg.model_L_whisper_root)
    # R_feature_extractor = WhisperFeatureExtractor.from_pretrained(cfg.model_R_whisper_root)
    # print(cfg.model_final_LR_root)

    # signal_datadict = load_dataset(cfg)
    # signal_feature = extract_dataset(signal_datadict, L_feature_extractor, R_feature_extractor)

    # 추출한 데이터셋이 있는 경우
    signal_feature = DatasetDict.load_from_disk(cfg.feature_root)

    model = run_training(
        train_dataset=signal_feature["train"],
        test_dataset=signal_feature["test"],
        L_model_path=cfg.model_L_whisper_root,
        R_model_path=cfg.model_R_whisper_root,
        save_path=cfg.model_final_LR_root,
        num_epochs=100,
        batch_size=64,
        lr=1e-5
    )

    torch.save(model.state_dict(), os.path.join(cfg.model_final_LR_new_root, "model_final.pth"))
       

if __name__ == "__main__":
    main()