import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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


from datasets import DatasetDict   ########
from torch.nn import TransformerEncoder, TransformerEncoderLayer

N_SAMPLES = None   # None for total

manager = Manager()
signal_manager = manager.list()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
    def __init__(self, L_model_path: str, R_model_path: str, hidden_size=512, num_layers=2, num_heads=4):
        super().__init__()

        self.L_whisper = WhisperForConditionalGeneration.from_pretrained(L_model_path)
        self.R_whisper = WhisperForConditionalGeneration.from_pretrained(R_model_path)

        # 디코더 사용 안함
        self.L_whisper.model.decoder.requires_grad_(False)
        self.R_whisper.model.decoder.requires_grad_(False)

        # 인코더 freeze
        self.L_whisper.model.encoder.requires_grad_(False)
        self.R_whisper.model.encoder.requires_grad_(False)

        L_hidden_size = self.L_whisper.config.d_model
        R_hidden_size = self.R_whisper.config.d_model
        self.hidden_size = L_hidden_size + R_hidden_size

        # Transformer Encoder
        encoder_layer = TransformerEncoderLayer(d_model=self.hidden_size, nhead=num_heads, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Regression head
        self.regressor = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, L_input_features, R_input_features, attention_mask=None):
        L_enc_out = self.L_whisper.model.encoder(input_features=L_input_features, attention_mask=attention_mask)
        R_enc_out = self.R_whisper.model.encoder(input_features=R_input_features, attention_mask=attention_mask)
        L_hidden = L_enc_out.last_hidden_state  # (B, T, H_L)
        R_hidden = R_enc_out.last_hidden_state  # (B, T, H_R)
        hidden = torch.cat([L_hidden, R_hidden], dim=-1)  # (B, T, H_L+H_R)

        # Transformer encoder
        x = self.transformer_encoder(hidden)  # (B, T, H)
        x = x.mean(dim=1)  # Mean pooling over time dimension → (B, H)

        score = self.regressor(x)  # (B, 1)
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


@hydra.main(config_path="..", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # L_feature_extractor = WhisperFeatureExtractor.from_pretrained(cfg.model_L_whisper_root)
    # R_feature_extractor = WhisperFeatureExtractor.from_pretrained(cfg.model_R_whisper_root)
    # print(cfg.model_final_LR_root)

    # signal_datadict = load_dataset(cfg)
    # signal_feature = extract_dataset(signal_datadict, L_feature_extractor, R_feature_extractor)

    # 추출한 데이터셋이 있는 경우
    signal_feature = DatasetDict.load_from_disk(cfg.feature_aug_root)
    os.makedirs(f"{cfg.model_final_LR_root}_aug", exist_ok=True)

    model = run_training(
        train_dataset=signal_feature["train"],
        test_dataset=signal_feature["test"],
        L_model_path=f"{cfg.model_L_whisper_root}_aug",
        R_model_path=f"{cfg.model_R_whisper_root}_aug",
        save_path=f"{cfg.model_final_LR_root}_aug",
        num_epochs=20,     # 5, 20, 100
        batch_size=64,
        lr=1e-5
    )

    torch.save(model.state_dict(), os.path.join(f"{cfg.model_final_LR_root}_aug", "model_final.pth"))
       

if __name__ == "__main__":
    main()