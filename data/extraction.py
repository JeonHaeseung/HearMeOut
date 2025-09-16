import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
import json
import glob
import logging
import multiprocessing
from functools import partial
from pathlib import Path

from tqdm import tqdm
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import WhisperFeatureExtractor

import parmap
from utils.file_io import read_signal
from omegaconf import OmegaConf

import numpy as np
from sklearn.model_selection import train_test_split
from multiprocessing import Manager

import logging
logger = logging.getLogger(__name__)
import hydra
from omegaconf import DictConfig

ASR_MODEL = "openai/whisper-small"
N_SAMPLES = None  # None for total
manager = Manager()
signal_manager = manager.list()
METHOD = "WN" # WN, TS, CS, CP

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def read_response(metadata_path, signal_name):
    with open(metadata_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for record in data:
        if record["signal"] == signal_name:
            return record["correctness"] / 100.0, record["response"]
    return None, None


def process_file(wav, metadata_path, sample_rate):
    signal_array = read_signal(filename=wav, sample_rate=sample_rate)
    L_array = signal_array[:, 0]
    R_array = signal_array[:, 1]

    signal_name = Path(wav).stem
    signal_name = signal_name.split("_aug")[0]

    correctness, signal_sentence = read_response(metadata_path, signal_name)

    signal_point = {
        "audio": {
            "path": wav,
            "L_array": L_array,
            "R_array": R_array,
            "sampling_rate": sample_rate,
        },
        "correctness": correctness,
        "sentence": signal_sentence,
    }
    signal_manager.append(signal_point)


def prepare_dataset(batch, L_feature_extractor, R_feature_extractor):
    audio = batch["audio"]
    sr = audio["sampling_rate"]
    L_array = audio["L_array"]
    R_array = audio["R_array"]

    batch["L_input_features"] = L_feature_extractor(L_array, sampling_rate=sr).input_features[0]
    batch["R_input_features"] = R_feature_extractor(R_array, sampling_rate=sr).input_features[0]
    return batch


def load_dataset(cfg):
    wavs_aug = glob.glob(os.path.join(f"{cfg.aug_dataset}_{METHOD}", cfg.aug_signals, "*.wav"), recursive=True)
    wavs_ori = glob.glob(os.path.join(cfg.train_dataset, cfg.train_signals, "*.wav"), recursive=True)
    wavs = wavs_aug + wavs_ori

    if N_SAMPLES is not None:
        wavs = wavs[:N_SAMPLES]

    logger.info(f"Found aug {len(wavs_aug)} ({cfg.aug_dataset}_{METHOD}), ori {len(wavs_ori)} .wav files")

    process_file_args = partial(
        process_file,
        metadata_path=cfg.aug_metadata,
        sample_rate=cfg.sample_rate,
    )
    parmap.map(process_file_args, wavs, pm_pbar=True, pm_processes=multiprocessing.cpu_count())

    signal_list = list(signal_manager)
    logger.info(f"Processed signals: {len(signal_list)}")

    train_data, test_data = train_test_split(signal_list, test_size=0.2, random_state=42)
    return DatasetDict({
        "train": Dataset.from_list(train_data),
        "test": Dataset.from_list(test_data),
    })


def extract_features(dataset_dict, L_feature_extractor, R_feature_extractor):
    logger.info(f"Extracting features...")
    return dataset_dict.map(
        prepare_dataset,
        fn_kwargs={
            "L_feature_extractor": L_feature_extractor,
            "R_feature_extractor": R_feature_extractor
        },
        num_proc=1
    )


@hydra.main(config_path="..", config_name="config", version_base=None)
def main(cfg: DictConfig):
    feature_save_root = f"{cfg.feature_root}_aug_{METHOD}"
    os.makedirs(feature_save_root, exist_ok=True)
    print(feature_save_root)
    # print(f"{cfg.model_L_whisper_root}_aug")
    # print(f"{cfg.model_R_whisper_root}_aug")

    logger.info("Loading Whisper feature extractors")
    # L_extractor = WhisperFeatureExtractor.from_pretrained(f"{cfg.model_L_whisper_root}_aug")
    # R_extractor = WhisperFeatureExtractor.from_pretrained(f"{cfg.model_R_whisper_root}_aug")
    L_extractor = WhisperFeatureExtractor.from_pretrained(ASR_MODEL)
    R_extractor = WhisperFeatureExtractor.from_pretrained(ASR_MODEL)

    dataset_dict = load_dataset(cfg)
    feature_dataset = extract_features(dataset_dict, L_extractor, R_extractor)

    logger.info(f"Saving dataset to {feature_save_root}")
    feature_dataset.save_to_disk(feature_save_root)

    logger.info("Dataset saved successfully")


if __name__ == "__main__":
    main()