import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import os
import json
import glob

from tqdm import tqdm
from pathlib import Path

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
from transformers import WhisperFeatureExtractor, WhisperTokenizer, \
                WhisperProcessor, WhisperForConditionalGeneration, WhisperModel, \
                Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import Dataset, DatasetDict
import evaluate

from utils.file_io import read_signal
from model_ft_data import DataCollatorSpeechSeq2SeqWithPadding

N_SAMPLES = None         # None for total
ASR_MODEL = "openai/whisper-small"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["HYDRA_FULL_ERROR"] = "1"
SETTING = "R"           # R, L

manager = Manager()
signal_manager = manager.list()


#MARK: 모델 로드
def load_model():
    feature_extractor = WhisperFeatureExtractor.from_pretrained(ASR_MODEL)
    tokenizer = WhisperTokenizer.from_pretrained(ASR_MODEL, language="English", task="transcribe")
    processor = WhisperProcessor.from_pretrained(ASR_MODEL, language="English", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(ASR_MODEL)
    model.generation_config.language = "English"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    return feature_extractor, tokenizer, processor, model


def read_response(metadata_path, signal_name):
    with open(metadata_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    response = None
    for record in data:
        if record["signal"] == signal_name:
            response = record["response"]
            break
    return response


def process_file(wav, metadata_path, sample_rate):
    signal_array = read_signal(filename=wav, sample_rate=sample_rate)

    if SETTING=="L":                        # multichannel
        signal_array = signal_array[:, 0]   # 왼쪽 채널
    elif SETTING=="R":
        signal_array = signal_array[:, 1]   # 오른쪽 채널

    signal_name = Path(wav).stem
    signal_sentence = read_response(metadata_path=metadata_path, signal_name=signal_name)
    signal_point = {
        "audio": {
            "path": wav,
            "array": signal_array,
            "sampling_rate": sample_rate,
        },
        "sentence": signal_sentence,
    }
    signal_manager.append(signal_point)


# datasets format:
# {
#   "audio": {
#     "path": <파일 경로>,
#     "array": <float32 numpy 배열>,
#     "sampling_rate": <샘플레이트>
#   },
#   "sentence": <문장 문자열>
# }
#MARK: 데이터 로드
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
    logger.info(f"One vector length: {len(signal_list[0]['audio']['array'])}")

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


def prepare_dataset(batch, feature_extractor, tokenizer):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids 
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch


def extract_dataset(signal_datadict, feature_extractor, tokenizer):
    logger.info(f"Mapping dataset")

    signal_feature = signal_datadict.map(
        prepare_dataset,
        fn_kwargs={
            "feature_extractor": feature_extractor,
            "tokenizer": tokenizer
        },
        remove_columns=signal_datadict.column_names["train"],
        num_proc=1 # cannot use more than 1
        )
    return signal_feature


def compute_metrics(pred, tokenizer):
    metric = evaluate.load("wer")

    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


def train_model(cfg, signal_feature, feature_extractor, tokenizer, processor, model):
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    whipser_save_path = f"{cfg.model_whisper_root}_{SETTING}"
    print(whipser_save_path)

    training_args = Seq2SeqTrainingArguments(
        output_dir=whipser_save_path,  # change to a repo name of your choice
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
        learning_rate=1e-5,
        warmup_steps=100,
        max_steps=500, # epoch
        gradient_checkpointing=True,
        fp16=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=16,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=50,
        eval_steps=50,
        logging_steps=10,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
    )

    # 일단 작은 버전으로 테스트
    # training_args = Seq2SeqTrainingArguments(
    #     output_dir=cfg.model_poc_root,  # change to a repo name of your choice
    #     per_device_train_batch_size=512,
    #     gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    #     learning_rate=1e-5,
    #     warmup_steps=20,
    #     max_steps=100, # epoch
    #     gradient_checkpointing=True,
    #     fp16=True,
    #     evaluation_strategy="steps",
    #     per_device_eval_batch_size=512,
    #     predict_with_generate=True,
    #     generation_max_length=225,
    #     save_steps=10,
    #     eval_steps=10,
    #     logging_steps=5,
    #     report_to=["tensorboard"],
    #     load_best_model_at_end=True,
    #     metric_for_best_model="wer",
    #     greater_is_better=False,
    # )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=signal_feature["train"],
        eval_dataset=signal_feature["test"],
        data_collator=data_collator,
        compute_metrics=partial(compute_metrics, tokenizer=tokenizer),
        processing_class=processor.feature_extractor,
    )

    trainer.train()

    trainer.save_model(whipser_save_path)
    processor.save_pretrained(whipser_save_path)
    tokenizer.save_pretrained(whipser_save_path)
    feature_extractor.save_pretrained(whipser_save_path)



@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig):
    whipser_save_path = f"{cfg.model_whisper_root}_{SETTING}"
    print(whipser_save_path)
    feature_extractor, tokenizer, processor, model = load_model()
    signal_datadict = load_dataset(cfg)
    signal_feature = extract_dataset(signal_datadict, feature_extractor, tokenizer)
    train_model(cfg, signal_feature, feature_extractor, tokenizer, processor, model)


if __name__ == "__main__":
    main()