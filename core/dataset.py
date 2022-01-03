from typing import Union, Optional, List, Dict, Any
import datasets
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from loguru import logger
from dataclasses import dataclass
from transformers.file_utils import PaddingStrategy
import torch


@dataclass
class DataCollator:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[str, PaddingStrategy] = "max_length"
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:

        labels = torch.LongTensor([feat["label"].item() for feat in features])
        for feat in features:
            feat.pop("label")
        max_length = max(feat["input_ids"].shape[-1] for feat in features)
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            **{key: value for key, value in batch.items()},
            "label": labels,
        }

        return batch


class MNLILightningDataModule(pl.LightningDataModule):
    def __init__(self, model_name, batch_size, num_workers):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.collate_fn = DataCollator(self.tokenizer)
        self.filter_fn = lambda item: item["label"] != -1

    def _transform(self, item):
        premise, hypothesis, label = (
            item["premise"],
            item["hypothesis"],
            item["label"],
        )
        inputs = self.tokenizer(premise, hypothesis)

        output = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "label": label,
        }

        return output

    def _data_processing(self, dataset: datasets.arrow_dataset.Dataset, name: str):
        logger.info(f"{name} data transformation...")
        dataset = dataset.filter(self.filter_fn)
        dataset = dataset.map(self._transform)
        dataset.set_format(type="torch", columns=self.columns)
        logger.info(f"{name} data transformation complted.")

        return dataset

    def prepare_data(self) -> None:
        logger.info("Dataset downloading...")
        self.dataset = datasets.load_dataset("multi_nli")
        self.train, self.validation, self.test = (
            self.dataset["train"],
            self.dataset["validation_matched"],
            self.dataset["validation_mismatched"],
        )

        self.columns = [
            "input_ids",
            "attention_mask",
            "label",
        ]

        logger.info("Dataset filtering")
        self.train = self._data_processing(self.train, "Training")
        self.validation = self._data_processing(self.validation, "Validation")
        self.test = self._data_processing(self.test, "Testing")

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.validation,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
