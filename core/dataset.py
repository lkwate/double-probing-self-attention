from typing import Union, Optional, List, Dict, Any
import datasets
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from loguru import logger
from dataclasses import dataclass
from transformers.file_utils import PaddingStrategy
import torch
import pandas as pd


class MNLIDataset(Dataset):
    def __init__(self, tokenizer: AutoTokenizer, src_file: str, batch_size: int):
        super().__init__()
        self.tokenizer = tokenizer
        self.src_file = src_file
        self.batch_size = batch_size

        self.data = pd.read_csv(src_file)
        self.label_factory = {"neutral": 0, "entailment": 1, "contradiction": 2}
        self.data = self.data[self.data["label"].isin(self.label_factory)]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        item = self.data.iloc[idx]
        sent1, sent2, label = (item["sentence1"], item["sentence2"], item["label"])
        label = float(self.label_factory[label])

        inputs = self.tokenizer(sent1, sent2)
        output = {
            **inputs,
            "label": label,
        }

        return output


@dataclass
class DataCollator:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        labels = torch.LongTensor([feat.pop("label") for feat in features])
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            **batch,
            "label": labels,
        }

        return batch


class MNLILightningDataModule(pl.LightningDataModule):
    def __init__(
        self, model_name, batch_size, num_workers, train_src, validation_src, test_src
    ):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.collate_fn = DataCollator(self.tokenizer)
        self.train_src = train_src
        self.validation_src = validation_src
        self.test_src = test_src

    def prepare_data(self) -> None:
        logger.info("Dataset downloading...")
        self.train = MNLIDataset(self.tokenizer, self.train_src, self.batch_size)
        self.validation = MNLIDataset(
            self.tokenizer, self.validation_src, self.batch_size
        )
        self.test = MNLIDataset(self.tokenizer, self.test_src, self.batch_size)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.validation,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
