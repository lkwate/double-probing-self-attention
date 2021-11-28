import datasets
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, AutoTokenizer
from loguru import logger

__LABEL_DIC = {"entailment": 1, "contradiction": 2, "neutral": 0}


class MNLILightningDataModule(pl.LightningDataModule):
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def _transform(self, item):
        premise, hypothesis, label = (
            item["premise"],
            item["hypothesis"],
            __LABEL_DIC[item["label"]],
        )
        premise_inputs = self.tokenizer(premise)
        hypothesis_inputs = self.tokenizer(hypothesis)

        output = {
            "premise_input_ids": premise_inputs["input_ids"],
            "premise_attention_mask": premise_inputs["attention_mask"],
            "hypothesis_input_ids": hypothesis_inputs["input_ids"],
            "hypothesis_attention_mask": hypothesis_inputs["attention_mask"],
            "label": label,
        }

        return output

    def prepare_data(self) -> None:
        logger.info("Dataset downloading...")
        self.dataset = datasets.load_dataset("multi_nli")
        self.train, self.validation, self.test = (
            self.dataset["train"],
            self.dataset["validation_matched"],
            self.dataset["validation_mismatched"],
        )

        self.columns = [
            "premise_input_ids",
            "premise_attention_mask",
            "hypothesis_input_ids",
            "hypothesis_attention_mask",
            "label",
        ]

        return super().prepare_data()
