import torch
import torch.nn as nn
import torch.optim as optim
from .utils import slice_transformers
from transformers import AutoConfig, AutoModel
import pytorch_lightning as pl
from torch.nn.utils.rnn import pack_padded_sequence

OPTMIZER_DIC = {"Adam": optim.Adam}


class DpsaModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        dropout_reducer: float,
        num_layer_reducer: int,
        num_class: int,
    ):
        super(DpsaModel, self).__init__()
        self.main = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.pooler = nn.Linear(768, 3)


    def forward(
        self,
        premise_input_ids,
        premise_attention_mask,
        hypothesis_input_ids,
        hypothesis_attention_mask,
    ):
        input_ids = torch.cat([premise_input_ids, hypothesis_input_ids], dim=-1)
        attention_mask = torch.cat([premise_attention_mask, hypothesis_attention_mask], dim=-1)
        output = self.main(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        output = self.dropout(output)
        output = self.pooler(output)
        
        return output


class DpsaLightningModule(pl.LightningModule):

    criterion = nn.CrossEntropyLoss()

    def __init__(
        self,
        model_name,
        dropout_reducer,
        num_layer_reducer,
        num_class,
        learning_rate,
        lr_factor,
        lr_schedule_patience,
        optimizer_name,
    ):
        super(DpsaLightningModule, self).__init__()
        self.model = DpsaModel(
            model_name, dropout_reducer, num_layer_reducer, num_class
        )
        self.learning_rate = learning_rate
        self.lr_factor = lr_factor
        self.lr_schedule_patience = lr_schedule_patience
        self.optimizer_name = optimizer_name

    def configure_optimizers(self):
        optimizer = OPTMIZER_DIC.get(self.optimizer_name, optim.Adam)(
            self.model.parameters(), lr=self.learning_rate
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", factor=self.lr_factor, patience=self.lr_schedule_patience
        )
        output = {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }
        return output

    def _metric_forward(self, batch):
        (
            premise_input_ids,
            premise_attention_mask,
            hypothesis_input_ids,
            hypothesis_attention_mask,
            label,
        ) = (
            batch["premise_input_ids"],
            batch["premise_attention_mask"],
            batch["hypothesis_input_ids"],
            batch["hypothesis_attention_mask"],
            batch["label"],
        )
        logits = self.model(
            premise_input_ids,
            premise_attention_mask,
            hypothesis_input_ids,
            hypothesis_attention_mask,
        )
        label = label.long()
        loss = self.criterion(logits, label)
        prediction = torch.argmax(logits, dim=-1)
        accuracy = (prediction == label).float().mean()

        return loss, accuracy

    def training_step(self, batch, batch_idx):
        loss, accuracy = self._metric_forward(batch)
        output = {"loss": loss, "train_accuracy": accuracy}
        self.log_dict(output)
        return output

    def validation_step(self, batch, batch_idx):
        loss, accuracy = self._metric_forward(batch)
        output = {"val_loss": loss, "val_accuracy": accuracy}
        self.log_dict(output)
        return output

    def test_step(self, batch, batch_idx):
        loss, accuracy = self._metric_forward(batch)
        output = {"test_loss": loss, "test_accuracy": accuracy}
        self.log_dict(output)
        return output
