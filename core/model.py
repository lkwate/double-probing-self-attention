import torch
import torch.nn as nn
import torch.optim as optim
from .utils import slice_transformers
from transformers import AutoConfig
import pytorch_lightning as pl
from torch.nn.utils.rnn import pack_padded_sequence
from torch_optimizer import Lamb
from collections import defaultdict

OPTMIZER_DIC = {"Adam": optim.Adam, "Lamb": Lamb}


class DpsaModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        dropout_reducer: float,
        num_layer_reducer: int,
        num_class: int,
    ):
        super(DpsaModel, self).__init__()
        self.base_model, _ = slice_transformers(model_name)
        config = AutoConfig.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_reducer)
        self.linear = nn.Linear(config.hidden_size, num_class)

    def _pack_mask_transformer_output(self, output, attention_mask):
        zero_indices = (1 - attention_mask).nonzero(as_tuple=True)
        lengths = attention_mask.sum(-1).cpu()
        output[zero_indices] = 0
        output = pack_padded_sequence(
            output, lengths=lengths, batch_first=True, enforce_sorted=False
        )

        return output

    def forward(
        self,
        input_ids,
        attention_mask,
    ):
        pooler_output = self.base_model(
            input_ids, attention_mask=attention_mask
        ).pooler_output
        output = self.linear(pooler_output)

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
        accumulate_grad_batches,
        log_every_n_steps,
    ):
        super(DpsaLightningModule, self).__init__()
        self.model = DpsaModel(
            model_name, dropout_reducer, num_layer_reducer, num_class
        )
        self.learning_rate = learning_rate
        self.lr_factor = lr_factor
        self.lr_schedule_patience = lr_schedule_patience
        self.optimizer_name = optimizer_name
        self.accumulate_grad_batches = accumulate_grad_batches
        self.log_every_n_steps = log_every_n_steps
        self.metrics = defaultdict(list)
        
    def load_metrics(self, inputs):
        for metric, value in inputs.items():
            self.metrics[metric].append(value.item())
            
    def reduce_metrics(self):
        output = {}
        for metric in self.metrics:
            output[metric] = torch.Tensor(self.metrics[metric]).float().mean()
        return output
    
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
        (input_ids, attention_mask, label,) = (
            batch["input_ids"],
            batch["attention_mask"],
            batch["label"],
        )
        logits = self.model(input_ids, attention_mask)
        label = label.long()
        loss = self.criterion(logits, label)
        prediction = torch.argmax(logits, dim=-1)
        accuracy = (prediction == label).float().mean()

        return loss, accuracy

    def training_step(self, batch, batch_idx):
        loss, accuracy = self._metric_forward(batch)
        output = {"loss": loss, "train_accuracy": accuracy}
        self.load_metrics(output)
        if (self.global_step + 1) % self.log_every_n_steps == 0:
            output = self.reduce_metrics()
            self.metrics = defaultdict(list)
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
