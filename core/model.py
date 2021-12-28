import torch
import torch.nn as nn
import torch.optim as optim
from .utils import slice_transformers
from transformers import AutoConfig
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
        pivot,
    ):
        super(DpsaModel, self).__init__()
        self.base_model, self.cross_model = slice_transformers(model_name, pivot)
        config = AutoConfig.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_reducer)
        self.linear = nn.Linear(2 * config.hidden_size, num_class)

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
        premise_input_ids,
        premise_attention_mask,
        hypothesis_input_ids,
        hypothesis_attention_mask,
    ):
        premise_hidden_state = self.base_model(
            input_ids=premise_input_ids, attention_mask=premise_attention_mask
        ).last_hidden_state
        hypothesis_hidden_state = self.base_model(
            input_ids=hypothesis_input_ids, attention_mask=hypothesis_attention_mask
        ).last_hidden_state

        premise_hypothesis = self.cross_model(
            hidden_states=premise_hidden_state,
            encoder_hidden_states=hypothesis_hidden_state,
            encoder_attention_mask=self.base_model.invert_attention_mask(
                hypothesis_attention_mask
            ),
        ).last_hidden_state[:, 0, :]

        hypothesis_premise = self.cross_model(
            hidden_states=hypothesis_hidden_state,
            encoder_hidden_states=premise_hidden_state,
            encoder_attention_mask=self.base_model.invert_attention_mask(
                premise_attention_mask
            ),
        ).last_hidden_state[:, 0, :]

        # pooler_output = torch.cat([premise_hypothesis, hypothesis_premise], dim =-1)
        # batch_dim = premise_hidden_state.shape[0]
        # inputs_embeds = torch.stack(
        #     [premise_hidden_state, hypothesis_hidden_state], dim=1
        # )
        # attention_mask = torch.stack(
        #     [premise_attention_mask, hypothesis_attention_mask], dim=1
        # )
        # selection_indices = torch.randint(0, 2, (batch_dim,))
        # batch_indices = torch.arange(batch_dim)

        # inputs_embeds, encoder_hidden_states = (
        #     inputs_embeds[batch_indices, selection_indices],
        #     inputs_embeds[batch_indices, 1 - selection_indices],
        # )
        # attention_mask = attention_mask[batch_indices, 1 - selection_indices]

        # pooler_output = self.cross_model(
        #     hidden_states=inputs_embeds,
        #     encoder_hidden_states=encoder_hidden_states,
        #     encoder_attention_mask=self.base_model.invert_attention_mask(
        #         attention_mask
        #     ),
        # ).last_hidden_state[:, 0, :]
        
        pooler_output = torch.cat([premise_hypothesis, hypothesis_premise], dim=-1)
        pooler_output = self.dropout(pooler_output)
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
        pivot,
    ):
        super(DpsaLightningModule, self).__init__()
        self.model = DpsaModel(
            model_name, dropout_reducer, num_layer_reducer, num_class, pivot
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
