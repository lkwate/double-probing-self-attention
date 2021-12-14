import click
from loguru import logger
import pytorch_lightning as pl
from core.dataset import MNLILightningDataModule
from core.model import DpsaLightningModule
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"


@click.command()
@click.argument("model_name", type=str)
@click.argument("batch_size", type=int)
@click.argument("log_path", type=click.Path())
@click.option("--dropout_reducer", type=float, default=0.1)
@click.option("--num_layer_reducer", type=int, default=1)
@click.option("--num_class", type=int, default=3)
@click.option("--learning_rate", type=float, default=1e-4)
@click.option("--lr_factor", type=float, default=0.1)
@click.option("--lr_schedule_patience", type=float, default=4)
@click.option("--optimizer_name", type=str, default="Adam")
@click.option("--patience_early_stopping", type=float, default=5)
@click.option("--accumulate_grad_batches", type=int, default=1)
@click.option("--val_check_interval", type=float, default=0.20)
@click.option("--seed", type=int, default=2021)
@click.option("--max_epochs", type=int, default=5)
@click.option("--checkpoint_path", type=click.Path(exists=True))
@click.option("--save_top_k", type=int, default=5)
@click.option("--num_workers", type=int, default=4)
@click.option("--save_weights_only", is_flag=True)
@click.option("--train", is_flag=True)
@click.option("--overfit_batches", is_flag=True)
def main(
    model_name,
    batch_size,
    log_path,
    dropout_reducer,
    num_layer_reducer,
    num_class,
    learning_rate,
    lr_factor,
    lr_schedule_patience,
    optimizer_name,
    patience_early_stopping,
    accumulate_grad_batches,
    val_check_interval,
    seed,
    max_epochs,
    checkpoint_path,
    save_top_k,
    num_workers,
    save_weights_only,
    train,
    overfit_batches,
):
    pl.seed_everything(seed)
    logger.info("Lightning Data module creation...")
    data_module = MNLILightningDataModule(model_name, batch_size, num_workers)

    logger.info("Lightning module creation...")
    model_config = {
        "model_name": model_name,
        "dropout_reducer": dropout_reducer,
        "num_layer_reducer": num_layer_reducer,
        "num_class": num_class,
        "learning_rate": learning_rate,
        "lr_factor": lr_factor,
        "lr_schedule_patience": lr_schedule_patience,
        "optimizer_name": optimizer_name,
    }
    if checkpoint_path is not None:
        logger.info(f"Initialize the model from checkpoint ...{checkpoint_path[-50:]}")
        model = DpsaLightningModule.load_from_checkpoint(
            checkpoint_path, **model_config
        )
    else:
        logger.info("Initialize the model from the the pretrained checkpoint")
        model = DpsaLightningModule(**model_config)

    trainer_config = {
        "default_root_dir": log_path,
        "max_epochs": max_epochs,
        "val_check_interval": val_check_interval,
        "accumulate_grad_batches": accumulate_grad_batches,
    }
    if torch.cuda.is_available():
        trainer_config["gpus"] = -1

    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=patience_early_stopping, verbose=False, strict=True
    )

    model_checkpoint_callback = ModelCheckpoint(
        dirpath=log_path,
        filename="{epoch}-{val_accuracy:.3f}",
        monitor="val_accuracy",
        save_top_k=save_top_k,
        save_weights_only=save_weights_only,
    )

    trainer_config["callbacks"] = [early_stopping_callback, model_checkpoint_callback]
    if overfit_batches:
        trainer = pl.Trainer(overfit_batches=10)
    else:
        trainer = pl.Trainer(**trainer_config)
        
    if train:
        logger.info("Training...")
        trainer.fit(model=model, datamodule=data_module)

        logger.info("Testing...")
        trainer.test(model=model, datamodule=data_module)
    else:
        if checkpoint_path is None:
            logger.error("The checkpoint_path should be defined for the test step")
        logger.info("Testing...")
        trainer.test(model=model, datamodule=data_module)


if __name__ == "__main__":
    main()
