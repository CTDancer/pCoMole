import os
from datasets import load_from_disk
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from data import data
# from model.base_models import EditFlow, ProteinEditFlowModel, SMILESEditFlowModel
from model.reparam_models import EditFlow, ProteinEditFlowModel, SMILESEditFlowModel
from smiles_tokenizer.my_tokenizers import SMILES_SPE_Tokenizer
from smiles_tokenizer.selfies_tokenizers import SelfiesTokenizer

from logic import flow
from transformers import EsmTokenizer
import datetime

import yaml
from easydict import EasyDict as edict

import pdb

CONFIG_PATH = '/scratch/pranamlab/tong/pCoMol/configs/config_selfies.yaml'

def main():
    with open(CONFIG_PATH, 'r') as f:
        config_dict = yaml.safe_load(f)
    cfg = edict(config_dict)

    run_name = f"reparam_{cfg.task}_lr{cfg.optim.lr}_epoch{cfg.optim.n_epochs}_scale{cfg.model.scale_size}_optimal{cfg.model.p_optimal}"
    workdir = os.path.join(cfg.work_dir, run_name)
    os.makedirs(workdir, exist_ok=True)
    
    pl.seed_everything(cfg.training.seed, workers=True)

    # Data
    if cfg.task == 'protein':
        tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        vocab_size = 24
        source_distribution = flow.get_source_distribution(
            source_distribution=cfg.flow.source_distribution, vocab_size=vocab_size, special_token_ids=[0,1,2,3]
        )
        pad_id = 1
        bos_id = 0
        eos_id = 2
        model = ProteinEditFlowModel(vocab_size=vocab_size, pad_id=pad_id, config=cfg.model)
    elif cfg.task == 'smiles':
        vocab_size = 586
        tokenizer = SMILES_SPE_Tokenizer('/scratch/pranamlab/tong/pCoMol/editflows/smiles_tokenizer/new_vocab.txt',
                                         '/scratch/pranamlab/tong/pCoMol/editflows/smiles_tokenizer/new_splits.txt')
        source_distribution = flow.get_source_distribution(
            source_distribution=cfg.flow.source_distribution, vocab_size=vocab_size, special_token_ids=[0,1,2,3,4]
        )
        pad_id = 0
        bos_id = 2
        eos_id = 3
        model = SMILESEditFlowModel(vocab_size=vocab_size, pad_id=pad_id, config=cfg.model)
    elif cfg.task == 'selfies':
        vocab_size = 44
        tokenizer = SelfiesTokenizer.load("/scratch/pranamlab/tong/data/selfies/28k_mimetics/tokenizer/vocab.json")
        source_distribution = flow.get_source_distribution(
            source_distribution=cfg.flow.source_distribution, vocab_size=vocab_size, special_token_ids=[0,1,2]
        )
        pad_id = 0
        bos_id = 1
        eos_id = 2
        model = SMILESEditFlowModel(vocab_size=vocab_size, pad_id=pad_id, config=cfg.model)
    else:
        raise NotImplementedError

    num_parameters = sum(p.numel() for p in model.parameters())
    print(f"NUM PARAMETERS: {num_parameters}")

    eps_id = getattr(cfg.flow, "eps_id", -1)
    path = flow.get_path(scheduler_type=cfg.flow.scheduler_type, exponent=cfg.flow.exponent, eps_id=eps_id)
    loss_fn = flow.get_loss_function(loss_function=cfg.flow.loss_function, path=path)

    editflow = EditFlow(
        model,
        loss_fn,
        path,
        source_distribution,
        pad_id,
        bos_id,
        eos_id,
        cfg
    )

    # Dataloader
    if cfg.task == 'protein':
        train_dataset = load_from_disk('/scratch/pranamlab/tong/data/cas9/train')
        val_dataset = load_from_disk('/scratch/pranamlab/tong/data/cas9/validation')
        train_dataloader = DataLoader(train_dataset, batch_size=None, shuffle=True, num_workers=4)
        val_dataloader = DataLoader(val_dataset, batch_size=None, shuffle=False, num_workers=4)
    elif cfg.task == 'smiles':
        train_dataset = load_from_disk('/scratch/pranamlab/tong/data/smiles/28k_mimetics/train')
        val_dataset = load_from_disk('/scratch/pranamlab/tong/data/smiles/28k_mimetics/validation')
        train_dataloader = DataLoader(train_dataset, batch_size=None, shuffle=True, num_workers=4)
        val_dataloader = DataLoader(val_dataset, batch_size=None, shuffle=False, num_workers=4)
    elif cfg.task == 'selfies':
        train_dataset = load_from_disk('/scratch/pranamlab/tong/data/selfies/28k_mimetics/train')
        val_dataset = load_from_disk('/scratch/pranamlab/tong/data/selfies/28k_mimetics/validation')
        test_dataset  = load_from_disk('/scratch/pranamlab/tong/data/selfies/28k_mimetics/test')

        train_dataloader = DataLoader(train_dataset, batch_size=None, shuffle=True, num_workers=4)
        val_dataloader = DataLoader(val_dataset, batch_size=None, shuffle=False, num_workers=4)
        test_dataloader = DataLoader(test_dataset, batch_size=None, shuffle=False, num_workers=4)
    else:
        raise NotImplementedError
    
    ckpt = ModelCheckpoint(
        dirpath=os.path.join(workdir, "checkpoint"),
        monitor="val_loss",          # the metric you log
        mode="min",                  # lower is better
        save_top_k=3,                # keep best 3
        save_last=True,
        filename="epoch{epoch:04d}-val{val_loss:.2f}",
        auto_insert_metric_name=False,  # <- this stops the extra "val_loss=..."
    )
    lrmon = LearningRateMonitor(logging_interval="step")

    wandb_logger = WandbLogger(
        project='COPE',
        name=run_name,
        entity='programmablebio',
    )
    
    trainer = pl.Trainer(
        default_root_dir=workdir,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=cfg.compute.ngpus,
        strategy="ddp" if cfg.compute.ngpus > 1 else "auto",
        precision='bf16-mixed',
        max_epochs=cfg.optim.n_epochs,
        log_every_n_steps=10,
        callbacks=[ckpt, lrmon],
        enable_checkpointing=True,
        gradient_clip_val=1.0,
        deterministic=False,
        # logger=wandb_logger,
    )

    # trainer.fit(editflow, train_dataloader, val_dataloader)
    trainer.validate(model=editflow, dataloaders=test_dataloader, ckpt_path="/scratch/pranamlab/tong/pCoMol/peptidomimetics/outputs/reparam_selfies_lr3e-4_epoch100_scale1.0_optimal0/checkpoint/epoch0092-val14.67.ckpt")

if __name__ == '__main__':
    main()