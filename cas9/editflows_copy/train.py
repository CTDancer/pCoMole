import os
import argparse
from datasets import load_from_disk
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from data import data
from model.base_models import EditFlow, ProteinEditFlowModel, SMILESEditFlowModel, ReparameterizedProteinEditFlowModel, ReparameterizedSMILESEditFlowModel
from smiles_tokenizer.my_tokenizers import SMILES_SPE_Tokenizer
from smiles_tokenizer.selfies_tokenizers import SelfiesTokenizer

from logic import flow
from transformers import EsmTokenizer
import datetime

import yaml
from easydict import EasyDict as edict

import pdb

DEFAULT_CONFIG_PATH = 'configs/config_test.yaml'

def main(config_path=None):
    # Use provided config path or default to hardcoded path
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    cfg = edict(config_dict)

    print(f"cfg: {cfg}")

    run_name = f"lr{cfg.optim.lr}_epoch{cfg.optim.n_epochs}_scale{cfg.model.scale_size}_optimal{cfg.model.p_optimal}_{cfg.logging.run_name}"
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
        if cfg.training.reparameterize:
            model = ReparameterizedProteinEditFlowModel(vocab_size=vocab_size, pad_id=pad_id, config=cfg.model)
        else:
            model = ProteinEditFlowModel(vocab_size=vocab_size, pad_id=pad_id, config=cfg.model)
            # model = SMILESEditFlowModel(vocab_size=vocab_size, pad_id=pad_id, config=cfg.model)

        if cfg.training.no_training:
            print(f"No training mode: freezing all parameters")
            for param in model.parameters():
                param.requires_grad = False
            
            first_param = next(model.parameters())
            first_param.requires_grad = True
            print(f"Keeping parameter with shape {first_param.shape} with requires_grad=True for backward compatibility")

    # if cfg.task == 'protein':
    #     tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    #     vocab_size = tokenizer.vocab_size
    #     source_distribution = flow.get_source_distribution(
    #         source_distribution=cfg.flow.source_distribution, vocab_size=vocab_size, special_token_ids=[0,1,2,3, 24, 25, 26, 27, 28, 29, 30, 31]
    #     )
    #     pad_id = 1
    #     bos_id = 0
    #     eos_id = 2
    #     if cfg.training.reparameterize:
    #         model = ReparameterizedProteinEditFlowModel(vocab_size=vocab_size, pad_id=pad_id, config=cfg.model)
    #     else:
    #         model = ProteinEditFlowModel(vocab_size=vocab_size, pad_id=pad_id, config=cfg.model)
    #         # model = SMILESEditFlowModel(vocab_size=vocab_size, pad_id=pad_id, config=cfg.model)

    #     if cfg.training.no_training:
    #         print(f"No training mode: freezing all parameters")
    #         for param in model.parameters():
    #             param.requires_grad = False
            
    #         first_param = next(model.parameters())
    #         first_param.requires_grad = True
    #         print(f"Keeping parameter with shape {first_param.shape} with requires_grad=True for backward compatibility")

            
    # if cfg.task == 'protein':
    #     tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    #     vocab_size = 24
    #     source_distribution = flow.get_source_distribution(source_distribution=cfg.flow.source_distribution, vocab_size=vocab_size, special_token_ids=[0,1,2,3])

    #     pad_id = 1
    #     bos_id = 0
    #     eos_id = 2
    #     model = ProteinEditFlowModel(vocab_size=vocab_size, pad_id=pad_id, config=cfg.model)
    elif cfg.task == 'smiles':
        vocab_size = 587
        tokenizer = SMILES_SPE_Tokenizer('smiles_tokenizer/new_vocab.txt',
                                         'smiles_tokenizer/new_splits.txt')
        source_distribution = flow.get_source_distribution(
            source_distribution=cfg.flow.source_distribution, vocab_size=vocab_size, special_token_ids=[0,1,2,3,4]
        )
        pad_id = 0
        bos_id = 2
        eos_id = 3

        if cfg.training.reparameterize:
            model = ReparameterizedSMILESEditFlowModel(vocab_size=vocab_size, pad_id=pad_id, config=cfg.model)
        else:
            model = SMILESEditFlowModel(vocab_size=vocab_size, pad_id=pad_id, config=cfg.model)
    elif cfg.task == 'selfies':
        vocab_size = 44
        tokenizer = SelfiesTokenizer.load(cfg.data.get('tokenizer_path', 'path/to/tokenizer/vocab.json'))
        source_distribution = flow.get_source_distribution(
            source_distribution=cfg.flow.source_distribution, vocab_size=vocab_size, special_token_ids=[0,1,2]
        )
        pad_id = 0
        bos_id = 1
        eos_id = 2
        if cfg.training.reparameterize:
            print(f"Using reparameterized SMILES edit flow model")
            model = ReparameterizedSMILESEditFlowModel(vocab_size=vocab_size, pad_id=pad_id, config=cfg.model)
        else:
            model = SMILESEditFlowModel(vocab_size=vocab_size, pad_id=pad_id, config=cfg.model)
    else:
        raise NotImplementedError

    num_parameters = sum(p.numel() for p in model.parameters())
    print(f"NUM PARAMETERS: {num_parameters}")
    
    # Print accumulation batch size
    grad_accumulation_steps = getattr(cfg.training, 'grad_accumulation_steps', 1)
    batch_size = getattr(cfg.training, 'batch_size', 1)
    num_gpus = getattr(cfg.compute, 'ngpus', 1)
    effective_batch_size = batch_size * grad_accumulation_steps * num_gpus
    print(f"GRADIENT ACCUMULATION STEPS: {grad_accumulation_steps}")
    print(f"EFFECTIVE BATCH SIZE (batch_size * grad_accum_steps * num_gpus): {effective_batch_size} (batch_size={batch_size}, grad_accum_steps={grad_accumulation_steps}, num_gpus={num_gpus})")
    print(f"LAM_PROP: {getattr(cfg.training, 'lam_prop', None)}")

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

    
    print("--------------------------------")
    print(f"run_name: {run_name}")
    print(f"config.training.loc_prop_path: {getattr(cfg.training, 'loc_prop_path', False)}")
    print(f"config.model.d_model: {cfg.model.d_model}")
    print("--------------------------------")

    # Dataloader
    if cfg.task == 'protein':
        # train_dataloader, val_dataloader = data.get_data_loaders(config=cfg, data_state=None)
        train_dataset = load_from_disk(cfg.data.train_path)
        val_dataset = load_from_disk(cfg.data.val_path)
        
        # Apply debug subset if specified
        subset_ratio = getattr(cfg.data, 'debug_subset_ratio', None)
        if subset_ratio is not None and subset_ratio < 1.0:
            train_size = len(train_dataset)
            train_subset_size = int(train_size * subset_ratio)
            train_dataset = train_dataset.shuffle(seed=cfg.training.seed).select(range(train_subset_size))
            
            val_size = len(val_dataset)
            val_subset_size = int(val_size * subset_ratio)
            val_dataset = val_dataset.select(range(val_subset_size))
            
            print(f"DEBUG MODE: Using {train_subset_size}/{train_size} training samples ({subset_ratio*100:.1f}%) and {val_subset_size}/{val_size} validation samples ({subset_ratio*100:.1f}%)")
        else:
            train_dataset = train_dataset.shuffle(seed=cfg.training.seed)
        
        print(f"train_dataset: {train_dataset[0]['input_ids']}")
        print(f"train_dataset length: {len(train_dataset[0]['input_ids'])}")
        print(f"train_dataset keys: {train_dataset[0].keys()}")
        train_dataloader = DataLoader(train_dataset, batch_size=None, shuffle=True, num_workers=4)
        val_dataloader = DataLoader(val_dataset, batch_size=None, shuffle=False, num_workers=4)
    elif cfg.task == 'smiles':
        train_dataset = load_from_disk(cfg.data.train_path)
        val_dataset = load_from_disk(cfg.data.val_path)
        train_dataloader = DataLoader(train_dataset, batch_size=None, shuffle=True, num_workers=4)
        val_dataloader = DataLoader(val_dataset, batch_size=None, shuffle=False, num_workers=4)
    elif cfg.task == 'selfies':
        train_dataset = load_from_disk(cfg.data.train_path)
        val_dataset = load_from_disk(cfg.data.val_path)
        
        # Apply debug subset if specified
        subset_ratio = getattr(cfg.data, 'debug_subset_ratio', None)
        if subset_ratio is not None and subset_ratio < 1.0:
            train_size = len(train_dataset)
            train_subset_size = int(train_size * subset_ratio)
            train_dataset = train_dataset.shuffle(seed=cfg.training.seed).select(range(train_subset_size))
            
            val_size = len(val_dataset)
            val_subset_size = int(val_size * subset_ratio)
            val_dataset = val_dataset.select(range(val_subset_size))
            
            print(f"DEBUG MODE: Using {train_subset_size}/{train_size} training samples ({subset_ratio*100:.1f}%) and {val_subset_size}/{val_size} validation samples ({subset_ratio*100:.1f}%)")
        else:
            train_dataset = train_dataset.shuffle(seed=cfg.training.seed)
        
        print(f"train_dataset: {train_dataset[0]['input_ids']}")
        print(f"train_dataset length: {len(train_dataset[0]['input_ids'])}")
        train_dataloader = DataLoader(train_dataset, batch_size=None, shuffle=True, num_workers=4)
        val_dataloader = DataLoader(val_dataset, batch_size=None, shuffle=False, num_workers=4)
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

    # Get checkpoint path from config (if specified)
    ckpt_path = getattr(cfg.training, 'ckpt_path', None) or getattr(cfg, 'ckpt_path', None)
    
    # Only resume wandb if we're actually resuming from a checkpoint
    # Otherwise, start a fresh run to avoid step counter mismatches
    # Setting resume=None ensures a fresh run, preventing "step less than current step" warnings
    wandb_resume = "allow" if ckpt_path else None
    
    wandb_logger = WandbLogger(
        project=cfg.logging.get('project', 'Gated proposal model'),
        name=run_name,
        entity=cfg.logging.get('entity') or None,
        resume=wandb_resume,
    )
    
    trainer = pl.Trainer(
        default_root_dir=workdir,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=cfg.compute.ngpus,
        strategy="ddp_find_unused_parameters_true" if cfg.compute.ngpus > 1 else "auto",
        precision='bf16-mixed',
        max_epochs=cfg.optim.n_epochs,
        log_every_n_steps=10,
        callbacks=[ckpt, lrmon],
        enable_checkpointing=True,
        gradient_clip_val=1.0,
        deterministic=False,
        logger=wandb_logger,
    )

    trainer.fit(editflow, train_dataloader, val_dataloader, ckpt_path=ckpt_path if ckpt_path else None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train EditFlow model')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config YAML file (default: uses hardcoded default config)')
    args = parser.parse_args()
    print(f"args.config: {args.config}")
    main(config_path=args.config)