import copy
import os
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from accelerate import Accelerator
from torch.optim import Adam, AdamW

import arg_parser
import unlearn
import utils
from trainer import validate, train

def get_optimizer_and_scheduler(model, args):
    optimizer = AdamW(
        model.parameters(), 
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=0
    )
    
    return optimizer, scheduler


def replace_loader_dataset(dataset, batch_size, seed=1, shuffle=True):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True,
        shuffle=shuffle,
    )


def main():
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision='no',
        log_with=None,
        project_dir='./logs'
    )
    
    args = arg_parser.parse_args()
    
    if args.seed:
        utils.setup_seed(args.seed)
    seed = args.seed

    if accelerator.is_main_process:
        os.makedirs(args.save_dir, exist_ok=True)
    
    accelerator.wait_for_everyone()
    
    accelerator.print(f"Process {accelerator.process_index}/{accelerator.num_processes} starting...")
    accelerator.print(f"Device: {accelerator.device}")

    if args.dataset == "imagenet":
        if args.num_indexes_to_replace is not None:
            model, train_loader_full, retain_loader, val_loader, forget_loader = utils.setup_model_dataset(args)
        else:
            model, train_loader, val_loader = utils.setup_model_dataset(args)
    else:
        model, train_loader_full, val_loader, test_loader, marked_loader = utils.setup_model_dataset(args)
    
    accelerator.print('DATASET DONE')
    
    if args.model_path and not args.resume:
        if accelerator.is_main_process:
            accelerator.print(f"Loading model weights from checkpoint {args.model_path}")
            checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=True)
            
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint
            
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
            model.load_state_dict(new_state_dict, strict=False)
            accelerator.print("Checkpoint loaded successfully.")
        
        accelerator.wait_for_everyone()
    
    accelerator.print('MODEL LOADING DONE')

    if not args.dataset == "imagenet":
        forget_dataset = copy.deepcopy(marked_loader.dataset)
        
        if args.dataset == "svhn":
            try:
                marked = forget_dataset.targets < 0
            except:
                marked = forget_dataset.labels < 0
            forget_dataset.data = forget_dataset.data[marked]
            try:
                forget_dataset.targets = -forget_dataset.targets[marked] - 1
            except:
                forget_dataset.labels = -forget_dataset.labels[marked] - 1
            forget_loader = replace_loader_dataset(forget_dataset, args.batch_size, 
                                                seed=seed, shuffle=True)
            
            retain_dataset = copy.deepcopy(marked_loader.dataset)
            try:
                marked = retain_dataset.targets >= 0
            except:
                marked = retain_dataset.labels >= 0
            retain_dataset.data = retain_dataset.data[marked]
            try:
                retain_dataset.targets = retain_dataset.targets[marked]
            except:
                retain_dataset.labels = retain_dataset.labels[marked]
            retain_loader = replace_loader_dataset(retain_dataset, args.batch_size, 
                                                seed=seed, shuffle=True)
        else:
            try:
                if accelerator.is_main_process:
                    accelerator.print("Processing forget/retain datasets...")
                marked = forget_dataset.dataset.targets < 0
                forget_dataset.data = forget_dataset.dataset.data[marked]
                forget_dataset.targets = -forget_dataset.dataset.targets[marked] - 1
                forget_loader = replace_loader_dataset(forget_dataset, args.batch_size, 
                                                    seed=seed, shuffle=True)
                
                retain_dataset = copy.deepcopy(marked_loader.dataset)
                marked = retain_dataset.dataset.targets >= 0
                retain_dataset.data = retain_dataset.dataset.data[marked]
                retain_dataset.targets = retain_dataset.dataset.targets[marked]
                retain_loader = replace_loader_dataset(retain_dataset, args.batch_size, 
                                                    seed=seed, shuffle=True)
            except:
                marked = forget_dataset.targets < 0
                forget_dataset.imgs = forget_dataset.imgs[marked]
                forget_dataset.targets = -forget_dataset.targets[marked] - 1
                forget_loader = replace_loader_dataset(forget_dataset, args.batch_size, 
                                                    seed=seed, shuffle=True)
                
                retain_dataset = copy.deepcopy(marked_loader.dataset)
                marked = retain_dataset.targets >= 0
                retain_dataset.imgs = retain_dataset.imgs[marked]
                retain_dataset.targets = retain_dataset.targets[marked]
                retain_loader = replace_loader_dataset(retain_dataset, args.batch_size, 
                                                    seed=seed, shuffle=True)

    test_loader = replace_loader_dataset(val_loader.dataset, args.batch_size, shuffle=False)

    accelerator.print('DATA LOADERS DONE')
    
    if accelerator.is_main_process:
        accelerator.print(f"number of retain dataset {len(retain_loader.dataset)}")
        accelerator.print(f"number of forget dataset {len(forget_loader.dataset)}")

    model, retain_loader, forget_loader, test_loader = accelerator.prepare(
        model, retain_loader, forget_loader, test_loader
    )
    
    accelerator.print('ACCELERATE PREPARE DONE')

    unlearn_data_loaders = OrderedDict(
        retain=retain_loader, forget=forget_loader, val=None, test=test_loader
    )
    
    criterion = nn.CrossEntropyLoss()

    if args.resume:
        pass
    else:
        if args.unlearn == 'retrain':
            accelerator.print("Starting retrain from scratch with cosine annealing...")
            
            optimizer, scheduler = get_optimizer_and_scheduler(model, args)
            
            optimizer, scheduler = accelerator.prepare(optimizer, scheduler)
            
            accelerator.print(f"Using cosine annealing for {args.epochs} epochs")
            
            for epoch in range(args.epochs):
                current_lr = optimizer.param_groups[0]['lr']
                accelerator.print(f"Train Epoch [{epoch+1}/{args.epochs}] - LR: {current_lr:.6f}")
                
                train_acc = train(retain_loader, model, criterion, optimizer, epoch, args, 
                                mask=None, l1=False, accelerator=accelerator)
                
                scheduler.step()
                
                if accelerator.is_main_process:
                    accelerator.print(f"Train Epoch {epoch+1} - Train Accuracy: {train_acc:.4f}")
                
                if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
                    val_acc = validate(test_loader, model, criterion, args)
                    if accelerator.is_main_process:
                        accelerator.print(f"Epoch {epoch+1} - Validation Accuracy: {val_acc:.4f}")
                    
                    unwrapped_model = accelerator.unwrap_model(model)
                    unlearn.save_unlearn_checkpoint(unwrapped_model, None, args)
                    accelerator.print("Checkpoint saved.")
            
            accelerator.print("Retrain with cosine annealing finished.")
            
        else:
            unlearn_method = unlearn.get_unlearn_method(args.unlearn)
            
            accelerator.print(f"Starting unlearning with method: {args.unlearn}")
            
            unlearn_method(unlearn_data_loaders, model, criterion, args)
            
            accelerator.print("Unlearning finished.")

        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            unlearn.save_unlearn_checkpoint(unwrapped_model, None, args)
            accelerator.print("Checkpoint saved.")
    accelerator.wait_for_everyone()
    
    accelerator.print("Starting accuracy evaluation...")
    evaluation_result = {}
    accuracy = {}
    
    for name, loader in unlearn_data_loaders.items():
        if loader is not None and len(loader.dataset) > 0:
            accelerator.print(f"Evaluating accuracy on {name} dataset...")
            val_acc = validate(loader, model, criterion, args)
            
            if accelerator.is_main_process:
                accuracy[name] = val_acc
                accelerator.print(f"Accuracy on {name}: {val_acc:.4f}")
    
    if accelerator.is_main_process:
        evaluation_result["accuracy"] = accuracy
        accelerator.print("\n=== ACCURACY SUMMARY ===")
        for name, acc in accuracy.items():
            accelerator.print(f"{name.upper()} Accuracy: {acc:.4f}")
        accelerator.print("========================")
        
        unwrapped_model = accelerator.unwrap_model(model)
        unlearn.save_unlearn_checkpoint(unwrapped_model, evaluation_result, args)
        accelerator.print("Checkpoint saved with accuracy results.")
    
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        accelerator.print("Unlearning and accuracy evaluation completed successfully!")
        accelerator.print("To run MIA evaluation, use the separate mia_eval.py script.")
    
    accelerator.print("All unlearning tasks completed.")


if __name__ == "__main__":
    main()