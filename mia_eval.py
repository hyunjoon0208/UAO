#!/usr/bin/env python3

import copy
import os
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict

import arg_parser
from evaluation.MIA_GPU import MIA_GPU
import unlearn
import utils


def replace_loader_dataset(dataset, batch_size, seed=1, shuffle=True):
    """Data loader for single GPU."""
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
        shuffle=shuffle,
        drop_last=False
    )


def main():
    print("=== MIA EVALUATION SCRIPT STARTED ===")
    
    args = arg_parser.parse_args()
    print(f"Arguments parsed: {vars(args)}")
    
    if args.seed:
        utils.setup_seed(args.seed)
    seed = args.seed
    print(f"Using seed: {seed}")
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")

    print("Setting up datasets and model...")
    try:
        if args.dataset == "imagenet":
            model, train_loader_full, retain_loader, val_loader, forget_loader = utils.setup_model_dataset(args)
        else:
            model, train_loader_full, val_loader, test_loader, marked_loader = utils.setup_model_dataset(args)
        print('DATASET SETUP COMPLETED')
    except Exception as e:
        print(f"ERROR in dataset setup: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("Moving model to device...")
    try:
        model = model.to(device)
        print(f"Model moved to {device}")
        print(f"Model type: {type(model)}")
        if hasattr(model, 'module'):
            print("Model has 'module' attribute (DDP wrapped)")
        else:
            print("Model is not DDP wrapped")
    except Exception as e:
        print(f"ERROR moving model to device: {e}")
        return
    
    print("Loading unlearned model checkpoint...")
    try:
        print(f"Looking for checkpoint in save_dir: {args.save_dir}")
        if not os.path.exists(args.save_dir):
            print(f"Save directory does not exist: {args.save_dir}")
            os.makedirs(args.save_dir, exist_ok=True)
            print(f"Created directory: {args.save_dir}")
        
        checkpoint = unlearn.load_unlearn_checkpoint(model, device, args)
        if checkpoint is not None:
            print("Unlearn checkpoint found and loaded")
            model, evaluation_result = checkpoint
            if evaluation_result is None:
                evaluation_result = {}
                print("No evaluation results in checkpoint")
            else:
                print(f"Existing evaluation results: {list(evaluation_result.keys())}")
        else:
            print("No unlearn checkpoint found, loading base model...")
            evaluation_result = {}
            if args.model_path:
                print(f"Loading base model from {args.model_path}")
                if not os.path.exists(args.model_path):
                    print(f"ERROR: Model path does not exist: {args.model_path}")
                    return
                
                checkpoint = torch.load(args.model_path, map_location=device, weights_only=True)
                print(f"Base checkpoint loaded, keys: {list(checkpoint.keys())}")
                
                if "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                elif "model" in checkpoint:
                    state_dict = checkpoint["model"]
                else:
                    state_dict = checkpoint
                
                print(f"State dict has {len(state_dict)} parameters")
                new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                model.load_state_dict(new_state_dict, strict=False)
                print("Base model loaded successfully")
            else:
                print("No model_path specified")
    except Exception as e:
        print(f"ERROR loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        evaluation_result = {}
    
    print('MODEL LOADING COMPLETED')

    print("Preparing data loaders...")
    try:
        if not args.dataset == "imagenet":
            print("Processing non-ImageNet dataset...")
            forget_dataset = copy.deepcopy(marked_loader.dataset)
            print(f"Original marked dataset size: {len(marked_loader.dataset)}")
            
            if args.dataset == "svhn":
                print("Processing SVHN dataset...")
                try:
                    marked = forget_dataset.targets < 0
                except:
                    marked = forget_dataset.labels < 0
                forget_dataset.data = forget_dataset.data[marked]
                try:
                    forget_dataset.targets = -forget_dataset.targets[marked] - 1
                except:
                    forget_dataset.labels = -forget_dataset.labels[marked] - 1
                forget_loader = replace_loader_dataset(forget_dataset, args.batch_size, seed=seed, shuffle=True)
                
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
                retain_loader = replace_loader_dataset(retain_dataset, args.batch_size, seed=seed, shuffle=True)
            else:
                print("Processing other dataset (CIFAR/ImageNet)...")
                try:
                    print("Trying dataset.dataset access...")
                    marked = forget_dataset.dataset.targets < 0
                    print(f"Found {marked.sum()} forget samples using dataset.targets")
                    forget_dataset.data = forget_dataset.dataset.data[marked]
                    forget_dataset.targets = -forget_dataset.dataset.targets[marked] - 1
                    forget_loader = replace_loader_dataset(forget_dataset, args.batch_size, seed=seed, shuffle=True)
                    
                    retain_dataset = copy.deepcopy(marked_loader.dataset)
                    marked = retain_dataset.dataset.targets >= 0
                    print(f"Found {marked.sum()} retain samples using dataset.targets")
                    retain_dataset.data = retain_dataset.dataset.data[marked]
                    retain_dataset.targets = retain_dataset.dataset.targets[marked]
                    retain_loader = replace_loader_dataset(retain_dataset, args.batch_size, seed=seed, shuffle=True)
                except Exception as inner_e:
                    print(f"dataset.dataset access failed: {inner_e}")
                    print("Trying direct access...")
                    marked = forget_dataset.targets < 0
                    print(f"Found {marked.sum()} forget samples using direct targets")
                    forget_dataset.imgs = forget_dataset.imgs[marked]
                    forget_dataset.targets = -forget_dataset.targets[marked] - 1
                    forget_loader = replace_loader_dataset(forget_dataset, args.batch_size, seed=seed, shuffle=True)
                    
                    retain_dataset = copy.deepcopy(marked_loader.dataset)
                    marked = retain_dataset.targets >= 0
                    print(f"Found {marked.sum()} retain samples using direct targets")
                    retain_dataset.imgs = retain_dataset.imgs[marked]
                    retain_dataset.targets = retain_dataset.targets[marked]
                    retain_loader = replace_loader_dataset(retain_dataset, args.batch_size, seed=seed, shuffle=True)
        
        print("Preparing test loader...")
        test_loader = replace_loader_dataset(val_loader.dataset, args.batch_size, shuffle=False)
        
        print('DATA LOADERS COMPLETED')
        print(f"Number of retain dataset: {len(retain_loader.dataset)}")
        print(f"Number of forget dataset: {len(forget_loader.dataset)}")
        print(f"Number of test dataset: {len(test_loader.dataset)}")
        
    except Exception as e:
        print(f"ERROR in data loader preparation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("Preparing MIA evaluation data...")
    try:
        retain_eval_dataset = retain_loader.dataset
        forget_eval_dataset = forget_loader.dataset  
        test_eval_dataset = test_loader.dataset
        
        print(f"Retain eval dataset size: {len(retain_eval_dataset)}")
        print(f"Forget eval dataset size: {len(forget_eval_dataset)}")
        print(f"Test eval dataset size: {len(test_eval_dataset)}")

        num_retain_samples = len(retain_eval_dataset)
        indices = list(range(num_retain_samples))
        np.random.seed(seed)
        np.random.shuffle(indices)
        
        split = num_retain_samples // 2
        shadow_indices = indices[:split]
        target_indices = indices[split:]
        
        print(f"Shadow train size: {len(shadow_indices)}")
        print(f"Target train size: {len(target_indices)}")
        
        shadow_train_dataset = torch.utils.data.Subset(retain_eval_dataset, shadow_indices)
        target_train_dataset = torch.utils.data.Subset(retain_eval_dataset, target_indices)

        eval_batch_size = min(args.batch_size, 256)
        print(f"Using MIA evaluation batch size: {eval_batch_size}")
        
        shadow_train_loader = torch.utils.data.DataLoader(
            shadow_train_dataset, 
            batch_size=eval_batch_size, 
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )
        target_train_loader = torch.utils.data.DataLoader(
            target_train_dataset, 
            batch_size=eval_batch_size, 
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )
        forget_loader_eval = torch.utils.data.DataLoader(
            forget_eval_dataset, 
            batch_size=eval_batch_size, 
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )
        test_loader_eval = torch.utils.data.DataLoader(
            test_eval_dataset, 
            batch_size=eval_batch_size, 
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )
        
        print("MIA data loaders created successfully")
        
    except Exception as e:
        print(f"ERROR in MIA data preparation: {e}")
        import traceback
        traceback.print_exc()
        return

    print("=== STARTING MIA EVALUATION ===")
    try:
        print("Calling MIA_GPU function...")
        print(f"Model device: {next(model.parameters()).device}")
        print(f"Evaluation device: {device}")
        print(f"num_classes: {args.num_classes}")
        
        mia_results = MIA_GPU(
            shadow_train_loader,
            target_train_loader,
            forget_loader_eval,
            test_loader_eval,
            model,
            device,
            num_classes=args.num_classes
        )
        evaluation_result['MIA_GPU'] = mia_results
        print(f"MIA Results: {mia_results}")
        
        print("Saving final results...")
        unlearn.save_unlearn_checkpoint(model, evaluation_result, args)
        print("MIA evaluation completed and saved.")
        
    except Exception as e:
        print(f"ERROR in MIA evaluation: {e}")
        import traceback
        traceback.print_exc()
        print("MIA evaluation failed but script will continue...")
    
    print("=== MIA EVALUATION SCRIPT COMPLETED ===")


if __name__ == "__main__":
    main()