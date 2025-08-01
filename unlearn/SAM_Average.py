import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from copy import deepcopy
from time import time
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from trainer import validate
from itertools import cycle
from torch.optim import Adam
import time 


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class UnlearningModel:
    def __init__(self, model, args):
        self.model = model.to(DEVICE)
        self.args = args
        self.device = DEVICE
        self.distances = {'positive': [], 'negative': []}
        self.criterion = nn.CrossEntropyLoss()

    def _create_optimizers(self, lr):
        return torch.optim.Adam(self.model.parameters(), lr=lr)


    def unlearn(self, data_loaders, args):
        optimizer = Adam(self.model.parameters(), lr=args.unlearn_lr)
        total_steps = len(data_loaders["retain"]) * args.unlearn_epochs
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

        alpha = args.alpha
        beta = args.beta
        rho = args.rho

        self.model.train()

        print("--- SAM Unlearning Process Start (Corrected Logic) ---")
        for epoch in range(args.unlearn_epochs):
            epoch_start_time = time.time()
            
            forget_loader = data_loaders['forget']
            forget_iter = iter(forget_loader)
            num_retain_batches = len(data_loaders["retain"])

            for i, retain_batch in enumerate(data_loaders['retain']):
                try:
                    forget_batch = next(forget_iter)
                except StopIteration:
                    forget_iter = iter(forget_loader)
                    forget_batch = next(forget_iter)
                
                optimizer.zero_grad()

                forget_image, forget_target = forget_batch[0].to(self.device), forget_batch[1].to(self.device)
                forget_out = self.model(forget_image)
                forget_loss = self.criterion(forget_out, forget_target)
                forget_loss.backward()
                
                g_forget_w = {name: param.grad.clone() for name, param in self.model.named_parameters() if param.grad is not None}
                optimizer.zero_grad()

                with torch.no_grad():
                    epsilon_dict = {}
                    for name, param in self.model.named_parameters():
                        if name in g_forget_w:
                            ascent_grad = g_forget_w[name]
                            epsilon = rho * ascent_grad / (ascent_grad.norm() + 1e-12)
                            epsilon_dict[name] = epsilon
                            param.add_(epsilon)

                retain_image, retain_target = retain_batch[0].to(self.device), retain_batch[1].to(self.device)
                retain_out = self.model(retain_image)
                retain_loss = self.criterion(retain_out, retain_target)
                retain_loss.backward()
                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        if name in epsilon_dict:
                            param.sub_(epsilon_dict[name])

                        if name in g_forget_w and param.grad is not None:
                            g_retain_we = param.grad 
                            final_grad = alpha * g_retain_we - beta * g_forget_w[name]
                            param.grad.copy_(final_grad)
                
                optimizer.step()
                scheduler.step()
                
                if (i + 1) % 10 == 0 or (i + 1) == num_retain_batches:
                    print(f"Epoch: [{epoch+1}/{args.unlearn_epochs}], Step: [{i+1}/{num_retain_batches}], Retain Loss: {retain_loss.item():.4f}, Forget Loss: {forget_loss.item():.4f}")

            epoch_end_time = time.time()
            print(f"Epoch {epoch+1} finished in {epoch_end_time - epoch_start_time:.2f} seconds.")

        print("--- SAM Unlearning Process Finished ---")
        return self.model

        
    def _create_subset_loader(self, loader, ratio=0.1):
        num_samples = int(len(loader.dataset) * ratio)
        subset_loader = torch.utils.data.DataLoader(
            loader.dataset,
            batch_size=loader.batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(range(num_samples)),
            num_workers=loader.num_workers,
            pin_memory=loader.pin_memory,
        )
        return subset_loader

    def FT(self, retain_loader, optimizer, scheduler):
        losses = []
        for image, target in tqdm(retain_loader):
            image, target = image.to(self.device), target.to(self.device)
            
            loss = F.cross_entropy(self.model(image), target)
            losses.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
        print(f'Loss: {np.mean(losses)}')

def SAM_AVERAGE(data_loader, model, criterion, args, mask=None):
    unlearning_model = UnlearningModel(model, args)
    return unlearning_model.unlearn(data_loader, args)

