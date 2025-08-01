import torch
import utils
from imagenet import get_x_y_from_data_dict
from torch.utils.data import DataLoader
import time
import torch.distributed as dist

def validate(val_loader, model, criterion, args):
    batch_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    
    model.eval()
    
    if hasattr(model, 'module'):
        device = next(model.module.parameters()).device
    else:
        device = next(model.parameters()).device

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            try:
                images = images.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                output = model(images)
                
                loss = criterion(output, target)

                output_contiguous = output.contiguous()
                target_contiguous = target.contiguous()
                
                acc1, acc5 = utils.accuracy(output_contiguous, target_contiguous, topk=(1, 5))
                
                if dist.is_initialized():
                    loss_tensor = torch.tensor(loss.item(), device=device)
                    acc1_tensor = torch.tensor(acc1[0].item(), device=device)
                    acc5_tensor = torch.tensor(acc5[0].item(), device=device)
                    
                    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                    dist.all_reduce(acc1_tensor, op=dist.ReduceOp.SUM)
                    dist.all_reduce(acc5_tensor, op=dist.ReduceOp.SUM)
                    
                    world_size = dist.get_world_size()
                    loss_tensor /= world_size
                    acc1_tensor /= world_size
                    acc5_tensor /= world_size
                    
                    losses.update(loss_tensor.item(), images.size(0))
                    top1.update(acc1_tensor.item(), images.size(0))
                    top5.update(acc5_tensor.item(), images.size(0))
                else:
                    losses.update(loss.item(), images.size(0))
                    top1.update(acc1[0].item(), images.size(0))
                    top5.update(acc5[0].item(), images.size(0))

                batch_time.update(time.time() - end)
                end = time.time()

                if (not dist.is_initialized() or dist.get_rank() == 0) and \
                   hasattr(args, 'print_freq') and args.print_freq > 0 and i % args.print_freq == 0:
                    print(f'Test: [{i}/{len(val_loader)}]\t'
                          f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                          f'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          f'Acc@5 {top5.val:.3f} ({top5.avg:.3f})')
                
                if i % 100 == 0:
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                if not dist.is_initialized() or dist.get_rank() == 0:
                    print(f"Error in validation batch {i}: {e}")
                continue

    if not dist.is_initialized() or dist.get_rank() == 0:
        print(f' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}')
    return top1.avg