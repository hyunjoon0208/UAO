import torch
import torch.nn.functional as F
import numpy as np
import time

def _log_value(probs, eps=1e-30):
    """Logarithm with numerical stability."""
    return -torch.log(torch.clamp(probs, min=eps))

def _entr_comp(probs):
    """Entropy calculation."""
    return torch.sum(probs * _log_value(probs), dim=1)

def _calculate_cross_entropy_loss(outputs, labels):
    """Calculate cross-entropy loss for each sample."""
    eps = 1e-30
    outputs_clamped = torch.clamp(outputs, min=eps, max=1-eps)
    
    log_probs = torch.log(outputs_clamped)
    losses = -torch.gather(log_probs, 1, labels.unsqueeze(1)).squeeze()
    
    return losses

def _m_entr_comp(probs, true_labels):
    """Modified entropy calculation."""
    log_probs = _log_value(probs)
    reverse_probs = 1 - probs
    log_reverse_probs = _log_value(reverse_probs)
    
    true_label_mask = F.one_hot(true_labels, num_classes=probs.shape[1]).bool()
    
    modified_probs = torch.where(true_label_mask, reverse_probs, probs)
    modified_log_probs = torch.where(true_label_mask, log_reverse_probs, log_probs)
    
    return torch.sum(modified_probs * modified_log_probs, dim=1)

def _find_best_threshold_vectorized(s_tr_values, s_te_values):
    """
    Finds the best threshold for MIA attack in a vectorized manner.
    This is the key optimization.
    """
    if len(s_tr_values) == 0 or len(s_te_values) == 0:
        return 0.0, 0.5
        
    all_values = torch.cat([s_tr_values, s_te_values])
    possible_thresholds = torch.unique(all_values)
    
    tr_above_threshold = s_tr_values.unsqueeze(0) >= possible_thresholds.unsqueeze(1)
    te_below_threshold = s_te_values.unsqueeze(0) < possible_thresholds.unsqueeze(1)
    
    tr_ratio = tr_above_threshold.float().mean(dim=1)
    te_ratio = te_below_threshold.float().mean(dim=1)
    
    accuracies = 0.5 * (tr_ratio + te_ratio)
    max_acc = accuracies.max()
    
    return max_acc

def _mem_inf_thre_gpu(v_name, s_tr_values, s_te_values, t_tr_values, t_te_values, s_tr_labels, s_te_labels, t_tr_labels, t_te_labels, num_classes):
    """Performs threshold-based MIA for a given metric (e.g., confidence) on GPU."""
    total_tr_acc = 0.0
    total_te_acc = 0.0
    
    print(f"  Calculating thresholds for {v_name} across {num_classes} classes...")
    start_time = time.time()
    
    for num in range(num_classes):
        if (num + 1) % 10 == 0:
            print(f"    ... processed {num + 1}/{num_classes} classes.")
            
        s_tr_class = s_tr_values[s_tr_labels == num]
        s_te_class = s_te_values[s_te_labels == num]
        t_tr_class = t_tr_values[t_tr_labels == num]
        t_te_class = t_te_values[t_te_labels == num]

        if len(s_tr_class) == 0 or len(s_te_class) == 0 or len(t_tr_class) == 0 or len(t_te_class) == 0:
            continue

        value_list = torch.cat((s_tr_class, s_te_class))
        thre, max_acc = 0, 0
        for value in value_list:
            tr_ratio = torch.sum(s_tr_class >= value) / len(s_tr_class)
            te_ratio = torch.sum(s_te_class < value) / len(s_te_class)
            acc = 0.5 * (tr_ratio + te_ratio)
            if acc > max_acc:
                thre, max_acc = value, acc
        
        total_tr_acc += torch.sum(t_tr_class >= thre)
        total_te_acc += torch.sum(t_te_class < thre)

    t_tr_acc = total_tr_acc / len(t_tr_labels) if len(t_tr_labels) > 0 else 0
    t_te_acc = total_te_acc / len(t_te_labels) if len(t_te_labels) > 0 else 0
    mem_inf_acc = 0.5 * (t_tr_acc + t_te_acc)
    
    end_time = time.time()
    print(f"  Finished threshold calculation for {v_name} in {end_time - start_time:.2f} seconds.")
    print(f"  [RESULT] Membership inference attack via {v_name}: attack_acc={mem_inf_acc:.3f}, train_acc={t_tr_acc:.3f}, test_acc={t_te_acc:.3f}")
    return t_tr_acc.item(), t_te_acc.item()


def collect_performance_gpu(data_loader, model, device, name=""):
    """Collects model predictions and labels, keeping them on the GPU."""
    print(f"  Collecting performance for '{name}'...")
    start_time = time.time()
    
    probs_list = []
    labels_list = []
    model.eval()

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            prob = F.softmax(output, dim=-1)
            probs_list.append(prob)
            labels_list.append(target)

    end_time = time.time()
    print(f"  Finished collecting for '{name}' in {end_time - start_time:.2f} seconds.")
    return torch.cat(probs_list), torch.cat(labels_list)

def MIA_GPU(retain_loader_train, retain_loader_test, forget_loader, test_loader, model, device, num_classes=1000):
    """
    GPU-accelerated Membership Inference Attack benchmark.
    """
    print("\n--- Starting GPU-accelerated MIA ---")
    
    s_tr_outputs, s_tr_labels = collect_performance_gpu(retain_loader_train, model, device, name="Shadow Train (Retain-Train)")
    s_te_outputs, s_te_labels = collect_performance_gpu(test_loader, model, device, name="Shadow Test (Test)")
    t_tr_outputs, t_tr_labels = collect_performance_gpu(retain_loader_test, model, device, name="Target Train (Retain-Test)")
    t_te_outputs, t_te_labels = collect_performance_gpu(forget_loader, model, device, name="Target Test (Forget)")

    print("\nCalculating attack metrics on GPU...")
    ret = {}
    
    print("\n[1/3] Calculating attack via 'confidence'...")
    s_tr_conf = torch.gather(s_tr_outputs, 1, s_tr_labels.unsqueeze(1)).squeeze()
    s_te_conf = torch.gather(s_te_outputs, 1, s_te_labels.unsqueeze(1)).squeeze()
    t_tr_conf = torch.gather(t_tr_outputs, 1, t_tr_labels.unsqueeze(1)).squeeze()
    t_te_conf = torch.gather(t_te_outputs, 1, t_te_labels.unsqueeze(1)).squeeze()
    ret['confidence'] = _mem_inf_thre_gpu("confidence", s_tr_conf, s_te_conf, t_tr_conf, t_te_conf, s_tr_labels, s_te_labels, t_tr_labels, t_te_labels, num_classes)

    print("\n[2/3] Calculating attack via 'entropy'...")
    s_tr_entr = _entr_comp(s_tr_outputs)
    s_te_entr = _entr_comp(s_te_outputs)
    t_tr_entr = _entr_comp(t_tr_outputs)
    t_te_entr = _entr_comp(t_te_outputs)
    ret['entropy'] = _mem_inf_thre_gpu("entropy", -s_tr_entr, -s_te_entr, -t_tr_entr, -t_te_entr, s_tr_labels, s_te_labels, t_tr_labels, t_te_labels, num_classes)

    print("\n[3/3] Calculating attack via 'loss'...")
    s_tr_loss = _calculate_cross_entropy_loss(s_tr_outputs, s_tr_labels)
    s_te_loss = _calculate_cross_entropy_loss(s_te_outputs, s_te_labels)
    t_tr_loss = _calculate_cross_entropy_loss(t_tr_outputs, t_tr_labels)
    t_te_loss = _calculate_cross_entropy_loss(t_te_outputs, t_te_labels)
    
    ret['loss'] = _mem_inf_thre_gpu("loss", -s_tr_loss, -s_te_loss, -t_tr_loss, -t_te_loss, 
                                   s_tr_labels, s_te_labels, t_tr_labels, t_te_labels, num_classes)

    print("\n--- MIA evaluation complete. ---")
    return ret
