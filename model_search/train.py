import torch
import torch.nn as nn
import wandb
import psutil
import logging 

logger = logging.getLogger(__name__)

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    running_local_losses = {}
    local_losses = {}  
    loss = torch.tensor(0.0, device=device,requires_grad=True)
    batch_count = 0
    #torch.autograd.set_detect_anomaly(True)
    for data in train_loader:
        if data is None:
            logger.info("Warning: Data is None, skipping this batch.")
            continue

        data = data.to(device)
        optimizer.zero_grad()

        output = model(data)

       

        if isinstance(output, dict):
            # Assuming the model output is a dictionary with 'switch_scores' and the target is data.edge_y
            predicted_scores = output.get("switch_scores") # Get the 'switch_scores' tensor
            target_switches = data.edge_y # Get the target switch states
            # debug output right here:
            # print("  predicted_scores.requires_grad =", predicted_scores.requires_grad)
            # print("  predicted_scores.grad_fn       =", predicted_scores.grad_fn)
            # print("  target_switches.requires_grad  =", target_switches.requires_grad)

            # # now compute loss
            # loss = criterion(predicted_scores, target_switches.float())
            # print("  loss.requires_grad        =", loss.requires_grad)
            # print("  loss.grad_fn              =", loss.grad_fn)
            # print(type(predicted_scores), type(target_switches))
            # print("predicted_scores shape:", predicted_scores.shape)
            # print("target_switches shape:", target_switches.shape)
            # print("predicted_scores :", predicted_scores)
            # print("target_switches :", target_switches)
            if predicted_scores is not None and target_switches is not None:
                 # Squeeze the last dimension of predicted_scores if it's 1 to match target_switches shape
                if predicted_scores.ndim > target_switches.ndim and predicted_scores.shape[-1] == 1:
                    predicted_scores = predicted_scores.squeeze(-1)

                 # Ensure shapes match before calculating loss
                if predicted_scores.shape == target_switches.shape:
                     # Criterion expects input and target tensors
                     print("si")
                     loss = criterion(predicted_scores, target_switches.float()) 
                     accuracy = (predicted_scores.round() == target_switches).float().mean()
                     local_losses = {"accuracy": accuracy} 
                else:
                     # Handle shape mismatch - this might indicate a problem in your model or data
                     logger.debug(f"Warning: Shape mismatch after squeezing: predicted scores ({predicted_scores.shape}) and target switches ({target_switches.shape}). Skipping loss calculation for this batch.")
                     # Add debug prints for the batch causing the shape mismatch
                     logger.debug("--- Debugging Shape Mismatch Batch ---")
                     logger.debug("Batch keys:", data.keys())
                     logger.debug("Batch x shape:", data.x.shape)
                     logger.debug("Batch edge_index shape:", data.edge_index.shape)
                     logger.debug("Batch edge_attr shape:", data.edge_attr.shape)
                     if hasattr(data, 'batch'): logger.debug("Batch 'batch' tensor shape:", data.batch.shape)
                     if hasattr(data, 'ptr'): logger.debug("Batch 'ptr' tensor shape:", data.ptr.shape)
                     logger.debug("--------------------------------------")

                     loss = torch.tensor(0.0, requires_grad=True)
                     local_losses = {}
            else:
                 logger.debug("Warning: 'switch_scores' not found in model output or 'data.edge_y' is missing. Skipping loss calculation for this batch.")
                 loss = torch.tensor(0.0, requires_grad=True)
                 local_losses = {}

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        for key, value in local_losses.items():
            if torch.is_tensor(value):
                running_local_losses[key] = running_local_losses.get(key, 0) + float(value.item())
            else:
                running_local_losses[key] = running_local_losses.get(key, 0) + float(value)
        batch_count += 1

    avg_loss = running_loss / batch_count if batch_count > 0 else 0.0
    log_dict = {"train_total_loss": avg_loss}

    for key, value in running_local_losses.items():
        log_dict[f"train_{key}"] = value / batch_count
    #log_dict.update(log_hardware_usage())
    return avg_loss, log_dict

def compute_switch_metrics(
        scores: torch.Tensor,
        targets: torch.Tensor,
        threshold: float = 0.5,
        eps: float = 1e-8,
) -> dict:

    # --- squeeze any trailing singleton dimension --------------------------------
    if scores.ndim == targets.ndim + 1 and scores.size(-1) == 1:
        scores = scores.squeeze(-1)

    # both → float32 for all computations
    targets = targets.float()
    # binarise predictions
    preds   = (scores > threshold).float()

    tp = (preds * targets).sum()              # 1 · 1  ➜  TP
    fp = (preds * (1 - targets)).sum()        # 1 · 0  ➜  FP
    fn = ((1 - preds) * targets).sum()        # 0 · 1  ➜  FN
    tn = ((1 - preds) * (1 - targets)).sum()  # 0 · 0  ➜  TN

    accuracy  = (tp + tn) / (tp + fp + fn + tn + eps)
    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)

    # “Intersection over Union” (a.k.a. Jaccard index)
    jaccard   = tp / (tp + fp + fn + eps)

    # Dice coefficient (identical to F1 for binary, but kept for clarity)
    dice      = 2 * tp / (2 * tp + fp + fn + eps)

    return {
        "accuracy":  accuracy.item(),
        "precision": precision.item(),
        "recall":    recall.item(),
        "f1":        f1.item(),
        "jaccard":   jaccard.item(),
        "dice":      dice.item(),
        "tp": tp.item(), "fp": fp.item(),
        "fn": fn.item(), "tn": tn.item(),
    }

def test(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_local_losses = {}
    local_losses = {}  
    loss = torch.tensor(0.0, device=device,requires_grad=True)
    batch_count = 0
    

    total_switched = 0
    total_edges = 0

    with torch.no_grad():
        for data in loader:
            if data is None:
                logger.info("Warning: Data is None, skipping this batch.")
                continue

            data = data.to(device)

            output = model(data)

            if isinstance(output, dict):
                # Assuming the model output is a dictionary with 'switch_scores' and the target is data.edge_y
                predicted_scores = output.get("switch_scores") # Get the 'switch_scores' tensor
                target_switches = data.edge_y # Get the target switch states

                if predicted_scores is not None and target_switches is not None:
                    # Squeeze the last dimension of predicted_scores if it's 1 to match target_switches shape
                    if predicted_scores.ndim > target_switches.ndim and predicted_scores.shape[-1] == 1:
                        
                        predicted_scores = predicted_scores.squeeze(-1)

                    # Ensure shapes match before calculating loss
                    if predicted_scores.shape == target_switches.shape:
                        # Criterion expects input and target tensors
                        loss = criterion(predicted_scores, target_switches.float()) # Ensure target is float if needed by criterion
                        metrics = compute_switch_metrics(predicted_scores, target_switches)
                        # add each metric to the running dict
                        for k, v in metrics.items():
                            running_local_losses[k] = running_local_losses.get(k, 0.0) + v# Initialize local_losses if your criterion doesn't return them
                    else:
                        # Handle shape mismatch - this might indicate a problem in your model or data
                        logger.debug(f"Warning: Shape mismatch after squeezing: predicted scores ({predicted_scores.shape}) and target switches ({target_switches.shape}). Skipping loss calculation for this batch.")
                        # Add debug prints for the batch causing the shape mismatch
                        logger.debug("--- Debugging Shape Mismatch Batch ---")
                        logger.debug("Batch keys:", data.keys())
                        logger.debug("Batch x shape:", data.x.shape)
                        logger.debug("Batch edge_index shape:", data.edge_index.shape)
                        logger.debug("Batch edge_attr shape:", data.edge_attr.shape)
                        if hasattr(data, 'batch'): logger.debug("Batch 'batch' tensor shape:", data.batch.shape)
                        if hasattr(data, 'ptr'): logger.debug("Batch 'ptr' tensor shape:", data.ptr.shape)
                        logger.debug("--------------------------------------")

                        loss = torch.tensor(0.0, requires_grad=True) # Assign a zero loss to continue training
                        local_losses = {}
                else:
                    logger.warning("Warning: 'switch_scores' not found in model output or 'data.edge_y' is missing. Skipping loss calculation for this batch.")
                    loss = torch.tensor(0.0, requires_grad=True)
                    local_losses = {}

            running_loss += loss.item()
            for key, value in local_losses.items():
                if torch.is_tensor(value):
                    running_local_losses[key] = running_local_losses.get(key, 0) + value.item()
                else:
                    running_local_losses[key] = running_local_losses.get(key, 0) + value
            
            batch_count+=1
            # Check and accumulate the number of switches changed
            if isinstance(output, dict) and "switch_scores" in output and hasattr(data, 'edge_attr'):
                # Ensure the edge_attr tensor has at least three columns
                if data.edge_attr.size(1) >= 3:
                    # Original switch states are assumed to be stored in the third column (index 2)
                    original_switch_states = data.edge_attr[:, 2]
                    updated_switch_scores = output["switch_scores"]
                    # Threshold updated switch scores to get binary predictions (assumes 0.5 as threshold)
                    updated_switches = (updated_switch_scores > 0.5).float()
                    
                    # Determine if the model outputs are per-edge or per-graph:
                    if updated_switch_scores.numel() == original_switch_states.numel():
                         # Threshold updated switch scores to get binary predictions (assumes 0.5 as threshold)
                         updated_switches = (updated_switch_scores > 0.5).float()

                         # Compare flattened tensors to handle potential batching correctly
                         batch_switched = torch.sum(updated_switches.view(-1) != original_switch_states.view(-1)).item()
                         total_switched += batch_switched
                         total_edges += original_switch_states.numel()
                    else:
                         logger.warning(f"Warning: Mismatched shapes for switch state comparison: model output {updated_switch_scores.shape}, target {original_switch_states.shape}. Skipping switch count for this batch.")
                         # Still increment batch_count even if switch comparison fails


    if batch_count == 0:
        logger.warning("Warning: No valid batches were processed in the loader.")
        return 0.0, {}
    else:
        avg_loss = running_loss / batch_count
        log_dict = {"test_loss": avg_loss}
        for key, value in running_local_losses.items():
            log_dict[f"test_{key}"] = value / batch_count

        # Calculate overall percentage of switches changed.
        if total_edges > 0:
            percent_switched = total_switched / total_edges * 100
        else:
            percent_switched = 0

        log_dict["num_switched"] = total_switched
        log_dict["percent_switched"] = percent_switched
        #log_dict.update(log_hardware_usage())
        return avg_loss, log_dict
    
    
def log_hardware_usage():
    metrics = {
        "cpu_percent": psutil.cpu_percent(),
        "memory_used_MB": psutil.virtual_memory().used / (1024 ** 2),
        "memory_total_MB": psutil.virtual_memory().total / (1024 ** 2),
    }

    if torch.cuda.is_available():
        metrics.update({
            "gpu_memory_MB": torch.cuda.memory_allocated() / (1024 ** 2),
            "gpu_memory_max_MB": torch.cuda.max_memory_allocated() / (1024 ** 2),
        })

    return metrics
