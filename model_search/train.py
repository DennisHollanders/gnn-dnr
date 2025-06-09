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
    for data in train_loader:
        if data is None:
            logger.debug("Warning: Data is None, skipping this batch.")
            continue

        data = data.to(device)
        optimizer.zero_grad()

        output = model(data)
        final_switch_scores = output.get("switch_scores", None)

        if final_switch_scores is not None and isinstance(final_switch_scores, torch.Tensor):
            logger.debug(f"Gradient Check: 'switch_scores' tensor requires_grad: {final_switch_scores.requires_grad}")
            logger.debug(f"Gradient Check: 'switch_scores' tensor grad_fn: {final_switch_scores.grad_fn}")
        else:
            logger.warning("'switch_scores' not found in model output or is not a tensor.")

        if isinstance(output, dict):
        
            predicted_scores = output.get("switch_scores") 
            target_switches = data.edge_y 
            logger.debug(f"predicted_scores shape: {predicted_scores.shape if predicted_scores is not None else 'None'}")
            logger.debug(f"target_switches shape: {target_switches.shape if target_switches is not None else 'None'}")
            logger.debug(f"mean predicted_scores: {predicted_scores.mean().item() if predicted_scores is not None else 'None'}")
            logger.debug(f"mean target_switches: {target_switches.float().mean().item() if target_switches is not None else 'None'}")

            if predicted_scores is not None and target_switches is not None:
                
                if predicted_scores.dim() == 2 and predicted_scores.size(0) == 1:
                    predicted_scores = predicted_scores.squeeze(0)
                    logger.debug(f"Squeezed batch dimension: new shape {predicted_scores.shape}")

                if predicted_scores.ndim > target_switches.ndim and predicted_scores.shape[-1] == 1:
                    predicted_scores = predicted_scores.squeeze(-1)
                    logger.debug(f"Shape of predicted_scores: {predicted_scores.shape}")
                    logger.debug(f"Shape of target_switches: {target_switches.shape}")

                if predicted_scores.shape == target_switches.shape:
                    loss = criterion(predicted_scores, target_switches.float()) 
                    logger.debug(f"Loss calculated: {loss.item()}")
                    logger.debug(f"predicted_scores shape: {predicted_scores}")
                    logger.debug(f"target_switches shape: {target_switches}")
                    logger.debug(f"accuracy: {torch.sum(predicted_scores == target_switches.float()).item() / target_switches.numel()}")
                    metrics = compute_switch_metrics(predicted_scores, target_switches)
                    logger.debug(f"Switch metrics: {metrics}")
                    local_losses = metrics
                else:
                    logger.debug(f"Warning: Shape mismatch after squeezing: predicted scores ({predicted_scores.shape}) and target switches ({target_switches.shape}). Skipping loss calculation for this batch.")
                    logger.debug("--- Debugging Shape Mismatch Batch ---")
                    logger.debug(f"Batch keys: {data.keys()}")
                    logger.debug(f"batch x shape: {data.x.shape}")
                    logger.debug(f"Batch edge_index shape: {data.edge_index.shape}")
                    logger.debug(f"Batch edge_attr shape: {data.edge_attr.shape}")

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
    return avg_loss, log_dict

def compute_switch_metrics(scores: torch.Tensor,targets: torch.Tensor,threshold: float = 0.5,
        eps: float = 1e-8,) -> dict:

    # --- squeeze any trailing singleton dimension --------------------------------
    if scores.ndim == targets.ndim + 1 and scores.size(-1) == 1:
        scores = scores.squeeze(-1)

    targets = targets.float()
    preds   = (scores > threshold).float()

    tp = (preds * targets).sum()              # 1 · 1  ➜  TP
    fp = (preds * (1 - targets)).sum()        # 1 · 0  ➜  FP
    fn = ((1 - preds) * targets).sum()        # 0 · 1  ➜  FN
    tn = ((1 - preds) * (1 - targets)).sum()  # 0 · 0  ➜  TN

    accuracy  = (tp + tn) / (tp + fp + fn + tn + eps)
    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)

    jaccard   = tp / (tp + fp + fn + eps)

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

                predicted_scores = output.get("switch_scores") 
                target_switches = data.edge_y 

                if predicted_scores is not None and target_switches is not None:
                    if predicted_scores.ndim > target_switches.ndim and predicted_scores.shape[-1] == 1:
                        
                        predicted_scores = predicted_scores.squeeze(-1)
                    if predicted_scores.shape == target_switches.shape:
                        loss = criterion(predicted_scores, target_switches.float()) 
                        metrics = compute_switch_metrics(predicted_scores, target_switches)
                        running_local_losses= metrics
                    else:
                        logger.debug(f"Warning: Shape mismatch after squeezing: predicted scores ({predicted_scores.shape}) and target switches ({target_switches.shape}). Skipping loss calculation for this batch.")
                        logger.debug("--- Debugging Shape Mismatch Batch ---")
                        logger.debug("Batch keys:", data.keys())
                        logger.debug("Batch x shape:", data.x.shape)
                        logger.debug("Batch edge_index shape:", data.edge_index.shape)
                        logger.debug("Batch edge_attr shape:", data.edge_attr.shape)
                        if hasattr(data, 'batch'): logger.debug("Batch 'batch' tensor shape:", data.batch.shape)
                        if hasattr(data, 'ptr'): logger.debug("Batch 'ptr' tensor shape:", data.ptr.shape)
                        logger.debug("--------------------------------------")

                        loss = torch.tensor(0.0, requires_grad=True) 
                else:
                    logger.warning("Warning: 'switch_scores' not found in model output or 'data.edge_y' is missing. Skipping loss calculation for this batch.")
                    loss = torch.tensor(0.0, requires_grad=True)


            running_loss += loss.item()            
            batch_count+=1

            if isinstance(output, dict) and "switch_scores" in output and hasattr(data, 'edge_attr'):
                if data.edge_attr.size(1) >= 3:
                    original_switch_states = data.edge_attr[:, 2]
                    updated_switch_scores = output["switch_scores"]
                    updated_switches = (updated_switch_scores > 0.5).float()
                    
                    if updated_switch_scores.numel() == original_switch_states.numel():
                         updated_switches = (updated_switch_scores > 0.5).float()

                         batch_switched = torch.sum(updated_switches.view(-1) != original_switch_states.view(-1)).item()
                         total_switched += batch_switched
                         total_edges += original_switch_states.numel()
                    else:
                         logger.warning(f"Warning: Mismatched shapes for switch state comparison: model output {updated_switch_scores.shape}, target {original_switch_states.shape}. Skipping switch count for this batch.")



    if batch_count == 0:
        logger.warning("Warning: No valid batches were processed in the loader.")
        return 0.0, {}

    avg_loss = running_loss / batch_count
    log_dict = {"test_loss": avg_loss}

    for key, value in running_local_losses.items():
        log_dict[f"test_{key}"] = value / batch_count

    return avg_loss, log_dict

