import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import psutil
import logging 

logger = logging.getLogger(__name__)

def radiality_loss(switch_probs, num_nodes):
    expected_active_edges = switch_probs.sum()
    target_edges = num_nodes - 1
    radiality_penalty = (expected_active_edges - target_edges) ** 2
    return radiality_penalty

def approximate_connectivity_loss(switch_probs, edge_index, num_nodes, device):
    row, col = edge_index
    node_degrees = torch.zeros(num_nodes, device=device)
    node_degrees.index_add_(0, row, switch_probs)
    node_degrees.index_add_(0, col, switch_probs)
    
    disconnected_penalty = torch.relu(1.0 - node_degrees).sum()
    return disconnected_penalty

def enhanced_physics_loss(output, data, device, lambda_phy=0.1, 
                         lambda_connectivity=0.05, lambda_radiality=0.05,
                         normalization_type="adaptive"):
    # Get base physics loss
    base_phy_loss = physics_loss(output, data, device, normalization_type)
    
    # Get switch probabilities
    switch_probs = output.get("switch_predictions")
    if switch_probs is None and "switch_logits" in output:
        switch_probs = torch.sigmoid(output["switch_logits"])
    
    if switch_probs is None:
        return base_phy_loss
    
    edge_index = data.edge_index
    num_nodes = data.x.size(0)
    
    # Connectivity loss
    conn_loss = approximate_connectivity_loss(switch_probs, edge_index, num_nodes, device)
    conn_loss_normalized = conn_loss / num_nodes 
    
    # Radiality loss
    rad_loss = radiality_loss(switch_probs, num_nodes)
    rad_loss_normalized = rad_loss / (num_nodes - 1)**2
    
    # Combine all losses 
    total_loss = (lambda_phy * base_phy_loss + 
                  lambda_connectivity * conn_loss_normalized + 
                  lambda_radiality * rad_loss_normalized)
    
    return total_loss

def process_batch(model, data, criterion, device, is_training=True, 
                 lambda_phy_loss=0.1, lambda_mask=0.01, 
                 lambda_connectivity=0.05, lambda_radiality=0.05,
                 normalization_type="adaptive", 
                 loss_scaling_strategy="adaptive_ratio"):
    if data is None:
        logger.debug("Warning: Data is None, skipping this batch.")
        return torch.tensor(0.0, device=device), {}, {"valid_batch": False}

    data = data.to(device)
    
    # Forward pass
    output = model(data)
   
    if hasattr(data, 'batch') and data.batch.max() >= len(data.ptr) - 1:
        logger.warning("Invalid batch structure detected, skipping batch")
        return torch.tensor(0.0, device=device), {}, {"valid_batch": False}
    
    # Initialize loss and metrics
    total_loss = torch.tensor(0.0, device=device)
    loss_components = {}
    metrics = {}
    batch_stats = {"valid_batch": False}
    
    if not isinstance(output, dict):
        logger.warning("Model output is not a dictionary")
        return total_loss, metrics, batch_stats
    
    # Get predictions and targets
    switch_logits = output.get("switch_logits")
    target_switches = data.edge_y
    
    # Debug logging
    if switch_logits is not None:
        logger.debug(f"switch_logits shape: {switch_logits.shape}")
        logger.debug(f"mean switch_logits: {switch_logits.mean().item()}")
        logger.debug(f"requires_grad: {switch_logits.requires_grad}")
        logger.debug(f"grad_fn: {switch_logits.grad_fn}")
    
    if target_switches is not None:
        logger.debug(f"target_switches shape: {target_switches.shape}")
        logger.debug(f"mean target_switches: {target_switches.float().mean().item()}")
    
    # Process predictions if available
    if switch_logits is not None and target_switches is not None:
        if hasattr(model, 'output_type') and model.output_type == "multiclass":
    
            target_classes = target_switches.long()
            
            # Use CrossEntropyLoss for multiclass
            if isinstance(criterion, nn.CrossEntropyLoss):
                switch_loss = criterion(switch_logits, target_classes)
                logger.debug(f"Using CrossEntropyLoss: {switch_loss.item()}")
            else:
                # Fallback: convert logits to binary probabilities
                switch_probs = F.softmax(switch_logits, dim=1)[:, 1]  
                switch_loss = criterion(switch_probs, target_switches.float())
                logger.debug(f"Using fallback binary loss: {switch_loss.item()}")
                
            # For metrics, use binary probabilities
            if switch_logits.dim() > 1:
                predicted_scores = F.softmax(switch_logits, dim=1)[:, 1] 
            else:
                predicted_scores = torch.sigmoid(switch_logits)
        else:
            predicted_scores = _fix_prediction_shape(switch_logits, target_switches)
            
            if predicted_scores.shape == target_switches.shape:
                switch_loss = criterion(predicted_scores, target_switches.float())
                logger.debug(f"Using binary loss: {switch_loss.item()}")
            else:
                logger.debug(f"Shape mismatch: predicted ({predicted_scores.shape}) vs target ({target_switches.shape})")
                _log_batch_debug_info(data)

                return total_loss, metrics, batch_stats
        
        print(f"Predicted scores shape: {predicted_scores.shape}")
        print(f"Target switches shape: {target_switches.shape}")
        print(f"Switch loss: {switch_loss.item()}")

        loss_components["switch_loss"] = switch_loss
        total_loss = switch_loss

        if any(key in output for key in ["flows", "node_v", "switch_predictions"]):
            try:
                with torch.set_grad_enabled(is_training):
                    physics_loss = enhanced_physics_loss(
                        output, data, device, 
                        lambda_phy=1.0,  
                        lambda_connectivity=1.0,
                        lambda_radiality=1.0,
                        normalization_type=normalization_type
                    )
                    loss_components["physics_loss"] = physics_loss
                    
                    # Apply loss scaling strategy
                    scaled_physics_loss = apply_loss_scaling(
                        switch_loss, physics_loss, 
                        lambda_phy_loss, lambda_connectivity, lambda_radiality,
                        loss_scaling_strategy
                    )
                    
                    total_loss = switch_loss + scaled_physics_loss
                    loss_components["scaled_physics_loss"] = scaled_physics_loss
                    
                    logger.debug(f"Switch loss: {switch_loss.item():.6f}")
                    logger.debug(f"Raw physics loss: {physics_loss.item():.6f}")
                    logger.debug(f"Scaled physics loss: {scaled_physics_loss.item():.6f}")
                    logger.debug(f"Total loss: {total_loss.item():.6f}")
                
            except Exception as e:
                logger.warning(f"Enhanced physics loss computation failed: {e}")
  
        if "switch_mask" in output:
            sparsity_loss = output["switch_mask"].mean()
            scaled_sparsity = lambda_mask * sparsity_loss
            total_loss = total_loss + scaled_sparsity
            loss_components["sparsity_loss"] = scaled_sparsity
            logger.debug(f"Sparsity: {sparsity_loss.item()}, Scaled: {scaled_sparsity.item()}")
        

        metrics = compute_switch_metrics(predicted_scores, target_switches)
        
        for key, value in loss_components.items():
            metrics[key] = value.item()
        
        batch_stats = {
            "valid_batch": True,
            "loss": total_loss.item(),
            "batch_size": target_switches.numel()
        }
        
        logger.debug(f"Final total loss: {total_loss.item()}")
        logger.debug(f"Switch metrics: {metrics}")
            
    else:
        logger.debug("Missing predictions or targets")
    
    return total_loss, metrics, batch_stats

def apply_loss_scaling(switch_loss, physics_loss, lambda_phy, lambda_conn, lambda_rad, strategy="adaptive_ratio"):
    if strategy == "fixed":
        return lambda_phy * physics_loss
    
    elif strategy == "adaptive_ratio":
        target_ratio = lambda_phy  
        if switch_loss.item() > 1e-8:  
            # Use .detach() to prevent gradient flow issues
            adaptive_weight = target_ratio * switch_loss.detach() / (physics_loss.detach() + 1e-8)
            return adaptive_weight * physics_loss
        else:
            return lambda_phy * physics_loss
    
    elif strategy == "adaptive_magnitude":
        if physics_loss.item() > 1e-8 and switch_loss.item() > 1e-8:
            # Use .detach() to prevent gradient flow issues
            magnitude_ratio = switch_loss.detach() / physics_loss.detach()
            smooth_ratio = torch.clamp(magnitude_ratio, 0.1, 10.0)
            return lambda_phy * smooth_ratio * physics_loss
        else:
            return lambda_phy * physics_loss
    
    elif strategy == "uncertainty_weighting":
        if not hasattr(apply_loss_scaling, 'log_vars'):
            apply_loss_scaling.log_vars = {
                'switch': nn.Parameter(torch.tensor(0.0)),
                'physics': nn.Parameter(torch.tensor(0.0))
            }
        
        switch_weight = 1.0 / (2 * torch.exp(apply_loss_scaling.log_vars['switch']))
        physics_weight = lambda_phy / (2 * torch.exp(apply_loss_scaling.log_vars['physics']))
        
        uncertainty_loss = (apply_loss_scaling.log_vars['switch'] + 
                          apply_loss_scaling.log_vars['physics'])
        
        return physics_weight * physics_loss + uncertainty_loss
    
    else:
        raise ValueError(f"Unknown loss scaling strategy: {strategy}")

def get_loss_statistics(switch_loss, physics_loss):
    """Helper function to get loss statistics for monitoring."""
    return {
        "switch_loss_magnitude": switch_loss.item(),
        "physics_loss_magnitude": physics_loss.item(),
        "loss_ratio": physics_loss.item() / (switch_loss.item() + 1e-8),
        "switch_loss_log10": torch.log10(switch_loss + 1e-8).item(),
        "physics_loss_log10": torch.log10(physics_loss + 1e-8).item(),
    }

def _fix_prediction_shape(predicted_scores, target_switches):
    """Fix shape mismatches between predictions and targets."""
    # Add contiguous() calls to fix stride issues
    if predicted_scores.dim() == 2 and predicted_scores.size(0) == 1:
        predicted_scores = predicted_scores.squeeze(0).contiguous()
        logger.debug(f"Squeezed batch dimension: new shape {predicted_scores.shape}")
    
    if predicted_scores.ndim > target_switches.ndim and predicted_scores.shape[-1] == 1:
        predicted_scores = predicted_scores.squeeze(-1).contiguous()
        logger.debug(f"Squeezed trailing dimension: new shape {predicted_scores.shape}")
    
    # Ensure both tensors are contiguous
    predicted_scores = predicted_scores.contiguous()
    target_switches = target_switches.contiguous()
    
    return predicted_scores

def _log_batch_debug_info(data):
    """Log debug information for problematic batches."""
    logger.debug("--- Debugging Shape Mismatch Batch ---")
    logger.debug(f"Batch keys: {list(data.keys())}")
    logger.debug(f"Batch x shape: {data.x.shape}")
    logger.debug(f"Batch edge_index shape: {data.edge_index.shape}")
    logger.debug(f"Batch edge_attr shape: {data.edge_attr.shape}")
    if hasattr(data, 'batch'):
        logger.debug(f"Batch 'batch' tensor shape: {data.batch.shape}")
    if hasattr(data, 'ptr'):
        logger.debug(f"Batch 'ptr' tensor shape: {data.ptr.shape}")
    logger.debug("--------------------------------------")

def train(model, train_loader, optimizer, criterion, device, 
          lambda_phy_loss=0.1, lambda_mask=0.01,
          lambda_connectivity=0.05, lambda_radiality=0.05,
          normalization_type="adaptive", loss_scaling_strategy="adaptive_ratio"):
    """Training loop using common batch processing."""
    model.train()
    
    running_loss = 0.0
    running_metrics = {}
    valid_batch_count = 0
    
    for data in train_loader:
        optimizer.zero_grad()
        
        loss, metrics, batch_stats = process_batch(
            model, data, criterion, device, is_training=True,
            lambda_phy_loss=lambda_phy_loss, 
            lambda_mask=lambda_mask,
            lambda_connectivity=lambda_connectivity,
            lambda_radiality=lambda_radiality,
            normalization_type=normalization_type,
            loss_scaling_strategy=loss_scaling_strategy
        )
        
        if batch_stats["valid_batch"] and loss.requires_grad:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item()
            valid_batch_count += 1
            
            # Accumulate metrics
            for key, value in metrics.items():
                running_metrics[key] = running_metrics.get(key, 0) + value
        else:
            logger.debug("Skipping invalid batch or zero loss")
    
    # Compute averages
    avg_loss = running_loss / valid_batch_count if valid_batch_count > 0 else 0.0
    log_dict = {"train_total_loss": avg_loss}
    
    for key, value in running_metrics.items():
        log_dict[f"train_{key}"] = value / valid_batch_count if valid_batch_count > 0 else 0.0
    
    return avg_loss, log_dict

def test(model, test_loader, criterion, device, 
         lambda_phy_loss=0.1, lambda_mask=0.01,
         lambda_connectivity=0.05, lambda_radiality=0.05,
         normalization_type="adaptive", loss_scaling_strategy="adaptive_ratio"):
    """Testing loop using common batch processing."""
    model.eval()
    
    running_loss = 0.0
    running_metrics = {}
    valid_batch_count = 0
    
    total_switched = 0
    total_edges = 0
    
    with torch.no_grad():
        for data in test_loader:
            loss, metrics, batch_stats = process_batch(
                model, data, criterion, device, is_training=False, 
                lambda_phy_loss=lambda_phy_loss, 
                lambda_mask=lambda_mask,
                lambda_connectivity=lambda_connectivity,
                lambda_radiality=lambda_radiality,
                normalization_type=normalization_type,
                loss_scaling_strategy=loss_scaling_strategy
            )
            
            if batch_stats["valid_batch"]:
                running_loss += loss.item()
                valid_batch_count += 1
                
                # Accumulate metrics
                for key, value in metrics.items():
                    running_metrics[key] = running_metrics.get(key, 0) + value
    
    # Compute averages
    avg_loss = running_loss / valid_batch_count if valid_batch_count > 0 else 0.0
    log_dict = {"test_loss": avg_loss}
    
    for key, value in running_metrics.items():
        log_dict[f"test_{key}"] = value / valid_batch_count if valid_batch_count > 0 else 0.0
    
    # Add switch change statistics
    if total_edges > 0:
        log_dict["test_switch_change_rate"] = total_switched / total_edges
    
    return avg_loss, log_dict

def compute_switch_metrics(scores: torch.Tensor, targets: torch.Tensor, 
                          threshold: float = 0.5, eps: float = 1e-8) -> dict:
    """Compute classification metrics for switch predictions."""
    # Fix shape mismatches
    if scores.ndim == targets.ndim + 1 and scores.size(-1) == 1:
        scores = scores.squeeze(-1)

    targets = targets.float()
    preds = (scores > threshold).float()

    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()
    tn = ((1 - preds) * (1 - targets)).sum()

    accuracy = (tp + tn) / (tp + fp + fn + tn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    jaccard = tp / (tp + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)

    return {
        "accuracy": accuracy.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item(),
        "jaccard": jaccard.item(),
        "dice": dice.item(),
        "tp": tp.item(), "fp": fp.item(),
        "fn": fn.item(), "tn": tn.item(),
    }


def physics_loss(output, data, device, normalization_type="adaptive"):
    """
    Compute physics-informed loss based on power flow constraints.
    
    Args:
        normalization_type: 
            - "none": No normalization
            - "simple": Divide by network size
            - "injection_scale": Scale by magnitude of injections
            - "adaptive": Use running statistics for normalization
            - "per_node": Normalize each constraint separately
    """
    flows_hat = output.get("flows")  
    volt_hat = output.get("node_v")  
    
    if flows_hat is None:
        return torch.tensor(0.0, device=device)
    
    P_hat, Q_hat = flows_hat.T  # Active and reactive power flows
    ei = data.edge_index
    inj_P = data.x[:, 0]  # Active power injections
    inj_Q = data.x[:, 1]  # Reactive power injections
    num_nodes = data.x.size(0)
    num_edges = ei.size(1)

    # Kirchhoff's current law (KCL) violations
    kcl_P = torch.zeros_like(inj_P).index_add_(0, ei[0], P_hat) \
                                   .index_add_(0, ei[1], -P_hat) - inj_P
    kcl_Q = torch.zeros_like(inj_Q).index_add_(0, ei[0], Q_hat) \
                                   .index_add_(0, ei[1], -Q_hat) - inj_Q
    
    # Apply normalization
    if normalization_type == "none":
        kcl_loss_P = kcl_P.pow(2).mean()
        kcl_loss_Q = kcl_Q.pow(2).mean()
        
    elif normalization_type == "simple":
        # Normalize by network size
        kcl_loss_P = kcl_P.pow(2).sum() / num_nodes
        kcl_loss_Q = kcl_Q.pow(2).sum() / num_nodes
        
    elif normalization_type == "injection_scale":
        # Scale by magnitude of injections
        P_scale = torch.clamp(inj_P.abs().mean(), min=1e-6)
        Q_scale = torch.clamp(inj_Q.abs().mean(), min=1e-6)
        kcl_loss_P = (kcl_P.pow(2) / P_scale.pow(2)).mean()
        kcl_loss_Q = (kcl_Q.pow(2) / Q_scale.pow(2)).mean()
        
    elif normalization_type == "adaptive":
        if not hasattr(physics_loss, 'P_running_var'):
            physics_loss.P_running_var = None
            physics_loss.Q_running_var = None
            physics_loss.momentum = 0.1
        
        current_P_var = kcl_P.var() + 1e-8
        current_Q_var = kcl_Q.var() + 1e-8
        
        if physics_loss.P_running_var is None:
            # Store as detached values, not tensors with gradients
            physics_loss.P_running_var = current_P_var.detach()
            physics_loss.Q_running_var = current_Q_var.detach()
        else:
            # Update with detached values
            physics_loss.P_running_var = (1 - physics_loss.momentum) * physics_loss.P_running_var + \
                                    physics_loss.momentum * current_P_var.detach()
            physics_loss.Q_running_var = (1 - physics_loss.momentum) * physics_loss.Q_running_var + \
                                    physics_loss.momentum * current_Q_var.detach()
        
        kcl_loss_P = (kcl_P.pow(2) / physics_loss.P_running_var).mean()
        kcl_loss_Q = (kcl_Q.pow(2) / physics_loss.Q_running_var).mean()
        
    elif normalization_type == "per_node":
        # Normalize each node's constraint by its injection magnitude
        P_node_scale = torch.clamp(inj_P.abs(), min=1e-6)
        Q_node_scale = torch.clamp(inj_Q.abs(), min=1e-6)
        kcl_loss_P = (kcl_P.pow(2) / P_node_scale.pow(2)).mean()
        kcl_loss_Q = (kcl_Q.pow(2) / Q_node_scale.pow(2)).mean()
        
    else:
        raise ValueError(f"Unknown normalization_type: {normalization_type}")
    
    phy_loss = kcl_loss_P + kcl_loss_Q

    # Optional: supervised flow loss if ground truth available
    if hasattr(data, "edge_flow_gt"):
        flow_gt = data.edge_flow_gt.to(device)
        if normalization_type == "injection_scale":
            flow_scale = torch.clamp(flow_gt.abs().mean(), min=1e-6)
            supervised_loss = ((flows_hat - flow_gt).pow(2) / flow_scale.pow(2)).mean()
        else:
            supervised_loss = (flows_hat - flow_gt).pow(2).mean()
        phy_loss += 0.1 * supervised_loss

    # Voltage drop constraints with normalization
    if volt_hat is not None and data.edge_attr.size(1) >= 2:
        R = data.edge_attr[:, 0]  # Resistance
        X = data.edge_attr[:, 1]  # Reactance
        vdrop = volt_hat[ei[0]] - volt_hat[ei[1]] - (R * P_hat + X * Q_hat)
        
        if normalization_type == "injection_scale":
            # Normalize by typical voltage magnitude
            volt_scale = torch.clamp(volt_hat.abs().mean(), min=1e-6)
            vdrop_loss = (vdrop.pow(2) / volt_scale.pow(2)).mean()
        elif normalization_type == "adaptive":
            # ensure voltage‐running var never tracks gradients
            if not hasattr(physics_loss, 'V_running_var') or physics_loss.V_running_var is None:
                physics_loss.V_running_var = (vdrop.var() + 1e-8).detach()
            else:
                new_var = (vdrop.var() + 1e-8).detach()
                physics_loss.V_running_var = (1 - physics_loss.momentum) * physics_loss.V_running_var \
                                            + physics_loss.momentum * new_var
                
            vdrop_loss = (vdrop.pow(2) / physics_loss.V_running_var).mean()
        else:
            vdrop_loss = vdrop.pow(2).mean()
            
        phy_loss += vdrop_loss
    
    logger.debug(f"Base physics loss ({normalization_type}): {phy_loss.item()}")
    return phy_loss