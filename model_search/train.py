import torch
import torch.nn as nn
import wandb
import psutil

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    running_local_losses = {}
    batch_count = 0
    torch.autograd.set_detect_anomaly(True)
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()

        output = model(data)

        if isinstance(output, dict):
            loss, local_losses = criterion(output, data)
        else:
            loss = criterion(output, data.x.float())
            local_losses = {}

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        for key, value in local_losses.items():
            running_local_losses[key] = running_local_losses.get(key, 0) + value.item()

        batch_count += 1

    avg_loss = running_loss / batch_count
    log_dict = {"train_total_loss": avg_loss}

    for key, value in running_local_losses.items():
        log_dict[f"train_{key}"] = value / batch_count
    log_dict.update(log_hardware_usage())
    return avg_loss, log_dict

def test(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_local_losses = {}
    batch_count = 0

    total_switched = 0
    total_edges = 0

    with torch.no_grad():
        for data in loader:
            if data is None:
                continue
            data = data.to(device)
            output = model(data)
            if isinstance(output, dict):
                loss, local_losses = criterion(output, data)
            else:
                loss = criterion(output, data.x.float())
                local_losses = {}

            running_loss += loss.item()
            for key, value in local_losses.items():
                running_local_losses[key] = running_local_losses.get(key, 0) + value.item()

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
                    if updated_switches.numel() == original_switch_states.numel():
                        # If they match, compare directly.
                        batch_switched = torch.sum(updated_switches != original_switch_states).item()
                    elif updated_switches.numel() == data.num_graphs:
                        # Model output is per graph. Map each edge to its graph prediction.
                        # We assume that each edge belongs to the graph of its source node.
                        edge_batch = data.batch[data.edge_index[0]]
                        updated_switches_edges = updated_switches[edge_batch]
                        batch_switched = torch.sum(updated_switches_edges != original_switch_states).item()
                    else:
                        print("Warning: Mismatched shapes for switch state comparison.")
                        batch_switched = 0

                    total_switched += batch_switched
                    total_edges += original_switch_states.numel()
            batch_count += 1

    if batch_count == 0:
        print("Warning: No valid batches were processed in the loader.")
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
        log_dict.update(log_hardware_usage())
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
