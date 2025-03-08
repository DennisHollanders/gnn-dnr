import torch
import torch.nn as nn
import wandb
import psutil

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    running_local_losses = {}
    batch_count = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()

        output = model(data)
        
        # Handle both cases: combined loss or separate losses
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

    with torch.no_grad():
        for data in loader:
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

            batch_count += 1

    avg_loss = running_loss / batch_count
    log_dict = {"test_loss": avg_loss}

    for key, value in running_local_losses.items():
        log_dict[f"test_{key}"] = value / batch_count
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
