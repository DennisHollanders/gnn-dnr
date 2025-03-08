import torch
import torch.nn as nn
import wandb
import psutil

def train(model, train_loader, optimizer, criterion, device,):
    model.train()
    running_loss = 0.0
    batch_count = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.x.float())
        metrics = calculate_metrics(data, output)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        batch_count += 1
    avg_loss = running_loss / batch_count
    log_dict = {"train_loss": avg_loss}
    for name, total in metrics.items():
        log_dict[name] = total / batch_count
    wandb.log(log_dict)
    return avg_loss

def test(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    batch_count = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            loss = criterion(output, data.x.float())
            metrics = calculate_metrics(data, output)
            total_loss += loss.item()
            batch_count += 1
    avg_loss = total_loss / batch_count
    log_dict = {"test_loss": avg_loss}
    for name, total in metrics.items():
        log_dict[name] = total / batch_count
        
    log_dict.update(log_hardware_usage())
    wandb.log(log_dict)
    
    return avg_loss

def calculate_metrics(data, output):    

    def calculate_mse(data, output):
        return nn.MSELoss()(output, data.x.float()).item()
    
    def calculate_mae(data, output):
        return nn.L1Loss()(output, data.x.float()).item()
    
    def radiality_constraint(data, output):
        return torch.mean(torch.abs(data.x[:, 1] - output[:, 1])).item()
    
    def voltage_constraint(data, output):
        return torch.mean(torch.abs(data.x[:, 2] - output[:, 2])).item()
    
    return{ "mse": calculate_mse(data, output),
            "mae": calculate_mae(data, output),
            "radiality_constraint": radiality_constraint(data, output),
            "voltage_constraint": voltage_constraint(data, output)}

def log_hardware_usage():
    """Capture hardware usage metrics."""
    # CPU usage
    cpu_percent = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory()
    mem_used = mem.used / (1024 ** 2)  # MB
    mem_total = mem.total / (1024 ** 2)  # MB

    metrics = {
        "cpu_percent": cpu_percent,
        "memory_used_MB": mem_used,
        "memory_total_MB": mem_total,
    }
    # GPU metrics if available
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
        gpu_memory_max = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
        metrics.update({
            "gpu_memory_MB": gpu_memory,
            "gpu_memory_max_MB": gpu_memory_max,
        })
    return metrics