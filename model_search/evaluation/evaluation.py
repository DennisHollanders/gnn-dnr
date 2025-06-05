import nbformat
import nbconvert
import torch
import random
import matplotlib.pyplot as plt
import pandapower as pp
import pandapower.plotting as plot
import os
import logging 
import yaml

logger = logging.getLogger(__name__)

def run_evaluation(model, train_loader, val_loader, test_loader, device, args):
    """
    Run evaluation on the model and save the results to a file.
    """
    print("Running evaluation...")
    # Placeholder for evaluation logic
    # This function should implement the evaluation logic and save the results to a file
    evaluation_template_path = r"C:\Users\denni\Documents\thesis_dnr_gnn_dev\model_search\evaluation\evaluation_template .ipynb"
    output_dir = f"model_search/evaluations/{args.job_name}"
    os.makedirs(output_dir, exist_ok=True)
    executed_notebook_path = os.path.join(output_dir, f"{args.job_name}_executed.ipynb")
    html_output_path = os.path.join(output_dir, f"{args.job_name}_report.html")
    #create_evaluation()

    with open(evaluation_template_path) as f:
        nb = nbformat.read(f, as_version=4)

    notebook_globals = {
        'model': model,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'val_real_loader': val_real_loader,
        'test_loader': test_loader,
        'device': device,
        'args': args, # Pass the args object
        'training_metrics': calculate_metrics(model, train_loader, device),
        #'pprint': pprint, # Make pprint available in the notebook
        'random': random, # Make random available
        'torch': torch, # Make torch available
        'plt': plt, # Make matplotlib.pyplot available
        'os': os, # Make os available
        # Assuming pandapower and pandapower.plotting are installed in the notebook env
        # 'pp': pp,
        # 'plot': plot,
        # If calculate_metrics, plot_grid, plot_voltage_profile are in a util file, import them here:
        # 'calculate_metrics': calculate_metrics,
        # 'plot_grid': plot_grid,
        # 'plot_voltage_profile': plot_voltage_profile,
    }
    executor = nbconvert.preprocessors.ExecutePreprocessor(
        timeout=600, # Adjust timeout as needed
        kernel_name='python3',
        # stream_output=True # Uncomment to see notebook output in the console
    )

    try:
        logger.debug(f"Executing notebook: {evaluation_template_path}")
        # Execute the notebook, making notebook_globals available
        executor.preprocess(nb, {'metadata': {'path': './'}}, globals=notebook_globals)
        logger.debug("Notebook executed successfully.")

        # Save the executed notebook
        with open(executed_notebook_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        logger.debug(f"Executed notebook saved to: {executed_notebook_path}")

        # Convert the executed notebook to HTML
        html_exporter = nbconvert.HTMLExporter()
        # You can customize the export process if needed
        # html_exporter.template_name = 'classic'

        (body, resources) = html_exporter.from_notebook_node(nb)

        # Save the HTML report
        with open(html_output_path, 'w', encoding='utf-8') as f:
            f.write(body)
        logger.debug(f"HTML report saved to: {html_output_path}")

    except Exception as e:
        logger.debug(f"Error during notebook execution or conversion: {e}")
        logger.debug("Saving the notebook with injected code for debugging.")
        # Save the notebook even if execution fails for debugging
        with open(executed_notebook_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        logger.debug(f"Notebook saved to: {executed_notebook_path}")
    logger.debug("Evaluation completed.")



def calculate_metrics(model, data_loader, device):
    """
    Calculate evaluation metrics for the model.

    Args:
        model: The trained model.
        data_loader: The data loader for the evaluation dataset.
        device: The device to run the evaluation on.

    Returns:
        A dictionary of calculated metrics.
    """
    model.eval()
    metrics = {}
    total_loss = 0
    # Assuming your model outputs something that can be compared to a target
    # You will need to adapt this based on your model's output and loss function
    criterion = torch.nn.MSELoss() # Replace with your actual criterion

    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            output = model(data)
            # Assuming data.y contains the target values
            loss = criterion(output, data.y)
            total_loss += loss.item() * data.num_graphs # Multiply by num_graphs for correct average over batches

    metrics['average_loss'] = total_loss / len(data_loader.dataset)

    # Add other relevant metrics here (e.g., accuracy, precision, recall, F1-score)
    # depending on your GNN task (e.g., node classification, graph classification, link prediction)
    # Example placeholder for another metric:
    # metrics['accuracy'] = calculate_accuracy(model, data_loader, device)

    return metrics

def plot_grid(net,ax,title=""):
    pp.plotting.simple_plot(net, ax=ax, show=False)
    ax.set_title(title)
def plot_voltage_profile(voltage_data, ax, title=""):
    """Plots the voltage profiles."""
    # voltage_data is expected to be a pandas DataFrame or similar structure
    # with columns for each phase (e.g., 'vm_pu_a', 'vm_pu_b', 'vm_pu_c')
    # and bus indices as index.
    for col in voltage_data.columns:
        ax.plot(voltage_data.index, voltage_data[col], label=col)
    ax.set_title(title)
    ax.set_xlabel("Bus Index")
    ax.set_ylabel("Voltage (pu)")
    ax.legend()

def load_config_from_model_path(model_path):
    """
    Load configuration from the config_files directory based on the model path.
    Assumes the config file is in the same model directory under config_files/
    """
    model_dir = os.path.dirname(model_path)
    config_dir = os.path.join(model_dir, "config_files")
    
    # Extract job name from model filename (e.g., "clear-monkey-40" from "clear-monkey-40-Epoch99-Last.pt")
    model_filename = os.path.basename(model_path)
    job_name = model_filename.split('-Epoch')[0].split('-Best')[0]
    
    # Look for config files matching the job name
    config_files = []
    if os.path.exists(config_dir):
        for file in os.listdir(config_dir):
            if file.endswith('.yaml') and job_name in file:
                config_files.append(os.path.join(config_dir, file))
    
    if not config_files:
        raise FileNotFoundError(f"No config file found for job name '{job_name}' in {config_dir}")
    
    # Use the most recent config file if multiple exist
    config_file = sorted(config_files)[-1]
    
    print(f"Loading configuration from: {config_file}")
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

if __name__ == "__main__":
    import sys
    from pathlib import Path
    import argparse
    import torch
    
    # Add the parent directories to the path
    current_dir = Path(__file__).resolve().parent
    model_search_dir = current_dir.parent
    root_dir = model_search_dir.parent
    
    if str(model_search_dir) not in sys.path:
        sys.path.insert(0, str(model_search_dir))
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))
    
    from load_data import create_data_loaders
    import importlib

    # Define parse_args locally for standalone execution
    def parse_args():
        parser = argparse.ArgumentParser(description="Standalone evaluation script")
        parser.add_argument("--model_path",default = r"model_search\models\MLP\clear-monkey-40-Epoch99-Last.pt",  type=str, help="Path to the saved model")
        parser.add_argument("--config_path", default = r"model_search\models\MLP\config_files\MLP------clear-monkey-40.yaml",type=str, help="Path to the config YAML file (optional, will try to auto-detect)")
        parser.add_argument("--override_job_name", type=str, help="Override job name for evaluation output")
        return parser.parse_args()

    args = parse_args()

    # Load configuration from YAML file
    if args.config_path:
        print(f"Loading configuration from: {args.config_path}")
        with open(args.config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Try to auto-detect config file from model path
        config = load_config_from_model_path(args.model_path)
    
    # Convert config dict to argparse.Namespace for compatibility
    eval_args = argparse.Namespace(**config)
    
    # Override job name if specified
    if args.override_job_name:
        eval_args.job_name = args.override_job_name
    
    print(f"\nLoaded configuration:")
    print(f"  Model module: {eval_args.model_module}")
    print(f"  Hidden dims: {eval_args.hidden_dims}")
    print(f"  Latent dim: {eval_args.latent_dim}")
    print(f"  Activation: {eval_args.activation}")
    print(f"  Dropout rate: {eval_args.dropout_rate}")
    print(f"  Job name: {eval_args.job_name}")
    print()

    # Import and instantiate the model
    model_module = importlib.import_module(f"models.{eval_args.model_module}.{eval_args.model_module}")
    model_class = getattr(model_module, eval_args.model_module)
    
    # Create data loaders
    dataloaders = create_data_loaders(
        dataset_names=eval_args.dataset_names,
        folder_names=eval_args.folder_names,
        dataset_type=eval_args.dataset_type,
        batch_size=eval_args.batch_size,
        max_nodes=eval_args.max_nodes,
        max_edges=eval_args.max_edges,
        train_ratio=eval_args.train_ratio,
        seed=eval_args.seed,
        num_workers=eval_args.num_workers,
        batching_type=eval_args.batching_type,
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader = dataloaders.get("train")
    validation_loader = dataloaders.get("validation")
    test_loader = dataloaders.get("test")
    val_real_loader = None  # Set this if you have a real validation set
    
    # Get dimensions from the first batch
    sample_data = train_loader.dataset[0]
    node_input_dim = sample_data.x.shape[1]
    edge_input_dim = sample_data.edge_attr.shape[1]
    
    print(f"Data dimensions:")
    print(f"  Node input dim: {node_input_dim}")
    print(f"  Edge input dim: {edge_input_dim}")
    
    # Create model instance with the exact same configuration as training
    model = model_class(
        node_input_dim=node_input_dim,
        edge_input_dim=edge_input_dim,
        hidden_dims=eval_args.hidden_dims,
        latent_dim=eval_args.latent_dim,
        activation=eval_args.activation,
        dropout_rate=eval_args.dropout_rate
    ).to(device)
    
    # Load model weights
    print(f"\nLoading model weights from: {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print("Model loaded successfully!")
    
    # Run evaluation
    run_evaluation(model, train_loader, validation_loader, val_real_loader, test_loader, device, eval_args)