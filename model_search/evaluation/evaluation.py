import nbformat
import nbconvert
import torch
import random
import matplotlib.pyplot as plt
import pandapower as pp
import pandapower.plotting as plot
import os


def run_evaluation(model, train_loader, val_loader, val_real_loader, test_loader, device, args):
    """
    Run evaluation on the model and save the results to a file.
    """
    print("Running evaluation...")
    # Placeholder for evaluation logic
    # This function should implement the evaluation logic and save the results to a file
    evaluation_template_path = "model_search/evaluation/evaluation_template.ipynb"
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
        print(f"Executing notebook: {evaluation_template_path}")
        # Execute the notebook, making notebook_globals available
        executor.preprocess(nb, {'metadata': {'path': './'}}, globals=notebook_globals)
        print("Notebook executed successfully.")

        # Save the executed notebook
        with open(executed_notebook_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        print(f"Executed notebook saved to: {executed_notebook_path}")

        # Convert the executed notebook to HTML
        html_exporter = nbconvert.HTMLExporter()
        # You can customize the export process if needed
        # html_exporter.template_name = 'classic'

        (body, resources) = html_exporter.from_notebook_node(nb)

        # Save the HTML report
        with open(html_output_path, 'w', encoding='utf-8') as f:
            f.write(body)
        print(f"HTML report saved to: {html_output_path}")

    except Exception as e:
        print(f"Error during notebook execution or conversion: {e}")
        print("Saving the notebook with injected code for debugging.")
        # Save the notebook even if execution fails for debugging
        with open(executed_notebook_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        print(f"Notebook saved to: {executed_notebook_path}")
    print("Evaluation completed.")



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