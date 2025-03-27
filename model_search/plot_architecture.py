import argparse
import os
import importlib
import torch
from torch_geometric.data import Data
from torchviz import make_dot

def parse_args():
    parser = argparse.ArgumentParser(description="Plot Model Architecture")
    parser.add_argument("--model_module", type=str, default="PIGNN",
                        help="Name of the model module (also the model class name)")
    parser.add_argument("--input_dim", type=int, default = 30,  help="Input dimension")
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[64, 32, 16], help="Hidden dimensions")
    parser.add_argument("--latent_dim", type=int, default=8, help="Latent dimension")
    parser.add_argument("--activation", type=str, default="prelu",
                        choices=["relu", "leaky_relu", "elu", "selu", "prelu", "sigmoid", "tanh"],
                        help="Activation function")
    parser.add_argument("--dropout_rate", type=float, default=0.0, help="Dropout rate")
    parser.add_argument("--output", type=str, default="model_architecture", help="Output filename (without extension)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Dynamically import the model module and class.
    model_module = importlib.import_module(f"models.{args.model_module}.{args.model_module}")
    model_class = getattr(model_module, args.model_module)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class(
        input_dim=args.input_dim,
        hidden_dims=args.hidden_dims,
        latent_dim=args.latent_dim,
        activation=args.activation,
        dropout_rate=args.dropout_rate
    ).to(device)
    
    num_nodes = 10
    dummy_x = torch.randn(num_nodes, args.input_dim, device=device)
    dummy_edge_index = torch.tensor([[0, 1, 2, 3],
                                     [1, 2, 3, 4]], dtype=torch.long, device=device)
    dummy_batch = torch.zeros(num_nodes, dtype=torch.long, device=device)
    dummy_data = Data(x=dummy_x, edge_index=dummy_edge_index, batch=dummy_batch)
    dummy_data.conductance_matrix = torch.eye(num_nodes, device=device)
    dummy_data.edge_attr = torch.ones(dummy_edge_index.size(1), 2, device=device)
    
    output = model(dummy_data)
    y = output["switch_scores"].sum()
    
    dot = make_dot(y, params=dict(model.named_parameters()))
    dot.graph_attr.update(dpi='300')

    model_folder = os.path.dirname(model_module.__file__)
    output_filename = os.path.join(model_folder, args.output)
    dot.render(filename=output_filename, format='png', cleanup=True, view=True)
    print(f"Model architecture plot saved as {output_filename}.png")

if __name__ == "__main__":
    main()
