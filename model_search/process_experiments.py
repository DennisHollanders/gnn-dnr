import pandas as pd
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, recall_score, precision_score
import pandapower as pp
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import os 
import sys 
# Import functions from define_ground_truth.py

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data_generation.define_ground_truth import (
    build_nx_graph, 
    find_cycles, 
    find_disconnected_buses, 
    is_radial_and_connected,
    plot_grid_component,
    plot_voltage_profile,
    visualize_network_states
)

def analyze_csv_results(csv_file):
    df = pd.read_csv(csv_file)
    
    # Parse filename to extract configuration
    filename = csv_file.stem  # results-{model}-{rounding}-{warmstart}
    parts = filename.replace("results-", "").split("-")
    
    print(f"  Parsing filename: {filename}")
    print(f"  Filename parts: {parts}")
    
    # Handle more complex model names (e.g., "None-Best", "AdvancedMLP", etc.)
    if len(parts) >= 3:
        # Last two parts should be rounding and warmstart
        warmstart_mode = parts[-1]
        rounding_method = parts[-2]
        # Everything else is the model name
        model_name = "-".join(parts[:-2])
    else:
        # Fallback parsing
        model_name = "unknown"
        rounding_method = "unknown" 
        warmstart_mode = "unknown"
    
    print(f"  Parsed: model={model_name}, rounding={rounding_method}, warmstart={warmstart_mode}")
    
    # Initialize metrics
    all_f1 = []
    all_recall = []
    all_precision = []
    all_delta = []
    all_solve_times = []
    
    infeasible_count = 0
    radial_wrong_count = 0
    non_radial_count = 0
    
    print(f"  Processing {len(df)} rows...")
    
    for idx, row in df.iterrows():
        try:
            # Parse JSON fields
            ground_truth = json.loads(row['ground_truth'])
            gnn_prediction = json.loads(row['gnn_prediction']) 
            final_optima = json.loads(row['final_optima'])
            
            print(f"    Row {idx}: GT={len(ground_truth)}, GNN={len(gnn_prediction)}, Final={len(final_optima)}")
            
            # Handle length mismatches
            if len(ground_truth) != len(gnn_prediction):
                print(f"    Warning: Length mismatch - GT:{len(ground_truth)} vs GNN:{len(gnn_prediction)}")
                
                # Try to align arrays - assume GNN prediction corresponds to a subset of switches
                # This might happen if not all lines have switches, or preprocessing filtered some switches
                if len(gnn_prediction) < len(ground_truth):
                    print(f"    Truncating ground truth to match GNN prediction length")
                    ground_truth = ground_truth[:len(gnn_prediction)]
                elif len(ground_truth) < len(gnn_prediction):
                    print(f"    Truncating GNN prediction to match ground truth length")
                    gnn_prediction = gnn_prediction[:len(ground_truth)]
            
            if len(ground_truth) != len(final_optima):
                print(f"    Warning: Length mismatch - GT:{len(ground_truth)} vs Final:{len(final_optima)}")
                
                if len(final_optima) < len(ground_truth):
                    print(f"    Truncating ground truth to match final optima length")
                    ground_truth = ground_truth[:len(final_optima)]
                elif len(ground_truth) < len(final_optima):
                    print(f"    Truncating final optima to match ground truth length")
                    final_optima = final_optima[:len(ground_truth)]
            
            # Final validation
            if len(ground_truth) != len(gnn_prediction) or len(ground_truth) != len(final_optima):
                print(f"    Skipping row {idx} due to unresolvable length mismatch")
                infeasible_count += 1
                continue
                
            # Calculate prediction metrics (GNN vs ground truth)
            f1 = f1_score(ground_truth, gnn_prediction, average='binary', zero_division=0)
            recall = recall_score(ground_truth, gnn_prediction, average='binary', zero_division=0)
            precision = precision_score(ground_truth, gnn_prediction, average='binary', zero_division=0)
            
            all_f1.append(f1)
            all_recall.append(recall)
            all_precision.append(precision)
            
            # Calculate solution difference (δ solution = switches different from ground truth)
            delta = sum(1 for gt, opt in zip(ground_truth, final_optima) if gt != opt)
            all_delta.append(delta)
            
            # Add solve time if available (would need to be added to CSV by predict_then_optimize.py)
            # For now, use placeholder
            all_solve_times.append(0.0)
            
            print(f"    Row {idx}: F1={f1:.3f}, Delta={delta}")
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"    Error processing row {idx}: {e}")
            infeasible_count += 1
            continue
    
    # Calculate radiality metrics from network files if available
    predictions_path = csv_file.parent
    folder_pattern = filename.replace('results-', '')
    
    # Look for corresponding prediction folders
    prediction_folders = list(predictions_path.glob(f"prediction-{folder_pattern}*"))
    
    if prediction_folders:
        prediction_folder = prediction_folders[0]
        prediction_networks_folder = prediction_folder / "pandapower_networks"
        
        if prediction_networks_folder.exists():
            for json_file in prediction_networks_folder.glob("*.json"):
                try:
                    # Load final prediction network
                    final_net = pp.from_json(str(json_file))
                    
                    # Check if final network is radial and connected
                    final_radial, final_connected = is_radial_and_connected(final_net, include_switches=True)
                    
                    if not final_radial or not final_connected:
                        non_radial_count += 1
                        
                        # Additional check: if this should have been radial (simplified heuristic)
                        # You might want to implement more sophisticated logic here
                        if not final_radial:
                            radial_wrong_count += 1
                            
                except Exception as e:
                    print(f"Error analyzing network {json_file}: {e}")
                    infeasible_count += 1
                    continue
    
    # Calculate accuracy with length mismatch handling
    accuracy_scores = []
    for _, row in df.iterrows():
        if _is_valid_json_row(row):
            try:
                gt = json.loads(row['ground_truth'])
                fo = json.loads(row['final_optima'])
                # Handle length mismatch
                min_len = min(len(gt), len(fo))
                if min_len > 0:
                    acc = f1_score(gt[:min_len], fo[:min_len], average='binary', zero_division=0)
                    accuracy_scores.append(acc)
            except:
                continue
    
    # Compile results
    metrics = {
        'model': model_name,
        'warmstart_mode': warmstart_mode, 
        'rounding_method': rounding_method,
        'filename': csv_file.name,
        
        # Performance metrics
        'avg_f1': np.mean(all_f1) if all_f1 else 0.0,
        'avg_recall': np.mean(all_recall) if all_recall else 0.0,
        'avg_precision': np.mean(all_precision) if all_precision else 0.0,
        'avg_delta': np.mean(all_delta) if all_delta else 0.0,
        'avg_solve_time': np.mean(all_solve_times) if all_solve_times else 0.0,
        
        # Error counts
        'infeasible': infeasible_count,
        'radial_wrong': radial_wrong_count,
        'non_radial': non_radial_count,
        
        # Additional metrics
        'total_graphs': len(df),
        'successful_optimizations': len(all_f1),
        'accuracy': np.mean(accuracy_scores) if accuracy_scores else 0.0,
        'avg_loss_improvement': 0.0  # Placeholder - would need loss data
    }
    
    print(f"  Results: F1={metrics['avg_f1']:.3f}, Accuracy={metrics['accuracy']:.3f}, Delta={metrics['avg_delta']:.1f}")
    print(f"  Processed {metrics['successful_optimizations']}/{metrics['total_graphs']} graphs successfully")
    
    return metrics

def _is_valid_json_row(row):
    """Helper function to check if a row has valid JSON data"""
    try:
        ground_truth = json.loads(row['ground_truth'])
        final_optima = json.loads(row['final_optima'])
        # Allow for length mismatches, we'll handle them in processing
        return True
    except:
        return False

def visualize_prediction_pipeline(graph_id, mst_net, mst_opt_net, gnn_predictions, 
                                final_prediction_net, rounding_method, output_dir=None):
    """
    Create visualization of the prediction pipeline for a single graph
    
    Args:
        graph_id: Graph identifier
        mst_net: Original MST network
        mst_opt_net: Optimized MST network (ground truth)
        gnn_predictions: List of GNN prediction probabilities
        final_prediction_net: Final predicted network after optimization
        rounding_method: Rounding method used
        output_dir: Directory to save plots
    """
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create rounded predictions for visualization
    if rounding_method == "round":
        rounded_predictions = [1 if pred > 0.5 else 0 for pred in gnn_predictions]
    elif rounding_method == "PhyR":
        # Placeholder for PhyR - implement your physics-informed rounding
        rounded_predictions = [1 if pred > 0.5 else 0 for pred in gnn_predictions]
    else:
        rounded_predictions = [1 if pred > 0.5 else 0 for pred in gnn_predictions]
    
    # Create network with GNN predictions applied
    gnn_prediction_net = mst_net.deepcopy()
    for i, pred in enumerate(rounded_predictions):
        if i < len(gnn_prediction_net.switch):
            gnn_prediction_net.switch.at[i, 'closed'] = bool(pred)
    
    # Prepare snapshots for visualization
    snapshots = {
        "original": mst_net,
        "gnn_prediction": gnn_prediction_net,
        "optimized_prediction": final_prediction_net,
        "ground_truth": mst_opt_net
    }
    
    # Use the visualization function from define_ground_truth.py
    visualize_network_states(snapshots, graph_id, output_dir=output_dir, debug=False)

def create_latex_table(results_df):
    """
    Create LaTeX table from results DataFrame
    
    Args:
        results_df: DataFrame with experimental results
    
    Returns:
        String containing LaTeX table
    """
    # Group results by model
    models = ['GAT', 'GIN', 'GCN']  # Adjust based on your actual model names
    
    # Start LaTeX table
    latex_lines = [
        "\\begin{table*}[p]",
        "  \\centering", 
        "  \\small",
        "  \\renewcommand{\\arraystretch}{1.2}",
        "  \\resizebox{\\textwidth}{!}{%",
        "    \\arrayrulecolor{black}%",
        "    \\begin{tabular}{lllcccccccc}",
        "      \\toprule",
        "      Model & Method & Variant & Time [s] & F1 & Recall & Precision",
        "            & $\\delta$ solution & Radial wrong & Non-radial & Infeasible \\\\",
        "      \\midrule"
    ]
    
    for model_idx, model in enumerate(models):
        if model_idx > 0:
            latex_lines.append("      \\arrayrulecolor{black}\\midrule")
            
        model_results = results_df[results_df['model'].str.contains(model, case=False, na=False)]
        
        if model_results.empty:
            # Add empty rows for missing model
            latex_lines.extend(_get_empty_model_rows(model))
            continue
        
        latex_lines.append(f"      \\multirow{{11}}{{*}}{{{model}}}")
        
        # DirectPrediction rows
        direct_round = model_results[
            (model_results['warmstart_mode'] == 'none') & 
            (model_results['rounding_method'] == 'round')
        ]
        direct_phyr = model_results[
            (model_results['warmstart_mode'] == 'none') & 
            (model_results['rounding_method'] == 'PhyR')
        ]
        
        latex_lines.append("        & \\multirow{2}{*}{DirectPrediction}")
        latex_lines.append(_format_result_row("Rounding", direct_round, first_in_group=False))
        latex_lines.append(_format_result_row("PhyR", direct_phyr, first_in_group=False))
        
        # SoftWarmStart rows  
        latex_lines.append("      \\arrayrulecolor{gray!60}\\cdashline{2-11}")
        
        soft_float = model_results[
            (model_results['warmstart_mode'] == 'float') & 
            (model_results['rounding_method'] == 'round')
        ]
        soft_round = model_results[
            (model_results['warmstart_mode'] == 'soft') & 
            (model_results['rounding_method'] == 'round')
        ]
        soft_phyr = model_results[
            (model_results['warmstart_mode'] == 'soft') & 
            (model_results['rounding_method'] == 'PhyR')
        ]
        
        latex_lines.append("        & \\multirow{3}{*}{SoftWarmStart}")
        latex_lines.append(_format_result_row("Floats", soft_float, first_in_group=False))
        latex_lines.append(_format_result_row("Binary Rounding", soft_round, first_in_group=False))
        latex_lines.append(_format_result_row("Binary PhyR", soft_phyr, first_in_group=False))
        
        # HardWarmStart rows
        latex_lines.append("      \\cdashline{2-11}")
        
        hard_results = model_results[model_results['warmstart_mode'] == 'hard']
        confidence_levels = ['0.9', '0.7', '0.5', '0.3', '0.1']
        
        latex_lines.append("        & \\multirow{5}{*}{HardWarmStart}")
        for i, conf in enumerate(confidence_levels):
            # Note: You might need to parse confidence from filename or add it to CSV
            conf_results = hard_results  # Simplified - you may need better logic here
            latex_lines.append(_format_result_row(conf, conf_results if i == 0 else pd.DataFrame(), 
                                                first_in_group=False))
        
        # DFL row (placeholder)
        latex_lines.append("      \\cdashline{2-11}")
        latex_lines.append("        & DFL                         & --                &  &  &  &  &  &  &  &  \\\\")
    
    # Close table
    latex_lines.extend([
        "      \\bottomrule",
        "    \\end{tabular}%",
        "  }",
        "  \\caption{Model Metrics and Error Distribution Comparison}",
        "  \\label{tab:full-result}",
        "\\end{table*}"
    ])
    
    return "\n".join(latex_lines)

def _format_result_row(variant, results_df, first_in_group=True):
    """Format a single result row for LaTeX table"""
    if results_df.empty:
        return f"        &                             & {variant:<15} &  &  &  &  &  &  &  &  \\\\"
    
    row = results_df.iloc[0]  # Take first matching result
    
    return (f"        &                             & {variant:<15} "
            f"& {row['avg_solve_time']:.1f} "
            f"& {row['avg_f1']:.3f} "
            f"& {row['avg_recall']:.3f} "
            f"& {row['avg_precision']:.3f} "
            f"& {row['avg_delta']:.1f} "
            f"& {row['radial_wrong']} "
            f"& {row['non_radial']} "
            f"& {row['infeasible']} \\\\")

def _get_empty_model_rows(model):
    """Get empty rows for a model with no results"""
    return [
        f"      \\multirow{{11}}{{*}}{{{model}}}",
        "        & \\multirow{2}{*}{DirectPrediction}",
        "        &                             & Rounding          &  &  &  &  &  &  &  &  \\\\",
        "        &                             & PhyR               &  &  &  &  &  &  &  &  \\\\",
        "      \\arrayrulecolor{gray!60}\\cdashline{2-11}",
        "        & \\multirow{3}{*}{SoftWarmStart}",
        "        &                             & Floats            &  &  &  &  &  &  &  &  \\\\",
        "        &                             & Binary Rounding   &  &  &  &  &  &  &  &  \\\\", 
        "        &                             & Binary PhyR       &  &  &  &  &  &  &  &  \\\\",
        "      \\cdashline{2-11}",
        "        & \\multirow{5}{*}{HardWarmStart}",
        "        &                             & 0.9               &  &  &  &  &  &  &  &  \\\\",
        "        &                             & 0.7               &  &  &  &  &  &  &  &  \\\\",
        "        &                             & 0.5               &  &  &  &  &  &  &  &  \\\\",
        "        &                             & 0.3               &  &  &  &  &  &  &  &  \\\\",
        "        &                             & 0.1               &  &  &  &  &  &  &  &  \\\\",
        "      \\cdashline{2-11}",
        "        & DFL                         & --                &  &  &  &  &  &  &  &  \\\\"
    ]

def analyze_network_solutions(predictions_folder, model_name_mapping):
    """
    Analyze all CSV files in predictions folder and extract radiality information
    
    Args:
        predictions_folder: Path to predictions folder
        model_name_mapping: Dict mapping model names to display names (GAT, GIN, GCN)
    
    Returns:
        Enhanced metrics with radiality analysis
    """
    predictions_path = Path(predictions_folder)
    all_results = {}
    
    # Find all CSV files
    csv_files = list(predictions_path.glob("results-*.csv"))
    print(f"Found {len(csv_files)} CSV files to process")
    
    for csv_file in csv_files:
        print(f"Processing {csv_file.name}...")
        
        try:
            # Analyze CSV
            metrics = analyze_csv_results(csv_file)
            
            # Parse filename to extract model, rounding method, and warmstart mode
            filename = csv_file.stem
            parts = filename.replace("results-", "").split("-")
            
            if len(parts) >= 3:
                model_key = parts[0]
                rounding_method = parts[1]
                warmstart_mode = parts[2]
                
                # Map model name
                model_display = model_name_mapping.get(model_key, model_key)
                
                key = (model_display, warmstart_mode, rounding_method)
                all_results[key] = metrics
                
                print(f"  -> {model_display} | {warmstart_mode} | {rounding_method}")
                
        except Exception as e:
            print(f"Error processing {csv_file.name}: {e}")
            continue
    
    return all_results

def create_visualizations_from_csv(predictions_folder, csv_file, model_name_mapping, 
                                 output_dir=None, max_graphs=5):
    """
    Create visualization plots for selected graphs from CSV results
    
    Args:
        predictions_folder: Path to predictions folder
        csv_file: CSV file with results
        model_name_mapping: Mapping of model names
        output_dir: Output directory for plots
        max_graphs: Maximum number of graphs to visualize per configuration
    """
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    predictions_path = Path(predictions_folder)
    df = pd.read_csv(csv_file)
    
    # Parse filename to get configuration
    filename = csv_file.stem
    folder_pattern = filename.replace('results-', '')
    parts = folder_pattern.split('-')
    
    if len(parts) >= 3:
        model_key = parts[0]
        rounding_method = parts[1]
        warmstart_mode = parts[2]
        
        model_display = model_name_mapping.get(model_key, model_key)
        
        # Find corresponding folders
        prediction_folders = list(predictions_path.glob(f"prediction-{folder_pattern}*"))
        warmstart_folders = list(predictions_path.glob(f"warm-start-{folder_pattern}*"))
        
        if prediction_folders and warmstart_folders:
            prediction_folder = prediction_folders[0]
            warmstart_folder = warmstart_folders[0]
            
            # Get source data folder (assuming it's parent of predictions)
            source_folder = predictions_path.parent
            mst_folder = source_folder / "mst" / "pandapower_networks"
            mst_opt_folder = source_folder / "mst_opt" / "pandapower_networks"
            
            if mst_folder.exists() and mst_opt_folder.exists():
                # Select a few graphs to visualize
                selected_graphs = df.head(max_graphs)['graph_id'].tolist()
                
                print(f"Creating visualizations for {len(selected_graphs)} graphs...")
                
                for graph_id in selected_graphs:
                    try:
                        # Load all networks
                        mst_net = pp.from_json(str(mst_folder / f"{graph_id}.json"))
                        mst_opt_net = pp.from_json(str(mst_opt_folder / f"{graph_id}.json"))
                        final_net = pp.from_json(str(prediction_folder / "pandapower_networks" / f"{graph_id}.json"))
                        
                        # Get GNN predictions from CSV
                        row = df[df['graph_id'] == graph_id].iloc[0]
                        gnn_predictions = json.loads(row['gnn_prediction'])
                        
                        # Create visualization
                        viz_output = output_dir / f"{model_display}_{warmstart_mode}_{rounding_method}" if output_dir else None
                        if viz_output:
                            viz_output.mkdir(parents=True, exist_ok=True)
                        
                        visualize_prediction_pipeline(
                            graph_id=f"{graph_id}_{model_display}_{warmstart_mode}_{rounding_method}",
                            mst_net=mst_net,
                            mst_opt_net=mst_opt_net,
                            gnn_predictions=gnn_predictions,
                            final_prediction_net=final_net,
                            rounding_method=rounding_method,
                            output_dir=viz_output
                        )
                        
                        print(f"  -> Visualized {graph_id}")
                        
                    except Exception as e:
                        print(f"Error visualizing graph {graph_id}: {e}")
                        continue

def save_latex_ready_table(predictions_folder, model_name_mapping, output_path=None, 
                          create_plots=True, plot_output_dir=None, max_plots_per_config=3):
    """
    Process all CSV files and create LaTeX-ready table with optional plotting
    
    Args:
        predictions_folder: Path to folder containing prediction results
        model_name_mapping: Dictionary mapping model keys to display names
        output_path: Path to save the LaTeX table (optional)
        create_plots: Whether to create visualization plots
        plot_output_dir: Directory to save plots
        max_plots_per_config: Maximum number of plots per configuration
    """
    predictions_path = Path(predictions_folder)
    
    if create_plots:
        if plot_output_dir is None:
            plot_output_dir = "prediction_pipeline_plots"
        plot_output_dir = Path(plot_output_dir)
        plot_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Analyzing results in {predictions_path}")
    
    # Analyze all results
    all_results = analyze_network_solutions(predictions_folder, model_name_mapping)
    
    if not all_results:
        print("No valid results found!")
        return None
    
    # Convert to DataFrame for easier manipulation
    results_list = []
    for (model, warmstart, rounding), metrics in all_results.items():
        metrics_copy = metrics.copy()
        metrics_copy.update({
            'model': model,
            'warmstart_mode': warmstart, 
            'rounding_method': rounding
        })
        results_list.append(metrics_copy)
    
    df = pd.DataFrame(results_list)
    
    # Create plots if requested
    if create_plots:
        print(f"\nCreating visualization plots...")
        for csv_file in predictions_path.glob("results-*.csv"):
            try:
                create_visualizations_from_csv(
                    predictions_folder, csv_file, model_name_mapping,
                    output_dir=plot_output_dir, max_graphs=max_plots_per_config
                )
            except Exception as e:
                print(f"Error creating plots for {csv_file.name}: {e}")
    
    # Display results summary
    print("\n" + "="*100)
    print("EXPERIMENTAL RESULTS SUMMARY")
    print("="*100)
    
    print(f"\nProcessed {len(results_list)} configurations:")
    for _, row in df.iterrows():
        print(f"  {row['model']} | {row['warmstart_mode']} | {row['rounding_method']} | "
              f"F1: {row['avg_f1']:.3f} | Δ: {row['avg_delta']:.1f}")
    
    # Create and save LaTeX table
    latex_table = create_latex_table(df)
    
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            f.write(latex_table)
        print(f"\nLaTeX table saved to: {output_file}")
    
    # Also save results as CSV for further analysis
    csv_output = predictions_path / "processed_results_summary.csv"
    df.to_csv(csv_output, index=False)
    print(f"Results summary saved to: {csv_output}")
    
    print("\nLaTeX Table Preview:")
    print("="*50)
    print(latex_table)
    
    if create_plots:
        print(f"\nVisualization plots saved to: {plot_output_dir}")
    
    return df

# Example usage
if __name__ == "__main__":
    import sys
    
    # Allow command line argument for predictions folder
    if len(sys.argv) > 1:
        predictions_folder = sys.argv[1]
    else:
        # Default path for testing
        predictions_folder = r"C:\Users\denni\Documents\thesis_dnr_gnn_dev\data\split_datasets\test_test\predictions"
    
    print(f"Using predictions folder: {predictions_folder}")
    
    # Map your actual model names to the desired display names
    model_mapping = {
        "jumping-wave-12": "GIN",       # Your actual model names from predict_then_optimize.py
       # "AdvancedMLP": "GAT",     # Alternative mapping if model name is different
       # "GIN_Model": "GIN",       # Add your actual GIN model name
       # "GCN_Model": "GCN",       # Add your actual GCN model name
        # Add more mappings as needed
    }
    
    # Process all results and generate table
    results_df = save_latex_ready_table(
        predictions_folder=predictions_folder,
        model_name_mapping=model_mapping,
        output_path="experimental_results_table.tex",
        create_plots=True,
        plot_output_dir="prediction_visualizations",
        max_plots_per_config=2  # Reduced for testing
    )
    
    if results_df is not None:
        print(f"\nProcessing complete! Generated {len(results_df)} result entries.")
        print("\nSample results:")
        print(results_df[['model', 'warmstart_mode', 'rounding_method', 'avg_f1', 'avg_delta']].head())
    else:
        print("No results generated. Please check your data and paths.")