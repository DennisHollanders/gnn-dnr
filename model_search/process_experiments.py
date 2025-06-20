import pandas as pd
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, recall_score, precision_score, balanced_accuracy_score, matthews_corrcoef
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
def calculate_topology_error_hamming(y_true, y_pred):    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Handle length mismatch
    if len(y_true) != len(y_pred):
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
    
    if len(y_true) == 0:
        return 1.0 
    hamming_distance = np.sum(y_true != y_pred)
    topology_error = hamming_distance / len(y_true)
    
    return topology_error

def calculate_specificity_sensitivity(y_true, y_pred):
    """Calculate specificity and sensitivity"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # True Positives, False Positives, True Negatives, False Negatives
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    # Sensitivity (True Positive Rate / Recall)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # Specificity (True Negative Rate)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return specificity, sensitivity

def calculate_f1_majority(y_true, y_pred):
    """Calculate F1 score for majority class (class 0)"""
    return f1_score(y_true, y_pred, pos_label=0, average='binary', zero_division=0)

def check_network_radiality(predictions_folder, filename_pattern, experiment_id):
    """Check if a specific network configuration is radial"""
    try:
        # Look for corresponding prediction folders
        predictions_path = Path(predictions_folder)
        prediction_folders = list(predictions_path.glob(f"prediction-{filename_pattern}*"))
        
        if prediction_folders:
            prediction_folder = prediction_folders[0]
            prediction_networks_folder = prediction_folder / "pandapower_networks"
            
            if prediction_networks_folder.exists():
                # Look for network file corresponding to experiment_id
                json_files = list(prediction_networks_folder.glob("*.json"))
                if experiment_id < len(json_files):
                    json_file = json_files[experiment_id]
                    
                    # Load final prediction network
                    final_net = pp.from_json(str(json_file))
                    
                    # Check if final network is radial and connected
                    is_radial, is_connected = is_radial_and_connected(final_net, include_switches=True)
                    return is_radial, is_connected
    except Exception as e:
        print(f"Error checking network radiality: {e}")
        
    return None, None

def process_all_experiments_detailed(predictions_folder, model_name_mapping=None, output_path=None, debug_mode=False):
    """Process all CSV files and create one row per experiment"""
    predictions_path = Path(predictions_folder)
    
    # Find all CSV files
    csv_files = list(predictions_path.glob("results-*.csv"))
    print(f"Found {len(csv_files)} CSV files to process")
    
    # Debug mode - process only first CSV
    if debug_mode:
        csv_files = csv_files[:1]
        print(f"DEBUG MODE: Processing only first CSV file")
    
    all_rows = []
    
    for csv_file in csv_files:
        print(f"\nProcessing {csv_file.name}...")
        
        try:
            df = pd.read_csv(csv_file)
            
            # Parse filename to extract configuration
            filename = csv_file.stem  # results-{model}-{rounding}-{warmstart}
            parts = filename.replace("results-", "").split("-")
            
            # Handle complex model names
            if len(parts) >= 3:
                warmstart_mode = parts[-1]
                rounding_method = parts[-2]
                model_name = "-".join(parts[:-2])
            else:
                model_name = "unknown"
                rounding_method = "unknown" 
                warmstart_mode = "unknown"
            
            # Extract GNN type
            gnn_type = "unknown"
            if "GAT" in model_name.upper():
                gnn_type = "GAT"
            elif "GIN" in model_name.upper():
                gnn_type = "GIN"
            elif "GCN" in model_name.upper():
                gnn_type = "GCN"
            else:
                gnn_type = model_name
            
            # Apply model name mapping if provided
            display_name = model_name
            if model_name_mapping and model_name in model_name_mapping:
                display_name = model_name_mapping[model_name]
            
            filename_pattern = filename.replace('results-', '')
            
            # Process each experiment (row) in the CSV
            for idx, row in df.iterrows():
                try:
                    # Parse JSON fields
                    ground_truth = json.loads(row['ground_truth'])
                    final_optima = json.loads(row['final_optima'])
                    
                    # Check if there's an error column indicating infeasibility
                    is_infeasible = False
                    infeasible_type = None
                    error_msg = ""
                    
                    # Check error column - if True or any non-empty value, it's infeasible
                    if 'error' in row:
                        error_val = row['error']
                        # Check for True, 'True', 'true', or any non-empty string
                        if (isinstance(error_val, bool) and error_val) or \
                           (isinstance(error_val, str) and error_val.lower() == 'true') or \
                           (pd.notna(error_val) and str(error_val).strip() and str(error_val).strip().lower() != 'false'):
                            is_infeasible = True
                            infeasible_type = 'error'
                            error_msg = str(error_val)
                    
                    # Check solve time for timeout (>299 seconds indicates timeout/infeasibility)
                    solve_time = row.get('solve_time', 0.0)
                    if pd.isna(solve_time):
                        solve_time = 0.0
                    elif solve_time > 599:
                        is_infeasible = True
                        if infeasible_type == 'error':
                            # Both error and timeout
                            infeasible_type = 'both'
                            error_msg += f" (solve_time: {solve_time}s)"
                        else:
                            infeasible_type = 'time_limit'
                            error_msg = f"Timeout - solve_time: {solve_time}s"
                    
                    # Handle length mismatches
                    if len(ground_truth) != len(final_optima):
                        min_len = min(len(ground_truth), len(final_optima))
                        if min_len > 0:
                            ground_truth = ground_truth[:min_len]
                            final_optima = final_optima[:min_len]
                        else:
                            is_infeasible = True
                            error_msg = "Length mismatch - no valid predictions"
                    
                    # Compare configurations directly
                    is_correct = ground_truth == final_optima if not is_infeasible else False
                    
                    # Calculate metrics for this experiment
                    if not is_infeasible and len(ground_truth) > 0:
                        mcc = matthews_corrcoef(ground_truth, final_optima) if len(set(ground_truth)) > 1 else 0.0
                        f1_majority = calculate_f1_majority(ground_truth, final_optima)
                        balanced_acc = balanced_accuracy_score(ground_truth, final_optima)
                        spec, sens = calculate_specificity_sensitivity(ground_truth, final_optima)
                        
                        # Add GraPhyR topology error metric
                        topology_error_hamming = calculate_topology_error_hamming(ground_truth, final_optima)
                        
                    else:
                        mcc = f1_majority = balanced_acc = spec = sens = 0.0
                        topology_error_hamming = 1.0  # Maximum error for infeasible cases
                    
                    # Solve time is already retrieved above
                    
                    # Get loss information
                    loss_ground_truth = row.get('loss_ground_truth', np.nan)
                    loss_final = row.get('loss_final', np.nan)
                    loss_difference = abs(loss_final - loss_ground_truth) if not np.isnan(loss_final) and not np.isnan(loss_ground_truth) else np.nan
                    
                    # Check radiality if prediction is wrong AND not infeasible
                    is_radial = is_connected = None
                    if not is_correct and not is_infeasible:
                        is_radial, is_connected = check_network_radiality(
                            predictions_folder, filename_pattern, idx
                        )
                    
                    # Determine category - each graph goes in exactly one category
                    if is_correct:
                        category = 'correct'
                    elif is_infeasible:
                        if infeasible_type == 'error':
                            category = 'infeasible_error'
                        elif infeasible_type == 'time_limit':
                            category = 'infeasible_time'
                        else:  # both
                            category = 'infeasible_error'  # Count in error category if both
                    else:
                        # Not correct and not infeasible - check radiality
                        if is_radial and is_connected:
                            category = 'radial_wrong'
                        else:
                            category = 'non_radial_wrong'
                    
                    # Create detailed row
                    experiment_row = {
                        'experiment_id': idx,
                        'csv_file': csv_file.name,
                        'model_name': model_name,
                        'display_name': display_name,
                        'gnn_type': gnn_type,
                        'rounding_method': rounding_method,
                        'warmstart_mode': warmstart_mode,
                        'ground_truth': str(ground_truth),
                        'final_optima': str(final_optima),
                        'is_correct': is_correct,
                        'is_infeasible': is_infeasible,
                        'infeasible_type': infeasible_type,
                        'error_message': error_msg,
                        'num_switches': len(ground_truth),
                        'mcc_score': mcc,
                        'f1_majority_score': f1_majority,
                        'balanced_accuracy': balanced_acc,
                        'specificity': spec,
                        'sensitivity': sens,
                        'topology_error_hamming': topology_error_hamming,  # Add this line
                        'solve_time': solve_time,
                        'loss_ground_truth': loss_ground_truth,
                        'loss_final': loss_final,
                        'loss_difference': loss_difference,
                        'is_radial': is_radial,
                        'is_connected': is_connected,
                        'category': category
                    }
                    
                    all_rows.append(experiment_row)
                    
                except Exception as e:
                    print(f"    Error processing experiment {idx}: {e}")
                    continue
            
            print(f"  Processed {len(df)} experiments from {csv_file.name}")
            
        except Exception as e:
            print(f"Error processing {csv_file.name}: {e}")
            continue
    
    if not all_rows:
        print("No valid results found!")
        return None
    
    # Convert to DataFrame
    detailed_df = pd.DataFrame(all_rows)
    
    # Save detailed results
    detailed_csv = predictions_path / "detailed_experiment_results.csv"
    detailed_df.to_csv(detailed_csv, index=False)
    print(f"\nDetailed results saved to: {detailed_csv}")
    
    # Create summary by configuration
    summary_results = []
    for config, group in detailed_df.groupby(['model_name', 'rounding_method', 'warmstart_mode']):
        model_name, rounding, warmstart = config
        
        # Calculate aggregated metrics
        all_gt = []
        all_pred = []
        for _, exp in group.iterrows():
            if not exp['is_infeasible']:
                gt = eval(exp['ground_truth'])
                pred = eval(exp['final_optima'])
                all_gt.extend(gt)
                all_pred.extend(pred)
        
        if all_gt:
            overall_mcc = matthews_corrcoef(all_gt, all_pred) if len(set(all_gt)) > 1 else 0.0
            overall_f1_majority = calculate_f1_majority(all_gt, all_pred)
            overall_balanced_acc = balanced_accuracy_score(all_gt, all_pred)
            overall_spec, overall_sens = calculate_specificity_sensitivity(all_gt, all_pred)
        else:
            overall_mcc = overall_f1_majority = overall_balanced_acc = overall_spec = overall_sens = 0.0
        
        # Calculate aggregated topology error
        all_topology_errors = []
        for _, exp in group.iterrows():
            if not exp['is_infeasible']:
                all_topology_errors.append(exp['topology_error_hamming'])
        
        avg_topology_error = np.mean(all_topology_errors) if all_topology_errors else 1.0
        
        # Count categories - each graph in exactly one category
        correct_count = sum(group['category'] == 'correct')
        radial_wrong = sum(group['category'] == 'radial_wrong')
        non_radial_wrong = sum(group['category'] == 'non_radial_wrong')
        infeasible_error_count = sum(group['category'] == 'infeasible_error')
        infeasible_time_count = sum(group['category'] == 'infeasible_time')
        
        # Total infeasible (for backward compatibility)
        infeasible_total = infeasible_error_count + infeasible_time_count
        
        # Verify each graph is counted exactly once
        total_categorized = correct_count + radial_wrong + non_radial_wrong + infeasible_error_count + infeasible_time_count
        assert total_categorized == len(group), f"Category count mismatch: {total_categorized} != {len(group)}"
        
        # Time calculations
        feasible_times = group[~group['is_infeasible']]['solve_time']
        time_feasible = feasible_times.mean() if len(feasible_times) > 0 else 0.0
        
        # For infeasible, assume max time
        max_time = 300.0
        all_times = group['solve_time'].fillna(max_time)
        all_times[group['is_infeasible']] = max_time
        time_including_infeasible = all_times.mean()
        
        # Loss difference
        valid_losses = group['loss_difference'].dropna()
        avg_loss_diff = valid_losses.mean() if len(valid_losses) > 0 else 0.0
        
        summary_row = {
            'Name of model': model_name_mapping.get(model_name, model_name) if model_name_mapping else model_name,
            'GNN type': group.iloc[0]['gnn_type'],
            'experiments': f"{model_name}-{rounding}-{warmstart}",
            'MCC score': overall_mcc,
            'F1-majority score': overall_f1_majority,
            'Balanced Accuracy': overall_balanced_acc,
            'Specificity': overall_spec,
            'Sensitivity': overall_sens,
            'Topology Error (Hamming)': avg_topology_error,  # Add this line
            'Time [s] (feasible only)': time_feasible,
            'Time [s] (including infeasible)': time_including_infeasible,
            'Difference in loss between optima': avg_loss_diff,
            'number of graphs: predicted correctly': correct_count,
            'number of graphs: Radial wrong': radial_wrong,
            'number of graphs: Non-Radial wrong': non_radial_wrong,
            'number of graphs: Infeasible (error)': infeasible_error_count,
            'number of graphs: Infeasible (time)': infeasible_time_count,
            'number of graphs: Infeasible (total)': infeasible_total,
            'total_experiments': len(group)
        }
        summary_results.append(summary_row)
    
    summary_df = pd.DataFrame(summary_results)
    
    # Save summary
    summary_csv = predictions_path / "experiment_summary_results.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"Summary results saved to: {summary_csv}")
    
    # Create LaTeX table
    if output_path:
        latex_table = create_latex_table_new_format(summary_df)
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            f.write(latex_table)
        print(f"LaTeX table saved to: {output_file}")
    
    print(f"\nProcessing complete! Generated {len(detailed_df)} detailed experiment rows.")
    print(f"Summary contains {len(summary_df)} configuration rows.")
    
    return detailed_df, summary_df

def create_latex_table_new_format(results_df):
    """Create LaTeX table with the new format including topology error"""
    if results_df.empty:
        return "No results to display"
    
    # Start LaTeX table
    latex_lines = [
        "\\begin{table*}[p]",
        "  \\centering", 
        "  \\small",
        "  \\renewcommand{\\arraystretch}{1.2}",
        "  \\resizebox{\\textwidth}{!}{%",
        "    \\begin{tabular}{llcccccccccccccc}",  # Added one more 'c' for topology error
        "      \\toprule",
        "      Model & GNN & MCC & F1-maj & Bal.Acc & Spec & Sens & TopErr & Time(F) & Time(I) & Loss & Correct & Radial & Non-Rad & Inf(E) & Inf(T) \\\\",  # Added TopErr column
        "      \\midrule"
    ]
    
    for idx, row in results_df.iterrows():
        latex_lines.append(
            f"      {row['Name of model']} & {row['GNN type']} & "
            f"{row['MCC score']:.3f} & {row['F1-majority score']:.3f} & "
            f"{row['Balanced Accuracy']:.3f} & {row['Specificity']:.3f} & "
            f"{row['Sensitivity']:.3f} & {row['Topology Error (Hamming)']:.3f} & "  # Added topology error
            f"{row['Time [s] (feasible only)']:.1f} & "
            f"{row['Time [s] (including infeasible)']:.1f} & "
            f"{row['Difference in loss between optima']:.2f} & "
            f"{row['number of graphs: predicted correctly']} & "
            f"{row['number of graphs: Radial wrong']} & "
            f"{row['number of graphs: Non-Radial wrong']} & "
            f"{row['number of graphs: Infeasible (error)']} & "
            f"{row['number of graphs: Infeasible (time)']} \\\\"
        )
    
    # Close table
    latex_lines.extend([
        "      \\bottomrule",
        "    \\end{tabular}%",
        "  }",
        "  \\caption{Experiment Results Summary (TopErr = Topology Error Hamming distance, Inf(E) = Infeasible due to error, Inf(T) = Infeasible due to time limit)}",
        "  \\label{tab:experiment-results}",
        "\\end{table*}"
    ])
    
    return "\n".join(latex_lines)

# Example usage
if __name__ == "__main__":
    import sys
    
    # Allow command line argument for predictions folder
    if len(sys.argv) > 1:
        predictions_folder = sys.argv[1]
    else:
        # Default path for testing
        predictions_folder = r"C:\Users\denni\Documents\thesis_dnr_gnn_dev\data\predictions"
    
    print(f"Using predictions folder: {predictions_folder}")
    
    # Map your actual model names to the desired display names
    model_mapping = {
        "whole-paper-13": "GAT",       
        "devout-glitter-19": "GIN",     
        "ancient-bush-22": "GCN",       
    }
    
    # Process all results and generate table
    detailed_df, summary_df = process_all_experiments_detailed(
        predictions_folder=predictions_folder,
        model_name_mapping=model_mapping,
        output_path="experiment_results_table.tex",
        debug_mode=False  # Set to False to process all CSV files
    )
    
    if summary_df is not None:
        print(f"\nSummary by GNN type:")
        summary = summary_df.groupby('GNN type')[['MCC score', 'F1-majority score', 'Balanced Accuracy']].mean()
        print(summary)