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
   
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))
   
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
   
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
   
    return specificity, sensitivity

def calculate_f1_majority(y_true, y_pred):
    """Calculate F1 score for majority class (class 0)"""
    return f1_score(y_true, y_pred, pos_label=0, average='binary', zero_division=0)

def check_network_radiality(predictions_folder, filename_pattern, experiment_id):
    """Check if a specific network configuration is radial"""
    try:
        predictions_path = Path(predictions_folder)
        prediction_folders = list(predictions_path.glob(f"prediction-{filename_pattern}*"))
       
        if prediction_folders:
            prediction_folder = prediction_folders[0]
            prediction_networks_folder = prediction_folder / "pandapower_networks"
           
            if prediction_networks_folder.exists():
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

def load_ground_truth_from_no_warmstart(predictions_folder, use_no_warmstart_gt=False):
    """Load ground truth from no_warmstart results if toggle is enabled"""
    if not use_no_warmstart_gt:
        return None
    
    predictions_path = Path(predictions_folder)
    

    no_warmstart_csv = None
    for csv_file in predictions_path.glob("results-*optimization_without_warmstart.csv"):
        no_warmstart_csv = csv_file
        break
    
    if no_warmstart_csv and no_warmstart_csv.exists():
        print(f"Loading ground truth from: {no_warmstart_csv.name}")
        df = pd.read_csv(no_warmstart_csv)
        # Create a dictionary mapping experiment names to their final_optima
        gt_dict = {}
        for idx, row in df.iterrows():
            # Assuming the CSVs have the same order and graph_id
            if 'reconfigurationgraph_id' in row:
                graph_id = row['reconfigurationgraph_id']
                final_optima = json.loads(row['final_optima'])
                gt_dict[graph_id] = final_optima
            else:
                # If no graph_id, use index
                gt_dict[idx] = json.loads(row['final_optima'])
        return gt_dict
    else:
        print("Warning: Could not find optimization_without_warmstart CSV file")
        return None

def detect_infeasibility_by_no_change(row, initial_state, final_optima, solve_time_threshold=0.1):
    """
    Detect likely infeasibility by checking if solve time is extremely low 
    and no switches have been changed from initial state
    """
    try:
        solve_time = row.get('solve_time', float('inf'))
        if pd.isna(solve_time):
            solve_time = float('inf')
        
        if solve_time < solve_time_threshold:
            if initial_state == final_optima:
                return True, 'no_change', f"No switches changed with very low solve time ({solve_time:.4f}s)"
    except:
        pass
    
    return False, None, ""

def process_all_experiments_detailed(predictions_folder, model_name_mapping=None, output_path=None, 
                                   debug_mode=False, use_no_warmstart_gt=False):
    """Process all CSV files and create one row per experiment"""
    predictions_path = Path(predictions_folder)
   
    # Find all CSV files
    csv_files = list(predictions_path.glob("results-*.csv"))
    print(f"Found {len(csv_files)} CSV files to process")

    if debug_mode:
        csv_files = csv_files[:1]
        print(f"DEBUG MODE: Processing only first CSV file")
    
    # Load alternative ground truth if requested
    no_warmstart_gt = load_ground_truth_from_no_warmstart(predictions_folder, use_no_warmstart_gt)
   
    all_rows = []
   
    for csv_file in csv_files:
        print(f"\nProcessing {csv_file.name}...")
       
        try:
            df = pd.read_csv(csv_file)
           
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
       
            display_name = model_name
            if model_name_mapping and model_name in model_name_mapping:
                display_name = model_name_mapping[model_name]
           
            filename_pattern = filename.replace('results-', '')
           
            # Process each experiment 
            for idx, row in df.iterrows():
                try:
                    initial_state = json.loads(row['initial_state']) if 'initial_state' in row else None
                    
                    final_optima = json.loads(row['final_optima'])
                    
                    # Determine which ground truth to use
                    if use_no_warmstart_gt and no_warmstart_gt:
                        # Try to match by graph_id or index
                        if 'reconfigurationgraph_id' in row and row['reconfigurationgraph_id'] in no_warmstart_gt:
                            ground_truth = no_warmstart_gt[row['reconfigurationgraph_id']]
                        elif idx in no_warmstart_gt:
                            ground_truth = no_warmstart_gt[idx]
                        else:
                            # Fallback to original ground truth
                            ground_truth = json.loads(row['ground_truth'])
                    else:
                        ground_truth = json.loads(row['ground_truth'])
                   
                    # Check if there's an error column indicating infeasibility
                    is_infeasible = False
                    infeasible_type = None
                    error_msg = ""
                    if 'error' in row:
                        error_val = row['error']
                        if (isinstance(error_val, bool) and error_val) or \
                           (isinstance(error_val, str) and error_val.lower() == 'true') or \
                           (pd.notna(error_val) and str(error_val).strip() and str(error_val).strip().lower() != 'false'):
                            is_infeasible = True
                            infeasible_type = 'error'
                            error_msg = str(error_val)
                   
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
                    
                    #Check for infeasibility by no changes and low solve time
                    if not is_infeasible and initial_state is not None:
                        no_change_infeasible, no_change_type, no_change_msg = detect_infeasibility_by_no_change(
                            row, initial_state, final_optima, solve_time_threshold=20
                        )
                        if no_change_infeasible:
                            is_infeasible = True
                            infeasible_type = 'no_change'
                            error_msg = no_change_msg
                   
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
                       
                        # topology error metric
                        topology_error_hamming = calculate_topology_error_hamming(ground_truth, final_optima)
                       
                    else:
                        mcc = f1_majority = balanced_acc = spec = sens = 0.0
                        topology_error_hamming = 1.0  
                   
                    # Get loss information
                    loss_ground_truth = row.get('loss_ground_truth', np.nan)
                    loss_final = row.get('loss_final', np.nan)
                    loss_difference = abs(loss_final - loss_ground_truth) if not np.isnan(loss_final) and not np.isnan(loss_ground_truth) else np.nan
                   
                    # Check radiality if prediction is wrong and not infeasible
                    is_radial = is_connected = None
                    if not is_correct and not is_infeasible:
                        is_radial, is_connected = check_network_radiality(
                            predictions_folder, filename_pattern, idx
                        )
                   
                    # Determine category 
                    if is_correct:
                        category = 'correct'
                    elif is_infeasible:
                        if infeasible_type == 'error':
                            category = 'infeasible_error'
                        elif infeasible_type == 'time_limit':
                            category = 'infeasible_time'
                        elif infeasible_type == 'no_change':
                            category = 'infeasible_no_change'
                        else:  # both
                            category = 'infeasible_error'  
                    else:
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
                        'topology_error_hamming': topology_error_hamming,
                        'solve_time': solve_time,
                        'loss_ground_truth': loss_ground_truth,
                        'loss_final': loss_final,
                        'loss_difference': loss_difference,
                        'is_radial': is_radial,
                        'is_connected': is_connected,
                        'category': category,
                        'using_no_warmstart_gt': use_no_warmstart_gt
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
       
        # Count categories each graph in exactly one category
        correct_count = sum(group['category'] == 'correct')
        radial_wrong = sum(group['category'] == 'radial_wrong')
        non_radial_wrong = sum(group['category'] == 'non_radial_wrong')
        infeasible_error_count = sum(group['category'] == 'infeasible_error')
        infeasible_time_count = sum(group['category'] == 'infeasible_time')
        infeasible_no_change_count = sum(group['category'] == 'infeasible_no_change')
       
        # Total infeasible 
        infeasible_total = infeasible_error_count + infeasible_time_count + infeasible_no_change_count
       
        # Verify each graph is counted exactly once
        total_categorized = correct_count + radial_wrong + non_radial_wrong + infeasible_error_count + infeasible_time_count + infeasible_no_change_count
        assert total_categorized == len(group), f"Category count mismatch: {total_categorized} != {len(group)}"
       
        # Time calculations
        feasible_times = group[~group['is_infeasible']]['solve_time']
        time_feasible = feasible_times.mean() if len(feasible_times) > 0 else 0.0
       
        # For infeasible assume max time
        max_time = 6000
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
            'Topology Error (Hamming)': avg_topology_error,
            'Time [s] (feasible only)': time_feasible,
            'Time [s] (including infeasible)': time_including_infeasible,
            'Difference in loss between optima': avg_loss_diff,
            'number of graphs: predicted correctly': correct_count,
            'number of graphs: Radial wrong': radial_wrong,
            'number of graphs: Non-Radial wrong': non_radial_wrong,
            'number of graphs: Infeasible (error)': infeasible_error_count,
            'number of graphs: Infeasible (time)': infeasible_time_count,
            'number of graphs: Infeasible (no change)': infeasible_no_change_count,
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
        latex_table = create_latex_table_overleaf_format(summary_df)
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            f.write(latex_table)
        print(f"LaTeX table saved to: {output_file}")
   
    print(f"\nProcessing complete! Generated {len(detailed_df)} detailed experiment rows.")
    print(f"Summary contains {len(summary_df)} configuration rows.")
    print(f"Ground truth mode: {'no_warmstart' if use_no_warmstart_gt else 'original CSV'}")
   
    return detailed_df, summary_df

def create_latex_table_overleaf_format(results_df):
    """Create LaTeX table matching the exact format from the overleaf example"""
    if results_df.empty:
        return "No results to display"
   
    # Start LaTeX table
    latex_lines = [
        "\\begin{table*}[t]",
        "  \\caption{Model Metrics and Error Distribution Comparison}",
        "  \\label{tab:full-result}",
        "  \\centering",
        "  \\small",
        "  \\renewcommand{\\arraystretch}{1.2}",
        "  \\resizebox{\\textwidth}{!}{%",
        "    \\arrayrulecolor{black}%",
        "    \\begin{tabular}{lllccccccccccc}",
        "      \\toprule",
        "      Model & Method & Variant & Time [F / I] & MCC & Balanced Acc. & F1-Minority & Specificity & Sensitivity & TopErr & \\# Correct & \\# Radial & \\# Non-Radial & Inf. (E/T) \\\\",
        "      \\midrule"
    ]
    
    # Group by GNN type
    for gnn_type in ['GCN', 'GAT', 'GIN']:
        gnn_rows = results_df[results_df['GNN type'] == gnn_type]
        if gnn_rows.empty:
            continue
            
        # First add header for this GNN type
        latex_lines.append(f"      % --- {gnn_type} ---")
        latex_lines.append(f"      \\multirow{{13}}{{*}}{{{gnn_type}}}")
        
        # Process different methods
        first_row = True
        
        # DirectPrediction methods
        for _, row in gnn_rows.iterrows():
            if 'round' in row['experiments'] and 'optimization_without_warmstart' in row['experiments']:
                method_str = "& \\multirow{2}{*}{DirectPrediction}" if first_row else "&"
                variant = "Rounding-only"
                first_row = False
            elif 'Best' in row['experiments'] and 'optimization_without_warmstart' in row['experiments']:
                method_str = "&"
                variant = "PhyR-only"
            else:
                continue
                
            latex_lines.append(
                f"        {method_str}"
                f"          & {variant:<20} & {row['Time [s] (feasible only)']:.0f}/{row['Time [s] (including infeasible)']:.0f} & "
                f"{row['MCC score']:.3f} & {row['Balanced Accuracy']:.3f} & "
                f"{row['F1-majority score']:.3f} & {row['Specificity']:.3f} & "
                f"{row['Sensitivity']:.3f} & {row['Topology Error (Hamming)']:.1f} & "
                f"{row['number of graphs: predicted correctly']} & "
                f"{row['number of graphs: Radial wrong']} & {row['number of graphs: Non-Radial wrong']} & "
                f"{row['number of graphs: Infeasible (error)']}/{row['number of graphs: Infeasible (time)']} \\\\"
            )
        
        # Add separator line
        latex_lines.append("      \\arrayrulecolor{gray!60}\\cdashline{2-14}")
        
        # SoftWarmStart methods
        first_soft = True
        for _, row in gnn_rows.iterrows():
            if 'float' in row['experiments']:
                method_str = "& \\multirow{3}{*}{SoftWarmStart}" if first_soft else "&"
                variant = "Float"
                first_soft = False
            elif 'soft' in row['experiments'] and 'Best' in row['experiments']:
                method_str = "&"
                variant = "PhyR-soft"
            elif 'soft' in row['experiments'] and 'round' in row['experiments']:
                method_str = "&"
                variant = "Rounding-soft"
            else:
                continue
                
            latex_lines.append(
                f"        {method_str}"
                f"          & {variant:<20} & {row['Time [s] (feasible only)']:.0f}/{row['Time [s] (including infeasible)']:.0f} & "
                f"{row['MCC score']:.3f} & {row['Balanced Accuracy']:.3f} & "
                f"{row['F1-majority score']:.3f} & {row['Specificity']:.3f} & "
                f"{row['Sensitivity']:.3f} & {row['Topology Error (Hamming)']:.1f} & "
                f"{row['number of graphs: predicted correctly']} & "
                f"{row['number of graphs: Radial wrong']} & {row['number of graphs: Non-Radial wrong']} & "
                f"{row['number of graphs: Infeasible (error)']}/{row['number of graphs: Infeasible (time)']} \\\\"
            )
        
        # Add separator line
        latex_lines.append("      \\cdashline{2-14}")
        
        # HardWarmStart methods
        first_hard = True
        for _, row in gnn_rows.iterrows():
            if 'soft' in row['experiments'] or 'float' in row['experiments'] or 'without_warmstart' in row['experiments']:
                continue
                
            method_str = "& \\multirow{8}{*}{HardWarmStart}" if first_hard else "&"
            first_hard = False
            if 'Best' in row['experiments']:
                if '0.95' in row['experiments']:
                    variant = "PhyR-0.95"
                elif '0.975' in row['experiments']:
                    variant = "PhyR-0.975"
                elif '0.99' in row['experiments'] and '0.999' not in row['experiments']:
                    variant = "PhyR-0.99"
                elif '0.999' in row['experiments']:
                    variant = "PhyR-0.999"
                else:
                    variant = "PhyR"
            elif 'round' in row['experiments']:
                if '0.95' in row['experiments']:
                    variant = "Round-0.95"
                elif '0.975' in row['experiments']:
                    variant = "Round-0.975"
                elif '0.99' in row['experiments'] and '0.999' not in row['experiments']:
                    variant = "Round-0.99"
                elif '0.999' in row['experiments']:
                    variant = "Round-0.999"
                else:
                    variant = "Round"
            else:
                continue
                
            latex_lines.append(
                f"        {method_str}"
                f"          & {variant:<20} & {row['Time [s] (feasible only)']:.0f}/{row['Time [s] (including infeasible)']:.0f} & "
                f"{row['MCC score']:.3f} & {row['Balanced Accuracy']:.3f} & "
                f"{row['F1-majority score']:.3f} & {row['Specificity']:.3f} & "
                f"{row['Sensitivity']:.3f} & {row['Topology Error (Hamming)']:.1f} & "
                f"{row['number of graphs: predicted correctly']} & "
                f"{row['number of graphs: Radial wrong']} & {row['number of graphs: Non-Radial wrong']} & "
                f"{row['number of graphs: Infeasible (error)']}/{row['number of graphs: Infeasible (time)']} \\\\"
            )
            
            # Add separator between PhyR and Round methods
            if variant == "PhyR-0.999":
                latex_lines.append("      \\arrayrulecolor{gray!60}\\cdashline{2-14}")
        
        # Add separator between GNN types
        if gnn_type != 'GIN':
            latex_lines.append("      \\arrayrulecolor{black}\\midrule")
   
    # Close table
    latex_lines.extend([
        "      \\bottomrule",
        "    \\end{tabular}%",
        "  }",
        "\\end{table*}"
    ])
   
    return "\n".join(latex_lines)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process experiment results with enhanced features')
    parser.add_argument('predictions_folder', nargs='?', 
                        default=r"C:\Users\denni\Documents\thesis_dnr_gnn_dev\data\test\predictions",
                        help='Path to predictions folder')
    parser.add_argument('--use-no-warmstart-gt', action='store_true',
                        help='Use optimization_without_warmstart results as ground truth')
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode - process only first CSV')
    
    args = parser.parse_args()
    
    print(f"Using predictions folder: {args.predictions_folder}")
    print(f"Ground truth mode: {'no_warmstart' if args.use_no_warmstart_gt else 'original CSV'}")
    
    model_mapping = {
        "GAT-stage2-hyperparameter-tuning-Best": "GAT",      
        "GIN-stage2-hyperparameter-tuning-Best": "GIN",    
        "GCN-stage2-hyperparameter-tuning-Best": "GCN",      
    }
   
    # Process all results and generate table
    detailed_df, summary_df = process_all_experiments_detailed(
        predictions_folder=args.predictions_folder,
        model_name_mapping=model_mapping,
        output_path="experiment_results_table.tex",
        debug_mode=args.debug,
        use_no_warmstart_gt=args.use_no_warmstart_gt
    )
   
    if summary_df is not None:
        print(f"\nSummary by GNN type:")
        summary = summary_df.groupby('GNN type')[['MCC score', 'F1-majority score', 'Balanced Accuracy']].mean()
        print(summary)
        
        print(f"\nInfeasibility breakdown:")
        for _, row in summary_df.iterrows():
            print(f"{row['Name of model']} - {row['experiments']}:")
            print(f"  Error: {row['number of graphs: Infeasible (error)']}")
            print(f"  Time: {row['number of graphs: Infeasible (time)']}")
            print(f"  No change: {row['number of graphs: Infeasible (no change)']}")