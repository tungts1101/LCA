from matplotlib import pyplot as plt
import numpy as np


DATASETS = ["CIFAR100", "IN-R", "IN-A", "CUB", "OB", "VTAB", "CARS"]

METHOD_COLORS = {
    "MM - CIL": "#87CEEB",  # Sky Blue (lightest)
    "MM - LCA": "#1f77b4",  # Blue (darkest)
    "MOS": "#ffbb78",  # Light Orange (lightest)
    "MOS - CIL": "#ff9f3f",  # Medium Orange
    "MOS - LCA": "#d62728",  # Red-Orange (darkest)
    "SLCA": "#98df8a",  # Light Green (lightest)
    "SLCA - CIL": "#7fbf7f",  # Medium Light Green
    "SLCA - LCA": "#1f5f1f",  # Dark Green (darkest)
    "EASE": "#9467bd",  # Purple
    "APER + Adapter": "#ffc0cb",  # Light Pink (lightest)
    "APER + Finetune": "#f7b6d3",  # Medium Light Pink
    "APER + SSF": "#e377c2",  # Pink (medium)
    "APER + VPT-Deep": "#c5b0d5",  # Light Purple (medium-dark)
    "APER + VPT-Shallow": "#ff1493",  # Deep Pink (darkest)
    "L2P": "#e49c3a",  # Golden Yellow (lightest)
    "CODA-Prompt": "#bcbd22",  # Olive/Yellow-Green (medium)
    "DualPrompt": "#8c564b",  # Brown (darkest)
}


def get_method_color(method_name):
    """
    Get the consistent color for a given method name.
    Returns a default color if method not found in the predefined scheme.
    """
    return METHOD_COLORS.get(method_name, "#808080")  # Default to gray if not found


def set_publication_style():
    """
    Set matplotlib parameters for publication-ready figures with larger, more readable text.
    """
    plt.rcParams.update(
        {
            "font.size": 16,  # Base font size
            "axes.titlesize": 20,  # Title font size
            "axes.labelsize": 18,  # Axis label font size
            "xtick.labelsize": 16,  # X-axis tick label size
            "ytick.labelsize": 16,  # Y-axis tick label size
            "legend.fontsize": 16,  # Legend font size
            "axes.linewidth": 2,  # Axis border line width
            "xtick.major.width": 2,  # X-axis tick line width
            "ytick.major.width": 2,  # Y-axis tick line width
            "lines.linewidth": 2,  # Line width
            "font.family": "serif",  # Use serif font for professional look
            "text.usetex": False,  # Don't require LaTeX (can be set to True if available)
        }
    )


def reset_plot_style():
    """
    Reset matplotlib parameters to default values.
    """
    plt.rcdefaults()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def calculate_method_stats(dataset_name, method_name, stat_type="final"):
    """
    Calculate mean and standard deviation for a method in a specific dataset.

    Args:
        dataset_name (str): Name of the dataset (e.g., "CIFAR100", "IN-R", etc.)
        method_name (str): Name of the method (e.g., "MM - LCA", "EASE", etc.)
        stat_type (str): Type of statistic to calculate:
            - "final": Mean and std of final accuracies (last value of each run)
            - "all": Mean and std of all accuracy values across all tasks
            - "task": Mean and std for each task position

    Returns:
        dict: Dictionary containing 'mean', 'std', and optionally 'values' or 'task_stats'
    """
    if dataset_name not in results:
        return {"error": f"Dataset '{dataset_name}' not found"}

    if method_name not in results[dataset_name]:
        return {
            "error": f"Method '{method_name}' not found in dataset '{dataset_name}'"
        }

    method_data = results[dataset_name][method_name]

    if not method_data:
        return {
            "error": f"No data available for method '{method_name}' in dataset '{dataset_name}'"
        }

    # Handle both single arrays and nested arrays
    if isinstance(method_data[0], list):
        # Nested arrays (multiple seeds/runs)
        runs_data = method_data
    else:
        # Single array - treat as one run
        runs_data = [method_data]

    if stat_type == "final":
        # Calculate stats for final accuracies only
        final_accuracies = [run[-1] for run in runs_data]
        return {
            "mean": np.mean(final_accuracies),
            "std": (
                np.std(final_accuracies, ddof=1) if len(final_accuracies) > 1 else 0.0
            ),
            "values": final_accuracies,
            "count": len(final_accuracies),
        }

    elif stat_type == "all":
        # Calculate stats for all accuracy values across all tasks and runs
        all_values = [value for run in runs_data for value in run]
        return {
            "mean": np.mean(all_values),
            "std": np.std(all_values, ddof=1) if len(all_values) > 1 else 0.0,
            "values": all_values,
            "count": len(all_values),
        }

    elif stat_type == "task":
        # Calculate stats for each task position
        num_tasks = len(runs_data[0])  # Assume all runs have same number of tasks
        task_stats = []

        for task_idx in range(num_tasks):
            task_accuracies = [run[task_idx] for run in runs_data]
            task_stats.append(
                {
                    "task": task_idx + 1,
                    "mean": np.mean(task_accuracies),
                    "std": (
                        np.std(task_accuracies, ddof=1)
                        if len(task_accuracies) > 1
                        else 0.0
                    ),
                    "values": task_accuracies,
                }
            )

        return {
            "task_stats": task_stats,
            "num_tasks": num_tasks,
            "num_runs": len(runs_data),
        }

    else:
        return {
            "error": f"Unknown stat_type '{stat_type}'. Use 'final', 'all', or 'task'"
        }


def get_all_methods_stats(dataset_name, stat_type="final"):
    """
    Get statistics for all methods in a specific dataset.

    Args:
        dataset_name (str): Name of the dataset
        stat_type (str): Type of statistic ("final", "all", or "task")

    Returns:
        dict: Dictionary with method names as keys and their stats as values
    """
    if dataset_name not in results:
        return {"error": f"Dataset '{dataset_name}' not found"}

    all_stats = {}
    for method_name in results[dataset_name]:
        stats = calculate_method_stats(dataset_name, method_name, stat_type)
        if "error" not in stats:
            all_stats[method_name] = stats

    return all_stats


def compare_methods_stats(dataset_name, method_names, stat_type="final"):
    """
    Compare statistics between multiple methods for a specific dataset.

    Args:
        dataset_name (str): Name of the dataset
        method_names (list): List of method names to compare
        stat_type (str): Type of statistic ("final", "all", or "task")

    Returns:
        dict: Dictionary with comparison results
    """
    comparison = {}

    for method_name in method_names:
        stats = calculate_method_stats(dataset_name, method_name, stat_type)
        if "error" not in stats:
            comparison[method_name] = stats

    # Add ranking based on mean performance
    if stat_type in ["final", "all"]:
        # Sort methods by mean performance (descending)
        sorted_methods = sorted(
            comparison.items(), key=lambda x: x[1]["mean"], reverse=True
        )
        comparison["ranking"] = [
            {
                "rank": i + 1,
                "method": method,
                "mean": stats["mean"],
                "std": stats["std"],
            }
            for i, (method, stats) in enumerate(sorted_methods)
        ]

    return comparison


def print_all_methods_summary(stat_type="final", sort_by_dataset=True):
    """
    Print a comprehensive summary of mean and standard deviation for all methods across all datasets.

    Args:
        stat_type (str): Type of statistic to calculate:
            - "final": Mean and std of final accuracies (last value of each run)
            - "all": Mean and std of all accuracy values across all tasks
            - "task": Mean and std for each task position
        sort_by_dataset (bool): If True, organize by dataset; if False, organize by method
    """
    print("=" * 100)
    print(f"COMPREHENSIVE METHODS SUMMARY ({stat_type.upper()} ACCURACIES)")
    print("=" * 100)

    if sort_by_dataset:
        # Organize by dataset
        for dataset in DATASETS:
            if dataset not in results:
                continue

            print(f"\n{dataset} Dataset:")
            print("-" * 60)

            # Get all methods for this dataset and sort by performance
            dataset_methods = []
            for method_name in results[dataset]:
                stats = calculate_method_stats(dataset, method_name, stat_type)
                if "error" not in stats and "mean" in stats:
                    dataset_methods.append(
                        (method_name, stats["mean"], stats.get("std", 0))
                    )

            # Sort by mean performance (descending)
            dataset_methods.sort(key=lambda x: x[1], reverse=True)

            # Print results
            for i, (method, mean_val, std_val) in enumerate(dataset_methods, 1):
                if std_val > 0:
                    print(f"{i:2d}. {method:<20} | {mean_val:6.2f}% ± {std_val:5.2f}%")
                else:
                    print(f"{i:2d}. {method:<20} | {mean_val:6.2f}%")

    else:
        # Organize by method
        all_methods = set()
        for dataset in DATASETS:
            if dataset in results:
                all_methods.update(results[dataset].keys())

        all_methods = sorted(list(all_methods))

        for method in all_methods:
            print(f"\n{method}:")
            print("-" * 60)

            method_results = []
            for dataset in DATASETS:
                if dataset in results and method in results[dataset]:
                    stats = calculate_method_stats(dataset, method, stat_type)
                    if "error" not in stats and "mean" in stats:
                        method_results.append(
                            (dataset, stats["mean"], stats.get("std", 0))
                        )

            # Sort by performance (descending)
            method_results.sort(key=lambda x: x[1], reverse=True)

            # Print results
            for dataset, mean_val, std_val in method_results:
                if std_val > 0:
                    print(f"  {dataset:<12} | {mean_val:6.2f}% ± {std_val:5.2f}%")
                else:
                    print(f"  {dataset:<12} | {mean_val:6.2f}%")

    print("\n" + "=" * 100)


def print_latex_table(datasets=None, methods=None, stat_type="final", precision=1):
    """
    Print results in LaTeX table format with highest value in bold and second highest in italics.
    Format: Method & Dataset1 & Dataset2 & ... & DatasetN & Overall \\
    
    Args:
        datasets (list): List of datasets to include (default: all datasets)  
        methods (list): List of methods to include (default: all methods)
        stat_type (str): Type of statistic ("final", "all", or "task")
        precision (int): Number of decimal places for the values
    """
    if datasets is None:
        datasets = ["CIFAR100", "IN-R", "IN-A", "CUB", "OB", "VTAB", "CARS"]
    
    # Define method order and name mapping as shown in the image
    method_order_and_names = [
        ("APER + Adapter", "APER+Adapter"),
        ("APER + Finetune", "APER+Finetune"), 
        ("APER + SSF", "APER+SSF"),
        ("APER + VPT-Deep", "APER+VPT-Deep"),
        ("APER + VPT-Shallow", "APER+VPT-Shallow"),
        ("CODA-Prompt", "CODA-Prompt"),
        ("DualPrompt", "DualPrompt"),
        ("EASE", "EASE"),
        ("L2P", "L2P"),
        ("MOS", "MOS"),
        ("SLCA", "SLCA"),
        ("MM - CIL", "IM"),
        ("MM - LCA", "IM+LCA")
    ]

    # Filter datasets and methods that actually exist
    available_datasets = [d for d in datasets if d in results]
    available_methods = []
    method_name_mapping = {}
    
    for original_name, display_name in method_order_and_names:
        if any(original_name in results.get(d, {}) for d in available_datasets):
            available_methods.append(original_name)
            method_name_mapping[original_name] = display_name

    # First pass: collect all values and std for each dataset to find highest and second highest
    dataset_values = {}
    method_results = {}
    overall_values = {}  # For calculating overall averages

    for dataset in available_datasets:
        dataset_values[dataset] = []
        method_results[dataset] = {}

    for method in available_methods:
        overall_values[method] = []
        for dataset in available_datasets:
            if dataset in results and method in results[dataset]:
                stats = calculate_method_stats(dataset, method, stat_type)
                if "error" not in stats and "mean" in stats:
                    mean_val = stats["mean"]

                    # Special handling for MM - LCA: use predefined STD values
                    if method == "MM - LCA" and dataset in STD.get("MM - LCA", {}):
                        std_val = STD["MM - LCA"][dataset]
                    else:
                        std_val = stats.get("std", 0)

                    dataset_values[dataset].append(mean_val)
                    method_results[dataset][method] = {"mean": mean_val, "std": std_val}
                    overall_values[method].append(mean_val)
                else:
                    method_results[dataset][method] = None
            else:
                method_results[dataset][method] = None

    # Find highest and second highest values for each dataset
    dataset_rankings = {}
    for dataset in available_datasets:
        valid_values = [v for v in dataset_values[dataset] if v is not None]
        if len(valid_values) >= 2:
            sorted_values = sorted(valid_values, reverse=True)
            dataset_rankings[dataset] = {
                "highest": sorted_values[0],
                "second_highest": sorted_values[1],
            }
        elif len(valid_values) == 1:
            dataset_rankings[dataset] = {
                "highest": valid_values[0],
                "second_highest": None,
            }
        else:
            dataset_rankings[dataset] = {"highest": None, "second_highest": None}

    # Calculate overall rankings
    overall_means = {}
    for method in available_methods:
        if overall_values[method]:
            overall_means[method] = np.mean(overall_values[method])
        else:
            overall_means[method] = None

    valid_overall_values = [v for v in overall_means.values() if v is not None]
    if len(valid_overall_values) >= 2:
        sorted_overall = sorted(valid_overall_values, reverse=True)
        overall_rankings = {
            "highest": sorted_overall[0],
            "second_highest": sorted_overall[1],
        }
    elif len(valid_overall_values) == 1:
        overall_rankings = {
            "highest": valid_overall_values[0],
            "second_highest": None,
        }
    else:
        overall_rankings = {"highest": None, "second_highest": None}

    print("\n" + "=" * 120)
    print(f"LATEX TABLE FORMAT ({stat_type.upper()} ACCURACIES)")
    print("=" * 120)

    # Print all methods in the specified order
    for method in available_methods:
        display_name = method_name_mapping[method]
        row_values = [display_name]

        # Add dataset values
        for dataset in available_datasets:
            result = method_results[dataset].get(method)

            if result is not None:
                mean_val = result["mean"]
                std_val = result["std"]

                # Format based on ranking
                if (
                    dataset_rankings[dataset]["highest"] is not None
                    and abs(mean_val - dataset_rankings[dataset]["highest"]) < 0.01
                ):
                    # Highest value - bold
                    if std_val > 0:
                        cell_content = f"\\textbf{{{mean_val:.{precision}f} ± {std_val:.{precision}f}}}"
                    else:
                        cell_content = f"\\textbf{{{mean_val:.{precision}f}}}"
                elif (
                    dataset_rankings[dataset]["second_highest"] is not None
                    and abs(mean_val - dataset_rankings[dataset]["second_highest"])
                    < 0.01
                ):
                    # Second highest value - italic
                    if std_val > 0:
                        cell_content = f"\\textit{{{mean_val:.{precision}f} ± {std_val:.{precision}f}}}"
                    else:
                        cell_content = f"\\textit{{{mean_val:.{precision}f}}}"
                else:
                    # Regular value
                    if std_val > 0:
                        cell_content = f"{mean_val:.{precision}f} ± {std_val:.{precision}f}"
                    else:
                        cell_content = f"{mean_val:.{precision}f}"
            else:
                cell_content = "-"

            row_values.append(cell_content)

        # Add overall column
        if overall_means[method] is not None:
            overall_mean = overall_means[method]
            # Format based on overall ranking
            if (
                overall_rankings["highest"] is not None
                and abs(overall_mean - overall_rankings["highest"]) < 0.01
            ):
                # Highest overall value - bold
                overall_content = f"\\textbf{{{overall_mean:.{precision}f}}}"
            elif (
                overall_rankings["second_highest"] is not None
                and abs(overall_mean - overall_rankings["second_highest"]) < 0.01
            ):
                # Second highest overall value - italic
                overall_content = f"\\textit{{{overall_mean:.{precision}f}}}"
            else:
                # Regular overall value
                overall_content = f"{overall_mean:.{precision}f}"
        else:
            overall_content = "-"

        row_values.append(overall_content)

        # Print the complete row in the requested format
        row = " & ".join(row_values) + " \\\\"
        print(row)

    print("\n" + "=" * 120)
    print("LaTeX table ready to copy-paste into your paper!")
    print("Formatting: \\textbf{} = highest value, \\textit{} = second highest value")
    print("Values include mean ± std where available")
    print("Overall column shows average across all datasets")
    print("=" * 120)


def print_latex_table_only(datasets=None, methods=None, stat_type="final"):
    """
    Print ONLY the LaTeX table without any extra text - ready for copy-paste into paper.
    """
    if datasets is None:
        datasets = DATASETS
    if methods is None:
        # Get all unique methods across all datasets
        all_methods = set()
        for dataset in datasets:
            if dataset in results:
                all_methods.update(results[dataset].keys())
        methods = sorted(list(all_methods))

    # Filter datasets and methods that actually exist
    available_datasets = [d for d in datasets if d in results]
    available_methods = []
    for method in methods:
        if any(method in results.get(d, {}) for d in available_datasets):
            available_methods.append(method)

    # Print table header
    header = "Method"
    for dataset in available_datasets:
        header += f" & {dataset}"
    header += " \\\\"
    print(header)
    print("\\midrule")

    # Print each method's results
    for method in available_methods:
        row = method.replace("_", "\\_").replace(
            "&", "\\&"
        )  # Escape LaTeX special characters

        for dataset in available_datasets:
            if dataset in results and method in results[dataset]:
                stats = calculate_method_stats(dataset, method, stat_type)
                if "error" not in stats and "mean" in stats:
                    mean_val = stats["mean"]
                    std_val = stats.get("std", 0)

                    if std_val > 0:
                        # Format with larger mean and smaller std
                        cell = f" & \\textbf{{{mean_val:.2f}}} ± {{\\footnotesize {std_val:.2f}}}"
                    else:
                        # Only mean value available
                        cell = f" & \\textbf{{{mean_val:.2f}}}"
                else:
                    # No data or error
                    cell = " & -"
            else:
                # Method not found in this dataset
                cell = " & -"

            row += cell

        row += " \\\\"
        print(row)

    print("\\bottomrule")


def print_methods_comparison_table(datasets=None, methods=None, stat_type="final"):
    """
    Print a comparison table showing all methods' performance across selected datasets.

    Args:
        datasets (list): List of datasets to include (default: all datasets)
        methods (list): List of methods to include (default: all methods)
        stat_type (str): Type of statistic ("final", "all", or "task")
    """
    if datasets is None:
        datasets = DATASETS
    if methods is None:
        # Get all unique methods across all datasets
        all_methods = set()
        for dataset in datasets:
            if dataset in results:
                all_methods.update(results[dataset].keys())
        methods = sorted(list(all_methods))

    # Filter datasets and methods that actually exist
    available_datasets = [d for d in datasets if d in results]
    available_methods = [
        m for m in methods if any(m in results.get(d, {}) for d in available_datasets)
    ]

    print("=" * (20 + 15 * len(available_datasets)))
    print(f"METHODS COMPARISON TABLE ({stat_type.upper()} ACCURACIES)")
    print("=" * (20 + 15 * len(available_datasets)))

    # Print header
    header = f"{'Method':<18} |"
    for dataset in available_datasets:
        header += f" {dataset:>12} |"
    print(header)
    print("-" * (20 + 15 * len(available_datasets)))

    # Print data for each method
    for method in available_methods:
        row = f"{method:<18} |"
        for dataset in available_datasets:
            if dataset in results and method in results[dataset]:
                stats = calculate_method_stats(dataset, method, stat_type)
                if "error" not in stats and "mean" in stats:
                    if stats.get("std", 0) > 0:
                        row += f" {stats['mean']:5.1f}±{stats['std']:4.1f} |"
                    else:
                        row += f" {stats['mean']:11.1f} |"
                else:
                    row += f" {'---':>11} |"
            else:
                row += f" {'---':>11} |"
        print(row)

    print("=" * (20 + 15 * len(available_datasets)))


def print_top_methods_per_dataset(top_n=5, stat_type="final"):
    """
    Print the top N performing methods for each dataset.

    Args:
        top_n (int): Number of top methods to show per dataset
        stat_type (str): Type of statistic ("final", "all", or "task")
    """
    print("=" * 80)
    print(f"TOP {top_n} METHODS PER DATASET ({stat_type.upper()} ACCURACIES)")
    print("=" * 80)

    for dataset in DATASETS:
        if dataset not in results:
            continue

        print(f"\n{dataset} - Top {top_n} Methods:")
        print("-" * 50)

        # Get all methods with their performance
        method_performances = []
        for method_name in results[dataset]:
            stats = calculate_method_stats(dataset, method_name, stat_type)
            if "error" not in stats and "mean" in stats:
                method_performances.append(
                    (method_name, stats["mean"], stats.get("std", 0))
                )

        # Sort by mean performance (descending) and take top N
        method_performances.sort(key=lambda x: x[1], reverse=True)
        top_methods = method_performances[:top_n]

        # Print results
        for i, (method, mean_val, std_val) in enumerate(top_methods, 1):
            if std_val > 0:
                print(f"{i}. {method:<25} {mean_val:6.2f}% ± {std_val:5.2f}%")
            else:
                print(f"{i}. {method:<25} {mean_val:6.2f}%")

    print("\n" + "=" * 80)


# ============================================================================
# DATA
# ============================================================================

# results = {
#     "CIFAR100": {
#         "MM - CIL":             {"mean": 92.75},
#         "MM - CA":              {"mean": 94.00, "std": 0.39},
#         "MM - LCA":             {"mean": 94.8, "std": 0.34},
#         "MOS":                  {"mean": 94.27, "std": 0.31},
#         "MOS - CIL":            {},
#         "MOS - LCA":            {},
#         "SLCA":            {"mean": 93.70, "std": 0.33},
#         "EASE":                 {"mean": 91.69, "std": 0.33},
#         "APER + Adapter":       {"mean": 90.78, "std": 0.45},
#         "APER + Finetune":      {"mean": 81.65, "std": 0.93},
#         "APER + SSF":           {"mean": 89.45, "std": 0.97},
#         "APER + VPT-Deep":      {"mean": 88.96, "std": 0.91},
#         "APER + VPT-Shallow":   {"mean": 88.10, "std": 0.93},
#         "L2P":                  {"mean": 87.75, "std": 1.41},
#         "CODA-Prompt":          {"mean": 91.01, "std": 0.21},
#         "DualPrompt":           {"mean": 86.73, "std": 0.62},
#     },
#     "IN-R": {
#         "MM - CIL":             {"mean": 85.06},
#         "MM - CA":              {"mean": 85.20, "std": 0.22},
#         "MM - LCA":             {"mean": 85.83, "std": 0.25},
#         "MOS - CIL":            {}, # running - wan
#         "MOS":             {"mean": 83.30, "std": 0.61},
#         "MOS - LCA":            {}, # running - wan
#         "SLCA":            {"mean": 81.17, "std": 0.28},
#         "EASE":                 {"mean": 82.41, "std": 0.46},
#         "APER + Adapter":       {"mean": 78.80, "std": 0.58},
#         "APER + Finetune":      {"mean": 72.06, "std": 0.80},
#         "APER + SSF":           {"mean": 78.07, "std": 1.09},
#         "APER + VPT-Deep":      {"mean": 78.77, "std": 0.73},
#         "APER + VPT-Shallow":   {"mean": 67.29, "std": 3.49},
#         "L2P":                  {"mean": 77.27, "std": 0.61},
#         "CODA-Prompt":          {"mean": 78.22, "std": 0.36},
#         "DualPrompt":           {"mean": 74.61, "std": 0.48},
#     },
#     "IN-A": {
#         "MM - CIL":             {"mean": 66.52},
#         "MM - CA":              {"mean": 72.25, "std": 1.09},
#         "MM - LCA":             {"mean": 75.21, "std": 0.5},
#         "MOS - CIL":            {}, # running - wan
#         "MOS":             {"mean": 67.68, "std": 1.93},
#         "MOS - LCA":            {"mean": 70.71, "std": 1.51},
#         "SLCA - CIL":           {"mean": 63.46, "std": 0.64},
#         "SLCA":            {"mean": 45.15, "std": 19.78},
#         "SLCA - LCA":           {"mean": 68.86, "std": 0.54},
#         "EASE":                 {"mean": 67.77, "std": 1.84},
#         "APER + Adapter":       {"mean": 58.86, "std": 1.32},
#         "APER + Finetune":      {"mean": 58.69, "std": 3.69},
#         "APER + SSF":           {"mean": 61.58, "std": 0.45},
#         "APER + VPT-Deep":      {"mean": 56.96, "std": 0.37},
#         "APER + VPT-Shallow":   {"mean": 56.91, "std": 1.39},
#         "L2P":                  {"mean": 52.61, "std": 1.66},
#         "CODA-Prompt":          {"mean": 48.13, "std": 0.85},
#         "DualPrompt":           {"mean": 55.33, "std": 1.46},
#     },
#     "CUB": {
#         "MM - CIL":             {"mean": 87.16},
#         "MM - CA":              {"mean": 89.79, "std": 0.28},
#         "MM - LCA":             {"mean": 90.8, "std": 0.31},
#         "MOS - CIL":            {}, # running - wan
#         "MOS":             {"mean": 92.31, "std": 0.68},
#         "MOS - LCA":            {}, # running - wan
#         "SLCA - CA":            {"mean": 90.21, "std": 0.90},
#         "EASE":                 {"mean": 89.49, "std": 1.23},
#         "APER + Adapter":       {"mean": 89.74, "std": 1.29},
#         "APER + Finetune":      {"mean": 89.54, "std": 1.39},
#         "APER + SSF":           {"mean": 89.61, "std": 1.15},
#         "APER + VPT-Deep":      {"mean": 88.96, "std": 1.16},
#         "APER + VPT-Shallow":   {"mean": 89.52, "std": 1.44},
#         "L2P":                  {"mean": 75.82, "std": 1.77},
#         "CODA-Prompt":          {"mean": 75.65, "std": 1.24},
#         "DualPrompt":           {"mean": 78.86, "std": 0.97},
#     },
#     "OB": {
#         "MM - CIL":             {"mean": 80.20},
#         "MM - CA":              {"mean": 81.24, "std": 0.56},
#         "MM - LCA":             {"mean": 81.53, "std": 0.46},
#         "MOS - CIL":            {}, # running - wan
#         "MOS":             {"mean": 86.10, "std": 0.72},
#         "MOS - LCA":            {}, # running - wan
#         "SLCA - CIL":           {}, # running - wan
#         "SLCA":            {"mean": 82.67, "std": 0.61},
#         "SLCA - LCA":           {}, # running - wan
#         "EASE":                 {"mean": 80.78, "std": 0.15},
#         "APER + Adapter":       {"mean": 80.28, "std": 0.44},
#         "APER + Finetune":      {"mean": 77.83, "std": 1.19},
#         "APER + SSF":           {"mean": 80.31, "std": 0.56},
#         "APER + VPT-Deep":      {"mean": 79.76, "std": 0.38},
#         "APER + VPT-Shallow":   {"mean": 79.65, "std": 0.95},
#         "L2P":                  {"mean": 73.83, "std": 1.23},
#         "CODA-Prompt":          {"mean": 70.97, "std": 0.10},
#         "DualPrompt":           {"mean": 74.41, "std": 1.17}
#     },
#     "VTAB": {
#         "MM - CIL":             {"mean": 86.53},
#         "MM - CA":              {"mean": 94.27, "std": 1.19},
#         "MM - LCA":             {"mean": 94.32, "std": 1.10},
#         "MOS - CIL":            {}, # running - wan
#         "MOS":             {"mean": 92.56, "std": 0.61},
#         "MOS - LCA":            {}, # running - wan
#         "SLCA":            {"mean": 91.08, "std": 3.38},
#         "EASE":                 {"mean": 93.28, "std": 0.12},
#         "APER + Adapter":       {"mean": 90.66, "std": 0.57},
#         "APER + Finetune":      {"mean": 91.82, "std": 1.44},
#         "APER + SSF":           {"mean": 91.84, "std": 1.51},
#         "APER + VPT-Deep":      {"mean": 91.91, "std": 1.41},
#         "APER + VPT-Shallow":   {"mean": 91.55, "std": 0.76},
#         "L2P":                  {"mean": 82.37, "std": 2.85},
#         "CODA-Prompt":          {"mean": 65.58, "std": 2.63},
#         "DualPrompt":           {"mean": 83.99, "std": 5.89}
#     },
#     "CARS": {
#         "MM - CIL":             {"mean": 69.55},
#         "MM - CA":              {"mean": 75.71, "std": 1.39},
#         "MM - LCA":             {"mean": 76.16, "std": 1.41},
#         "MOS":             {"mean": 71.43, "std": 19.54},
#         "MOS - CIL":            {}, # running - wan
#         "MOS - LCA":            {}, # running - wan
#         "SLCA":            {"mean": 74.58, "std": 2.16},
#         "EASE":                 {"mean": 48.08, "std": 1.19},
#         "APER + Adapter":       {"mean": 50.60, "std": 1.08},
#         "APER + Finetune":      {"mean": 53.22, "std": 1.39},
#         "APER + SSF":           {"mean": 51.30, "std": 1.09},
#         "APER + VPT-Deep":      {"mean": 50.64, "std": 3.01},
#         "APER + VPT-Shallow":   {"mean": 50.87, "std": 0.80},
#         "L2P":                  {"mean": 53.41, "std": 1.16},
#         "CODA-Prompt":          {"mean": 26.29, "std": 0.56},
#         "DualPrompt":           {"mean": 49.40, "std": 2.10}
#     }
# }

STD = {
    "MM - LCA": {
        "CIFAR100": 0.34,
        "IN-R": 0.25,
        "IN-A": 0.50,
        "CUB": 0.31,
        "OB": 0.46,
        "VTAB": 1.10,
        "CARS": 1.41,
    }
}

results = {
    "CIFAR100": {
        "MM - CIL": [
            [98.8, 97.82, 97.03, 96.41, 95.67, 95.01, 94.51, 93.83, 93.25, 92.75],
            [97.7, 97.45, 97.09, 96.44, 95.84, 95.33, 94.76, 94.1, 93.46, 92.94],
            [99.1, 97.65, 96.76, 95.91, 95.16, 94.62, 94.11, 93.62, 93.17, 92.76],
        ],
        "MM - CA": [
            [98.8, 98.1, 97.71, 97.23, 96.74, 96.28, 95.81, 95.27, 94.8, 94.4],
            [97.7, 97.48, 97.32, 96.93, 96.48, 96.14, 95.62, 95.08, 94.56, 94.14],
            [99.1, 97.98, 97.22, 96.63, 96.03, 95.41, 94.89, 94.43, 93.92, 93.47],
        ],
        "MM - LCA": [
            [98.8, 98.12, 97.82, 97.38, 96.97, 96.51, 96.12, 95.61, 95.17, 94.8]
        ],
        "MOS": [
            [98.7, 97.95, 97.44, 96.98, 96.49, 95.99, 95.63, 95.2, 94.82, 94.48],
            [97.5, 97.48, 97.34, 96.92, 96.47, 96.07, 95.63, 95.22, 94.79, 94.45],
            [99.2, 98.12, 97.11, 96.41, 95.79, 95.32, 94.94, 94.58, 94.25, 93.93],
        ],
        "MOS - CIL": [
            [98.7, 97.3, 96.49, 95.7, 94.83, 94.08, 93.51, 92.85, 92.27, 91.77],
            [97.4, 97.05, 96.71, 96.09, 95.45, 94.93, 94.26, 93.58, 92.89, 92.31],
            [98.9, 97.62, 96.56, 95.6, 94.68, 93.93, 93.3, 92.68, 92.07, 91.53],
        ],
        "MOS - CA": [
            [98.7, 97.43, 96.58, 95.91, 95.28, 94.64, 94.21, 93.58, 93.05, 92.6],
            [97.4, 96.58, 96.11, 95.51, 94.98, 94.55, 94.09, 93.59, 93.09, 92.7],
            [98.9, 97.3, 96.12, 95.24, 94.45, 93.82, 93.3, 92.84, 92.4, 92.01],
        ],
        "MOS - LCA": [
            [98.7, 97.38, 96.72, 96.13, 95.57, 95.01, 94.57, 93.98, 93.48, 93.05],
            [97.4, 96.48, 96.09, 95.75, 95.34, 94.94, 94.53, 94.07, 93.65, 93.28],
            [98.9, 97.45, 96.43, 95.74, 95.01, 94.52, 94.06, 93.66, 93.25, 92.85],
        ],
        "SLCA": [
            [99.0, 98.08, 97.53, 97.02, 96.37, 95.86, 95.44, 94.87, 94.35, 93.93],
            [97.8, 97.62, 97.45, 96.9, 96.42, 95.97, 95.34, 94.79, 94.29, 93.85],
            [98.6, 97.9, 96.99, 96.32, 95.64, 95.07, 94.62, 94.18, 93.7, 93.32],
        ],
        "SLCA - CIL": [
            [99.0, 97.9, 97.28, 96.65, 95.88, 95.3, 94.77, 94.14, 93.63, 90.15],
            [97.8, 97.42, 97.14, 96.59, 96.06, 95.58, 95.0, 94.43, 93.82, 90.34],
            [98.6, 97.52, 96.67, 95.36, 94.53, 93.86, 93.23, 92.65, 92.16, 90.69],
        ],
        "SLCA - CA": [
            [99.0, 97.6, 96.57, 95.61, 94.91, 94.31, 93.69, 93.04, 92.47, 91.97],
            [97.8, 96.8, 96.09, 94.8, 93.89, 93.26, 92.31, 91.67, 91.2, 90.84],
            [98.6, 96.57, 95.48, 94.62, 93.81, 93.0, 92.29, 91.85, 91.23, 90.78],
        ],
        "SLCA - LCA": [
            [99.0, 97.55, 96.74, 96.15, 95.59, 94.98, 94.51, 93.95, 93.35, 92.79],
            [97.8, 97.1, 96.33, 95.34, 94.76, 94.16, 93.35, 92.76, 92.21, 91.8],
            [98.6, 97.3, 96.38, 95.51, 94.72, 93.84, 93.19, 92.62, 91.99, 91.41],
        ],
        "EASE": [
            [98.7, 97.62, 96.67, 95.96, 95.1, 94.42, 93.87, 93.18, 92.58, 92.07],
            [96.8, 96.28, 95.88, 95.23, 94.64, 94.1, 93.34, 92.64, 92.01, 91.49],
            [98.3, 97.12, 96.34, 95.27, 94.29, 93.54, 92.99, 92.5, 91.98, 91.52],
        ],
        "APER + Adapter": [
            [98.2, 97.0, 96.09, 95.16, 94.31, 93.5, 92.85, 92.12, 91.47, 90.9],
            [95.4, 95.25, 95.06, 94.64, 94.08, 93.51, 92.9, 92.28, 91.68, 91.16],
            [97.9, 96.08, 94.84, 93.96, 93.15, 92.46, 91.84, 91.29, 90.77, 90.27],
        ],
        "APER + Finetune": [
            [93.7, 90.95, 89.52, 88.15, 87.01, 85.93, 85.13, 84.18, 83.37, 82.64],
            [86.8, 87.35, 87.17, 86.41, 85.47, 84.64, 83.85, 83.0, 82.23, 81.53],
            [93.3, 90.1, 88.07, 86.32, 84.88, 83.82, 82.93, 82.16, 81.43, 80.78],
        ],
        "APER + SSF": [
            [98.8, 97.02, 95.73, 94.51, 93.41, 92.46, 91.73, 90.92, 90.2, 89.57],
            [97.0, 96.15, 95.57, 94.71, 93.82, 93.08, 92.37, 91.66, 90.97, 90.36],
            [98.9, 96.4, 94.61, 93.3, 92.13, 91.19, 90.39, 89.68, 89.02, 88.42],
        ],
        "APER + VPT-Deep": [
            [98.8, 96.82, 95.48, 94.31, 93.23, 92.23, 91.49, 90.66, 89.93, 89.27],
            [97.3, 96.32, 95.46, 94.37, 93.36, 92.54, 91.76, 90.99, 90.29, 89.67],
            [98.8, 96.4, 94.66, 93.18, 91.9, 90.88, 90.0, 89.22, 88.54, 87.93],
        ],
        "APER + VPT-Shallow": [
            [96.4, 94.82, 93.84, 92.85, 91.89, 91.01, 90.35, 89.55, 88.85, 88.24],
            [91.1, 91.48, 91.38, 90.86, 90.2, 89.58, 88.98, 88.31, 87.68, 87.12],
            [96.6, 94.82, 93.65, 92.72, 91.85, 91.17, 90.57, 90.0, 89.46, 88.95],
        ],
        "L2P": [
            [97.4, 96.15, 95.23, 93.98, 92.86, 91.96, 91.21, 90.46, 89.82, 89.28],
            [96.1, 94.98, 93.38, 92.14, 91.18, 90.34, 89.47, 88.67, 88.0, 87.46],
            [96.9, 94.05, 92.6, 91.04, 89.87, 89.09, 88.38, 87.74, 87.09, 86.49],
        ],
        "CODA-Prompt": [
            [99.0, 97.1, 95.94, 94.93, 94.07, 93.13, 92.38, 91.8, 91.34, 90.88],
            [99.3, 97.7, 96.58, 95.49, 94.61, 93.65, 92.88, 92.22, 91.75, 91.26],
            [98.7, 96.75, 95.69, 94.79, 94.01, 93.14, 92.36, 91.77, 91.33, 90.9],
        ],
        "DualPrompt": [
            [96.4, 94.95, 93.76, 92.59, 91.35, 90.32, 89.56, 88.69, 87.92, 87.35],
            [94.1, 93.57, 92.58, 91.73, 90.69, 89.96, 89.11, 88.25, 87.44, 86.73],
            [96.5, 93.85, 92.12, 90.54, 89.2, 88.27, 87.66, 87.15, 86.62, 86.11],
        ],
    },
    "IN-R": {
        "MM - CIL": [
            [94.63, 92.47, 91.04, 89.76, 88.48, 87.55, 86.82, 86.18, 85.61, 85.06],
            [93.84, 91.76, 90.31, 88.45, 87.11, 86.07, 85.19, 84.43, 83.75, 83.19],
            [94.0, 91.98, 90.37, 89.2, 88.25, 87.46, 86.56, 85.83, 85.17, 84.58],
        ],
        "MM - CA": [
            [94.63, 92.46, 91.2, 90.22, 88.88, 87.9, 87.16, 86.58, 86.03, 85.47],
            [93.84, 92.1, 90.76, 89.24, 88.25, 87.39, 86.76, 86.12, 85.49, 84.94],
            [94.0, 92.19, 90.63, 89.3, 88.33, 87.58, 86.86, 86.27, 85.66, 85.19],
        ],
        "MM - LCA": [
            [94.63, 92.57, 91.28, 90.28, 89.05, 88.17, 87.42, 86.85, 86.32, 85.83]
        ],
        "MOS": [
            [90.57, 89.04, 87.87, 86.82, 85.83, 85.04, 84.37, 83.81, 83.29, 82.81],
            [92.67, 90.76, 89.42, 88.19, 87.45, 86.6, 85.87, 85.2, 84.53, 83.99],
            [91.73, 89.32, 88.13, 87.04, 86.39, 85.7, 84.93, 84.35, 83.75, 83.24],
        ],
        "MOS - CIL": [
            [90.57, 88.4, 86.38, 84.0, 81.12, 77.75, 74.56, 71.31, 68.26, 65.33],
            [92.67, 89.75, 86.99, 84.07, 81.08, 77.92, 74.76, 71.62, 68.45, 65.61],
            [91.73, 88.78, 86.38, 83.71, 81.13, 78.08, 74.35, 70.76, 67.47, 64.54],
        ],
        "MOS - CA": [
            [90.57, 88.17, 86.66, 85.44, 84.2, 83.2, 82.49, 81.88, 81.28, 80.74],
            [92.67, 90.33, 88.52, 87.1, 86.04, 85.04, 84.15, 83.36, 82.62, 82.03],
            [91.73, 89.01, 87.01, 85.78, 84.83, 84.01, 83.19, 82.55, 81.89, 81.33],
        ],
        "MOS - LCA": [
            [90.57, 88.78, 87.36, 86.11, 84.92, 83.86, 83.04, 82.39, 81.76, 81.17],
            [92.67, 90.48, 88.71, 87.24, 86.18, 85.39, 84.65, 83.94, 83.27, 82.69],
            [91.73, 88.9, 87.48, 86.21, 85.36, 84.52, 83.71, 83.09, 82.44, 81.89],
        ],
        "SLCA": [
            [93.9, 91.99, 90.59, 89.27, 88.17, 87.27, 86.63, 86.04, 85.55, 85.04],
            [93.7, 92.24, 91.15, 89.84, 88.79, 87.93, 87.21, 86.6, 85.97, 85.46],
            [91.9, 90.58, 89.45, 88.55, 87.79, 87.19, 86.56, 86.0, 85.4, 84.92],
        ],
        "SLCA - CIL": [
            [93.9, 91.08, 89.59, 88.31, 87.09, 86.18, 85.45, 84.87, 84.32, 81.75],
            [93.7, 91.66, 90.25, 88.8, 87.68, 86.74, 85.88, 85.1, 84.4, 81.84],
            [91.9, 90.2, 88.99, 87.66, 86.69, 85.9, 85.15, 84.53, 83.91, 81.35],
        ],
        "SLCA - CA": [
            [93.9, 90.86, 88.96, 87.46, 86.1, 85.18, 84.44, 83.86, 83.19, 82.53],
            [93.7, 91.2, 89.57, 88.1, 86.75, 85.63, 84.67, 83.78, 83.19, 82.59],
            [91.9, 89.4, 87.56, 86.3, 85.34, 84.53, 83.81, 83.18, 82.51, 81.85],
        ],
        "SLCA - LCA": [
            [93.9, 91.2, 89.53, 88.06, 86.81, 86.04, 85.36, 84.76, 84.17, 83.61],
            [93.7, 91.82, 90.2, 88.81, 87.78, 86.82, 86.0, 85.3, 84.66, 84.15],
            [91.9, 89.55, 88.07, 87.05, 86.33, 85.6, 84.82, 84.21, 83.56, 83.02],
        ],
        "EASE": [
            [93.32, 90.72, 89.09, 87.7, 86.49, 85.52, 84.7, 83.99, 83.35, 82.75],
            [93.11, 90.59, 89.03, 87.7, 86.59, 85.61, 84.73, 83.96, 83.24, 82.6],
            [91.73, 89.77, 88.2, 86.81, 85.73, 84.86, 83.95, 83.2, 82.51, 81.89],
        ],
        "APER + Adapter": [
            [92.02, 88.48, 86.46, 84.94, 83.51, 82.32, 81.34, 80.48, 79.7, 79.01],
            [92.82, 89.78, 87.49, 85.7, 84.29, 83.03, 81.96, 80.97, 80.03, 79.24],
            [92.38, 88.92, 86.44, 84.31, 82.83, 81.7, 80.59, 79.69, 78.81, 78.14],
        ],
        "APER + Finetune": [
            [94.05, 88.09, 84.52, 81.82, 79.62, 77.78, 76.29, 74.98, 73.87, 72.85],
            [93.11, 87.65, 83.61, 80.84, 78.84, 77.05, 75.57, 74.28, 73.08, 72.07],
            [90.6, 84.68, 81.31, 78.62, 76.82, 75.4, 74.06, 73.02, 72.03, 71.26],
        ],
        "APER + SSF": [
            [93.47, 89.58, 87.34, 85.56, 83.89, 82.56, 81.47, 80.54, 79.72, 78.97],
            [92.67, 89.6, 87.15, 85.21, 83.71, 82.32, 81.13, 80.13, 79.17, 78.38],
            [92.71, 88.55, 85.75, 83.5, 81.9, 80.63, 79.45, 78.49, 77.57, 76.86],
        ],
        "APER + VPT-Deep": [
            [91.44, 88.6, 86.69, 85.12, 83.72, 82.54, 81.58, 80.77, 80.02, 79.3],
            [91.94, 89.04, 86.95, 85.18, 83.84, 82.66, 81.61, 80.69, 79.82, 79.06],
            [92.06, 88.6, 86.2, 84.12, 82.64, 81.51, 80.41, 79.51, 78.61, 77.94],
        ],
        "APER + VPT-Shallow": [
            [83.45, 79.96, 77.8, 76.34, 74.95, 73.75, 72.8, 71.99, 71.28, 70.6],
            [74.63, 72.6, 70.77, 69.39, 68.18, 66.99, 65.99, 65.13, 64.34, 63.64],
            [81.36, 77.2, 75.02, 73.07, 71.73, 70.72, 69.77, 68.97, 68.21, 67.63],
        ],
        "L2P": [
            [88.39, 85.6, 83.83, 82.56, 81.48, 80.56, 79.85, 79.18, 78.58, 77.96],
            [88.86, 86.02, 83.66, 82.12, 80.91, 79.84, 78.93, 78.11, 77.37, 76.8],
            [88.01, 84.86, 82.94, 81.61, 80.58, 79.72, 78.88, 78.27, 77.61, 77.06],
        ],
        "CODA-Prompt": [
            [86.01, 84.95, 83.92, 82.81, 81.83, 80.88, 79.95, 79.11, 78.45, 77.91],
            [86.64, 85.82, 84.5, 83.4, 82.35, 81.49, 80.6, 79.82, 79.2, 78.61],
            [85.38, 84.82, 83.72, 82.69, 81.82, 80.97, 80.06, 79.3, 78.72, 78.13],
        ],
        "DualPrompt": [
            [85.49, 83.1, 81.5, 80.13, 78.77, 77.77, 76.98, 76.31, 75.7, 75.04],
            [87.83, 84.42, 82.14, 80.33, 79.13, 77.85, 76.87, 76.02, 75.28, 74.7],
            [85.41, 82.96, 81.46, 79.85, 78.61, 77.52, 76.45, 75.62, 74.77, 74.08],
        ],
    },
    "IN-A": {
        "MM - CIL": [
            [87.43, 84.56, 81.13, 78.0, 75.16, 72.74, 70.74, 69.16, 67.73, 66.52],
            [82.14, 78.91, 77.04, 75.23, 73.71, 72.48, 71.33, 70.2, 68.94, 67.61],
            [85.29, 79.37, 75.84, 73.48, 71.49, 70.13, 68.77, 67.49, 66.42, 65.44],
        ],
        "MM - CA": [
            [87.43, 86.06, 84.14, 81.79, 80.06, 78.4, 77.07, 75.89, 74.79, 73.72],
            [82.14, 80.32, 78.95, 77.94, 76.74, 75.58, 74.63, 73.67, 72.74, 71.92],
            [85.29, 81.34, 78.83, 77.13, 75.53, 74.41, 73.43, 72.68, 71.94, 71.11],
        ],
        "MM - LCA": [
            [87.43, 86.05, 84.46, 82.73, 80.97, 79.62, 78.41, 77.2, 76.05, 74.99]
        ],
        "MOS": [
            [80.0, 80.14, 78.57, 77.26, 75.51, 74.22, 72.96, 71.72, 70.46, 69.39],
            [75.0, 73.78, 72.76, 71.63, 70.52, 69.33, 68.3, 67.4, 66.4, 65.47],
            [77.65, 75.68, 73.93, 72.56, 71.67, 71.13, 70.31, 69.6, 68.82, 67.95],
        ],
        "MOS - CIL": [
            [77.71, 75.1, 72.97, 70.31, 67.9, 65.91, 64.21, 62.52, 60.82, 59.3],
            [75.0, 71.34, 69.06, 66.95, 65.3, 63.73, 62.46, 61.18, 59.75, 58.38],
            [77.65, 73.28, 70.6, 68.56, 67.2, 66.34, 65.36, 64.21, 63.0, 61.66],
        ],
        "MOS - CA": [
            [80.0, 79.59, 78.27, 76.57, 74.65, 73.09, 71.78, 70.58, 69.39, 68.27],
            [75.0, 75.1, 73.86, 72.54, 71.11, 69.61, 68.39, 67.36, 66.33, 65.45],
            [77.65, 77.44, 75.26, 73.63, 72.36, 71.68, 70.83, 69.96, 69.06, 68.18],
        ],
        "MOS - LCA": [
            [80.0, 81.11, 79.56, 77.54, 75.59, 74.12, 72.85, 71.7, 70.65, 69.6],
            [75.0, 75.47, 74.11, 73.21, 71.96, 70.7, 69.63, 68.64, 67.62, 66.78],
            [77.65, 76.96, 75.16, 73.99, 72.99, 72.32, 71.51, 70.78, 69.98, 69.12],
        ],
        "SLCA": [
            [83.43, 80.74, 79.25, 77.85, 62.3, 51.94, 44.53, 38.97, 34.65, 31.19],
            [78.57, 76.69, 74.78, 73.75, 72.83, 71.48, 70.54, 69.8, 68.72, 67.79],
            [81.76, 77.42, 74.86, 73.17, 72.44, 60.45, 51.88, 45.44, 40.44, 36.42],
        ],
        "SLCA - CIL": [
            [83.43, 79.08, 75.41, 72.53, 70.58, 68.82, 67.4, 66.1, 64.98, 63.96],
            [78.57, 74.62, 73.55, 72.34, 70.44, 68.79, 67.68, 66.42, 65.05, 63.68],
            [81.76, 75.34, 72.19, 69.68, 68.34, 67.2, 66.03, 64.67, 63.57, 62.74],
        ],
        "SLCA - CA": [
            [83.43, 79.91, 77.64, 75.57, 73.41, 71.66, 69.83, 68.58, 67.36, 66.36],
            [78.57, 77.06, 74.51, 73.43, 71.98, 70.37, 68.89, 67.94, 66.71, 65.53],
            [81.76, 78.38, 74.3, 72.45, 71.08, 69.73, 68.51, 67.55, 66.36, 65.43],
        ],
        "SLCA - LCA": [
            [83.43, 81.02, 78.66, 77.09, 75.44, 73.65, 71.74, 70.21, 68.97, 67.97],
            [78.57, 76.32, 74.82, 73.9, 72.93, 71.49, 70.29, 69.39, 68.21, 67.08],
            [81.76, 77.9, 75.11, 73.23, 72.13, 71.13, 70.04, 69.09, 67.89, 67.0],
        ],
        "EASE": [
            [85.71, 82.85, 80.31, 78.35, 76.5, 75.05, 73.5, 72.05, 70.64, 69.44],
            [86.43, 80.24, 77.44, 75.59, 74.28, 72.85, 71.66, 70.46, 69.19, 68.06],
            [84.71, 79.54, 75.22, 72.63, 70.73, 69.48, 68.52, 67.57, 66.68, 65.8],
        ],
        "APER + Adapter": [
            [76.57, 73.7, 70.84, 68.57, 66.51, 64.93, 63.39, 61.96, 60.61, 59.48],
            [83.57, 78.44, 73.88, 70.4, 67.78, 65.68, 63.84, 62.26, 60.91, 59.74],
            [75.29, 69.86, 65.98, 63.44, 61.73, 60.75, 59.83, 59.05, 58.24, 57.34],
        ],
        "APER + Finetune": [
            [77.71, 72.6, 68.43, 65.4, 62.76, 60.84, 59.09, 57.51, 56.07, 54.82],
            [81.43, 76.62, 72.44, 69.48, 67.07, 65.06, 63.17, 61.61, 60.25, 59.08],
            [81.18, 74.72, 70.89, 68.48, 66.91, 65.8, 64.83, 63.92, 63.06, 62.17],
        ],
        "APER + SSF": [
            [83.43, 78.1, 74.06, 71.27, 69.19, 67.47, 65.95, 64.53, 63.2, 62.09],
            [83.57, 77.31, 73.71, 71.33, 69.01, 67.04, 65.24, 63.77, 62.42, 61.22],
            [79.41, 74.32, 70.84, 68.23, 66.59, 65.39, 64.32, 63.38, 62.41, 61.43],
        ],
        "APER + VPT-Deep": [
            [73.14, 71.02, 68.21, 66.27, 64.27, 62.74, 61.17, 59.61, 58.16, 56.93],
            [80.71, 74.56, 70.55, 67.34, 64.71, 62.66, 60.84, 59.29, 57.86, 56.61],
            [75.88, 70.47, 66.54, 63.98, 62.21, 61.08, 60.05, 59.17, 58.29, 57.34],
        ],
        "APER + VPT-Shallow": [
            [74.29, 71.45, 68.64, 66.24, 64.17, 62.67, 61.22, 59.85, 58.58, 57.51],
            [81.43, 76.43, 72.31, 68.9, 66.19, 64.12, 62.29, 60.65, 59.18, 57.89],
            [75.88, 69.67, 65.56, 62.91, 60.98, 59.65, 58.44, 57.35, 56.36, 55.32],
        ],
        "L2P": [
            [73.14, 66.57, 63.99, 61.32, 59.14, 57.5, 56.11, 54.83, 53.69, 52.67],
            [70.71, 67.68, 65.01, 62.82, 60.68, 58.97, 57.53, 56.4, 55.24, 54.24],
            [72.94, 65.47, 61.78, 58.84, 57.09, 55.68, 54.32, 53.03, 51.94, 50.92],
        ],
        "CODA-Prompt": [
            [72.31, 66.64, 62.01, 58.68, 56.2, 54.05, 52.18, 50.51, 48.97, 47.57],
            [72.64, 67.24, 62.74, 59.14, 56.51, 54.31, 52.4, 50.72, 49.14, 47.71],
            [71.34, 66.87, 62.93, 60.11, 57.81, 55.77, 53.91, 52.2, 50.61, 49.11],
        ],
        "DualPrompt": [
            [73.14, 71.16, 68.31, 65.84, 63.64, 61.9, 60.39, 58.97, 57.7, 56.56],
            [78.57, 72.56, 69.06, 66.02, 63.5, 61.45, 59.7, 58.21, 56.91, 55.72],
            [70.59, 64.94, 62.56, 60.36, 58.68, 57.57, 56.48, 55.52, 54.6, 53.71],
        ],
    },
    "CUB": {
        "MM - CIL": [
            [98.38, 97.29, 95.63, 94.21, 92.71, 90.96, 89.69, 88.76, 87.93, 87.16],
            [97.93, 97.44, 96.5, 95.11, 93.07, 91.6, 90.14, 88.94, 88.01, 87.14],
            [94.37, 93.59, 91.61, 91.01, 90.37, 89.24, 88.25, 87.23, 86.35, 85.72],
        ],
        "MM - CA": [
            [98.38, 97.72, 96.47, 95.35, 94.39, 93.37, 92.57, 91.72, 90.94, 90.18],
            [97.93, 97.34, 96.5, 95.53, 94.26, 93.38, 92.37, 91.34, 90.47, 89.66],
            [94.37, 94.21, 93.2, 92.81, 92.41, 91.73, 91.18, 90.56, 90.04, 89.53],
        ],
        "MM - LCA": [
            [98.38, 98.04, 96.56, 95.57, 94.66, 93.79, 93.03, 92.23, 91.49, 90.8]
        ],
        "MOS": [
            [97.98, 97.45, 96.27, 95.46, 94.77, 94.2, 93.66, 93.23, 92.86, 92.48],
            [98.34, 97.94, 96.89, 95.98, 95.26, 94.8, 94.24, 93.68, 93.25, 92.86],
            [94.84, 94.82, 94.18, 93.96, 93.54, 93.06, 92.69, 92.26, 91.93, 91.66],
        ],
        "MOS - CIL": [
            [97.98, 96.02, 94.42, 92.52, 90.82, 88.8, 87.23, 85.85, 84.52, 83.3],
            [98.34, 95.5, 93.09, 90.82, 88.89, 87.33, 85.73, 84.33, 83.22, 82.17],
            [94.84, 92.22, 89.61, 88.14, 87.18, 85.82, 84.53, 83.21, 82.02, 81.06],
        ],
        "MOS - CA": [
            [97.98, 97.01, 95.68, 94.58, 93.66, 92.75, 91.84, 91.02, 90.34, 89.72],
            [98.34, 96.82, 95.58, 94.2, 93.07, 92.33, 91.58, 90.84, 90.25, 89.74],
            [94.84, 92.0, 91.35, 91.01, 90.57, 89.9, 89.31, 88.8, 88.4, 88.07],
        ],
        "MOS - LCA": [
            [97.98, 97.56, 96.24, 95.08, 94.25, 93.32, 92.52, 91.83, 91.22, 90.71],
            [98.34, 97.54, 96.29, 95.22, 94.33, 93.69, 92.85, 92.16, 91.7, 91.25],
            [94.84, 92.65, 91.74, 91.6, 91.34, 90.61, 90.14, 89.74, 89.4, 89.16],
        ],
        "SLCA": [
            [99.19, 98.5, 97.16, 95.75, 94.83, 93.88, 93.03, 92.43, 91.78, 91.23],
            [97.93, 97.84, 96.77, 95.34, 94.02, 92.91, 91.89, 91.04, 90.38, 89.87],
            [93.43, 93.78, 93.07, 92.64, 92.17, 91.46, 91.01, 90.45, 89.9, 89.52],
        ],
        "SLCA - CIL": [
            [99.19, 97.28, 95.46, 93.58, 91.85, 90.14, 88.87, 87.82, 86.97, 85.99],
            [97.93, 95.8, 93.95, 91.76, 89.54, 87.9, 86.44, 85.25, 84.37, 83.51],
            [93.43, 89.88, 88.19, 87.62, 86.69, 85.3, 84.28, 83.34, 82.48, 81.86],
        ],
        "SLCA - CA": [
            [99.19, 97.94, 96.4, 94.96, 93.79, 92.6, 91.54, 90.7, 89.94, 89.28],
            [97.93, 97.13, 96.02, 94.0, 92.25, 91.24, 90.23, 89.4, 88.6, 87.85],
            [93.43, 92.49, 91.68, 91.04, 90.59, 89.64, 88.92, 88.15, 87.38, 86.87],
        ],
        "SLCA - LCA": [
            [99.19, 98.06, 96.82, 95.58, 94.59, 93.53, 92.55, 91.79, 91.11, 90.42],
            [97.93, 97.33, 96.24, 94.46, 92.88, 91.73, 90.69, 89.92, 89.19, 88.49],
            [93.43, 93.02, 92.13, 91.7, 91.28, 90.47, 89.87, 89.16, 88.48, 88.06],
        ],
        "EASE": [
            [97.98, 97.12, 95.9, 94.94, 94.2, 93.3, 92.41, 91.67, 91.1, 90.52],
            [97.93, 97.23, 95.75, 94.39, 93.33, 92.64, 91.74, 90.96, 90.37, 89.82],
            [92.49, 91.48, 90.91, 90.76, 90.46, 89.81, 89.31, 88.88, 88.48, 88.12],
        ],
        "APER + Adapter": [
            [97.98, 97.12, 95.9, 94.94, 94.22, 93.36, 92.55, 91.88, 91.3, 90.74],
            [97.51, 97.12, 95.82, 94.52, 93.55, 92.86, 92.02, 91.27, 90.72, 90.19],
            [91.55, 91.11, 90.72, 90.64, 90.4, 89.79, 89.35, 88.94, 88.58, 88.28],
        ],
        "APER + Finetune": [
            [98.79, 97.86, 96.44, 95.4, 94.55, 93.57, 92.7, 92.01, 91.38, 90.8],
            [97.51, 97.02, 95.8, 94.37, 93.29, 92.5, 91.61, 90.86, 90.33, 89.8],
            [91.08, 90.66, 90.32, 90.24, 90.02, 89.37, 89.01, 88.66, 88.33, 88.04],
        ],
        "APER + SSF": [
            [97.57, 96.8, 95.54, 94.53, 93.79, 93.0, 92.23, 91.62, 91.08, 90.54],
            [97.51, 97.32, 95.96, 94.48, 93.37, 92.65, 91.79, 91.03, 90.47, 89.97],
            [93.43, 92.05, 91.25, 90.96, 90.62, 89.96, 89.5, 89.05, 88.66, 88.33],
        ],
        "APER + VPT-Deep": [
            [96.36, 95.65, 94.17, 93.34, 92.7, 91.87, 91.06, 90.41, 89.86, 89.28],
            [97.51, 96.92, 95.69, 94.34, 93.29, 92.58, 91.73, 90.96, 90.42, 89.93],
            [92.49, 91.48, 90.77, 90.54, 90.27, 89.54, 89.01, 88.49, 88.04, 87.68],
        ],
        "APER + VPT-Shallow": [
            [97.98, 97.23, 95.92, 94.93, 94.16, 93.27, 92.43, 91.72, 91.12, 90.56],
            [97.93, 97.33, 96.01, 94.6, 93.55, 92.8, 91.95, 91.19, 90.65, 90.13],
            [90.14, 90.08, 89.98, 90.06, 89.86, 89.26, 88.85, 88.48, 88.14, 87.87],
        ],
        "L2P": [
            [98.79, 92.46, 89.46, 87.36, 84.96, 82.69, 80.98, 79.57, 78.41, 77.25],
            [97.1, 92.53, 90.45, 87.05, 84.11, 81.84, 80.13, 78.61, 77.43, 76.35],
            [92.49, 86.59, 83.49, 81.94, 80.4, 78.82, 77.43, 76.06, 74.88, 73.84],
        ],
        "CODA-Prompt": [
            [94.81, 90.05, 85.18, 82.75, 81.59, 80.45, 79.16, 78.18, 77.26, 76.47],
            [93.51, 89.82, 83.94, 81.06, 79.91, 78.41, 77.29, 76.18, 75.12, 74.23],
            [93.94, 90.04, 85.59, 82.98, 81.61, 80.41, 79.24, 78.19, 77.07, 76.25],
        ],
        "DualPrompt": [
            [98.79, 94.11, 91.56, 88.99, 86.97, 85.16, 83.67, 82.25, 81.03, 79.85],
            [97.93, 94.06, 91.76, 89.04, 86.42, 84.59, 82.69, 81.11, 79.94, 78.83],
            [93.43, 90.64, 88.09, 86.89, 85.57, 83.89, 82.06, 80.46, 79.09, 77.91],
        ],
    },
    "OB": {
        "MM - CIL": [
            [94.5, 91.92, 89.81, 87.85, 86.21, 84.84, 83.59, 82.39, 81.24, 80.2],
            [95.0, 92.08, 90.4, 89.22, 87.85, 86.27, 85.01, 83.74, 82.64, 81.59],
            [94.82, 92.69, 91.12, 89.27, 87.61, 86.2, 84.95, 83.78, 82.62, 81.57],
        ],
        "MM - CA": [
            [94.5, 93.25, 91.57, 89.34, 87.5, 85.89, 84.52, 83.13, 81.78, 80.5],
            [95.0, 93.16, 91.77, 90.47, 88.84, 87.13, 85.67, 84.21, 82.97, 81.86],
            [94.82, 92.15, 90.6, 88.86, 87.55, 86.24, 84.93, 83.73, 82.49, 81.36],
        ],
        "MM - LCA": [
            [94.5, 93.16, 91.62, 89.68, 88.02, 86.5, 85.2, 83.89, 82.62, 81.4]
        ],
        "MOS": [
            [92.33, 92.33, 91.72, 90.73, 89.69, 88.66, 87.78, 86.98, 86.22, 85.58],
            [95.33, 93.54, 92.72, 91.96, 91.02, 89.98, 89.22, 88.39, 87.68, 86.86],
            [94.65, 93.1, 92.1, 91.12, 89.99, 88.88, 87.97, 87.18, 86.51, 85.83],
        ],
        "MOS - CIL": [
            [92.33, 88.82, 86.93, 85.02, 83.47, 81.98, 80.68, 79.41, 78.34, 77.47],
            [94.67, 91.7, 89.82, 88.28, 86.87, 85.17, 83.81, 82.46, 81.29, 80.2],
            [95.65, 91.6, 89.39, 87.36, 85.71, 84.03, 82.67, 81.49, 80.27, 79.18],
        ],
        "MOS - CA": [
            [92.33, 90.04, 88.56, 86.96, 85.66, 84.32, 83.23, 82.21, 81.34, 80.7],
            [94.67, 90.87, 89.17, 87.82, 86.46, 85.15, 84.07, 83.17, 82.37, 81.53],
            [95.65, 90.8, 88.91, 87.31, 85.87, 84.58, 83.46, 82.57, 81.69, 80.87],
        ],
        "MOS - LCA": [
            [92.33, 90.04, 88.65, 87.16, 85.84, 84.57, 83.51, 82.54, 81.72, 81.03],
            [94.67, 90.83, 88.96, 87.65, 86.36, 84.96, 83.99, 83.09, 82.38, 81.62],
            [95.65, 91.48, 89.6, 88.13, 86.78, 85.49, 84.39, 83.45, 82.57, 81.76],
        ],
        "SLCA": [
            [93.0, 91.5, 90.18, 88.57, 87.23, 85.98, 84.9, 83.88, 82.97, 82.11],
            [94.0, 92.42, 90.66, 89.51, 88.25, 87.01, 86.01, 85.01, 84.17, 83.32],
            [94.48, 92.52, 90.78, 89.15, 87.86, 86.62, 85.49, 84.43, 83.55, 82.59],
        ],
        "SLCA - CIL": [
            [93.0, 90.46, 88.46, 86.52, 84.96, 83.53, 82.31, 81.14, 80.09, 78.18],
            [94.0, 91.41, 89.43, 88.02, 86.56, 84.97, 83.75, 82.5, 81.47, 78.47],
            [94.48, 92.4, 90.62, 88.66, 87.33, 85.92, 84.64, 83.52, 82.39, 79.34],
        ],
        "SLCA - CA": [
            [93.0, 89.2, 87.59, 85.92, 84.38, 82.97, 81.71, 80.65, 79.6, 78.6],
            [94.0, 91.16, 88.84, 87.27, 85.85, 84.53, 83.39, 82.22, 81.12, 79.93],
            [94.48, 90.89, 88.63, 86.64, 85.22, 83.87, 82.44, 81.1, 79.87, 78.62],
        ],
        "SLCA - LCA": [
            [93.0, 90.24, 88.55, 86.76, 85.38, 84.0, 82.82, 81.74, 80.84, 80.02],
            [94.0, 91.04, 88.94, 87.69, 86.3, 85.04, 84.1, 83.1, 82.24, 81.32],
            [94.48, 91.18, 89.3, 87.63, 86.26, 85.0, 83.75, 82.63, 81.7, 80.7],
        ],
        "EASE": [
            [90.17, 89.46, 88.34, 86.92, 85.6, 84.31, 83.26, 82.24, 81.34, 80.61],
            [92.67, 90.08, 88.49, 87.26, 86.0, 84.62, 83.55, 82.53, 81.7, 80.87],
            [92.98, 90.18, 88.91, 87.59, 86.27, 84.9, 83.73, 82.71, 81.75, 80.87],
        ],
        "APER + Adapter": [
            [88.83, 88.45, 87.52, 86.12, 84.81, 83.51, 82.42, 81.39, 80.5, 79.78],
            [92.17, 89.66, 88.03, 86.86, 85.68, 84.33, 83.28, 82.24, 81.4, 80.55],
            [92.47, 89.6, 88.38, 87.11, 85.81, 84.45, 83.3, 82.32, 81.38, 80.52],
        ],
        "APER + Finetune": [
            [87.0, 86.12, 84.8, 83.28, 81.94, 80.53, 79.32, 78.23, 77.29, 76.55],
            [92.17, 89.16, 87.2, 85.61, 84.08, 82.48, 81.21, 79.99, 79.01, 78.05],
            [94.65, 90.64, 88.49, 86.73, 85.1, 83.51, 82.15, 81.0, 79.91, 78.89],
        ],
        "APER + SSF": [
            [89.83, 89.04, 87.85, 86.39, 85.08, 83.73, 82.63, 81.61, 80.71, 79.98],
            [93.0, 89.91, 88.12, 86.63, 85.28, 83.84, 82.76, 81.68, 80.84, 80.01],
            [94.15, 90.94, 89.39, 87.96, 86.59, 85.14, 83.92, 82.86, 81.87, 80.96],
        ],
        "APER + VPT-Deep": [
            [88.33, 87.74, 86.77, 85.43, 84.21, 82.94, 81.92, 80.93, 80.05, 79.33],
            [92.83, 90.0, 88.18, 86.72, 85.37, 83.95, 82.85, 81.78, 80.92, 80.04],
            [93.31, 89.97, 88.32, 86.83, 85.45, 84.01, 82.83, 81.81, 80.82, 79.91],
        ],
        "APER + VPT-Shallow": [
            [87.0, 86.78, 85.96, 84.7, 83.47, 82.24, 81.19, 80.2, 79.35, 78.65],
            [90.5, 88.74, 87.32, 86.16, 84.96, 83.62, 82.53, 81.47, 80.62, 79.78],
            [92.98, 89.85, 88.52, 87.27, 85.99, 84.62, 83.44, 82.41, 81.43, 80.53],
        ],
        "L2P": [
            [89.67, 85.87, 83.44, 81.2, 79.41, 77.65, 76.26, 74.9, 73.66, 72.59],
            [93.67, 89.66, 85.93, 84.21, 82.64, 80.78, 79.08, 77.52, 76.22, 75.05],
            [92.81, 88.43, 85.6, 83.08, 81.06, 79.28, 77.81, 76.42, 75.06, 73.85],
        ],
        "CODA-Prompt": [
            [90.47, 87.42, 78.42, 74.3, 72.92, 72.48, 72.26, 71.78, 71.37, 71.03],
            [89.3, 86.04, 78.11, 73.87, 72.39, 72.0, 71.89, 71.56, 71.2, 70.86],
            [88.96, 85.71, 77.63, 73.74, 72.43, 72.15, 72.09, 71.75, 71.4, 71.04],
        ],
        "DualPrompt": [
            [88.67, 85.32, 83.36, 81.56, 80.02, 78.49, 77.03, 75.56, 74.27, 73.22],
            [93.67, 90.37, 87.43, 84.85, 82.89, 80.89, 79.29, 77.87, 76.72, 75.56],
            [93.65, 89.52, 87.27, 84.52, 82.28, 80.22, 78.61, 77.14, 75.71, 74.45],
        ],
    },
    "VTAB": {
        "MM - CIL": [
            [99.52, 98.38, 94.64, 89.9, 86.53],
            [97.31, 87.48, 84.3, 81.58, 79.07],
            [96.96, 97.43, 92.71, 90.4, 88.22],
        ],
        "MM - CA": [
            [99.52, 98.63, 97.0, 95.88, 94.56],
            [97.31, 96.14, 95.46, 94.26, 92.68],
            [96.96, 98.03, 97.62, 96.89, 95.56],
        ],
        "MM - LCA": [[99.52, 98.69, 97.42, 96.42, 95.21]],
        "MOS": [
            [94.0, 92.65, 92.61, 92.55, 92.71],
            [92.3, 91.04, 91.29, 91.48, 91.81],
            [95.1, 92.87, 92.76, 92.64, 92.73],
        ],
        "MOS - CIL": [
            [94.0, 91.4, 91.02, 89.86, 89.42],
            [92.5, 89.57, 89.66, 88.88, 88.67],
            [95.0, 92.1, 91.54, 90.18, 89.63],
        ],
        "MOS - CA": [
            [94.0, 91.22, 91.11, 90.36, 90.02],
            [92.5, 91.35, 91.3, 90.74, 90.42],
            [95.0, 92.54, 92.2, 91.19, 90.87],
        ],
        "MOS - LCA": [
            [94.0, 91.85, 91.55, 90.7, 90.42],
            [92.5, 91.72, 91.95, 91.34, 91.29],
            [95.0, 92.43, 92.24, 91.43, 91.23],
        ],
        "SLCA": [
            [98.91, 98.24, 96.14, 94.51, 92.47],
            [95.79, 90.06, 89.47, 88.42, 87.23],
            [97.18, 97.53, 96.74, 95.08, 93.55],
        ],
        "SLCA - CIL": [
            [98.91, 97.56, 92.32, 88.04, 83.53],
            [95.79, 87.12, 83.82, 81.11, 79.88],
            [97.18, 93.6, 86.16, 80.42, 77.13],
        ],
        "SLCA - CA": [
            [98.91, 94.75, 92.08, 90.08, 87.42],
            [95.79, 88.52, 88.44, 86.83, 84.8],
            [97.18, 97.37, 95.45, 92.0, 89.07],
        ],
        "SLCA - LCA": [
            [98.91, 92.68, 90.24, 88.89, 86.6],
            [95.79, 88.04, 87.24, 86.38, 84.79],
            [97.18, 97.76, 96.7, 93.2, 90.89],
        ],
        "EASE": [
            [95.7, 93.88, 93.66, 93.41, 93.41],
            [95.4, 93.66, 93.46, 93.18, 93.16],
            [95.6, 93.76, 93.55, 93.3, 93.28],
        ],
        "APER + Adapter": [
            [99.39, 97.02, 94.32, 92.75, 90.92],
            [98.23, 94.94, 93.22, 91.6, 90.0],
            [94.27, 94.97, 94.33, 92.92, 91.05],
        ],
        "APER + Finetune": [
            [99.03, 96.83, 94.27, 92.75, 90.96],
            [98.49, 95.1, 93.82, 92.48, 91.0],
            [97.22, 97.52, 96.8, 95.35, 93.48],
        ],
        "APER + SSF": [
            [99.39, 97.22, 94.66, 93.15, 91.32],
            [98.4, 95.11, 93.63, 92.21, 90.66],
            [97.87, 98.08, 97.21, 95.5, 93.54],
        ],
        "APER + VPT-Deep": [
            [99.52, 97.3, 94.77, 93.32, 91.59],
            [98.74, 95.38, 93.85, 92.25, 90.68],
            [97.27, 97.56, 96.9, 95.4, 93.45],
        ],
        "APER + VPT-Shallow": [
            [99.64, 97.34, 94.58, 92.99, 91.28],
            [98.32, 95.64, 94.13, 92.62, 90.96],
            [96.23, 96.66, 95.9, 94.35, 92.41],
        ],
        "L2P": [
            [98.67, 96.82, 92.39, 88.67, 84.4],
            [98.15, 88.68, 86.15, 82.41, 79.11],
            [95.05, 94.17, 91.02, 87.13, 83.59],
        ],
        "CODA-Prompt": [
            [73.8, 66.97, 65.89, 67.49, 67.46],
            [73.4, 62.48, 60.83, 63.16, 62.58],
            [69.1, 64.87, 65.55, 66.95, 66.72],
        ],
        "DualPrompt": [
            [98.31, 95.89, 92.66, 89.61, 86.24],
            [96.64, 87.0, 83.45, 79.6, 77.3],
            [96.88, 95.29, 92.84, 90.75, 88.42],
        ],
    },
    "CARS": {
        "MM - CIL": [
            [90.34, 85.74, 82.5, 80.07, 77.69, 75.45, 73.69, 72.09, 70.72, 69.55],
            [88.18, 84.46, 81.06, 78.51, 76.02, 73.98, 72.51, 71.24, 70.12, 69.01],
            [90.45, 87.08, 83.9, 81.46, 79.46, 77.58, 75.92, 74.39, 73.0, 71.84],
        ],
        "MM - CA": [
            [90.34, 88.48, 86.91, 85.48, 83.91, 82.31, 80.83, 79.29, 77.94, 76.77],
            [88.18, 87.27, 85.69, 83.48, 81.72, 79.75, 78.16, 76.57, 75.07, 73.74],
            [90.45, 89.87, 88.02, 86.15, 84.3, 82.57, 80.8, 79.28, 77.85, 76.62],
        ],
        "MM - LCA": [
            [90.34, 88.5, 87.18, 85.93, 84.46, 82.93, 81.43, 79.81, 78.39, 77.18],
            [88.18, 87.57, 86.17, 84.0, 82.11, 80.24, 78.65, 77.04, 75.53, 74.16],
            [90.45, 89.94, 88.48, 86.93, 85.06, 83.39, 81.57, 79.96, 78.44, 77.14],
        ],
        "MOS": [
            [91.08, 89.46, 88.65, 87.83, 87.04, 86.22, 85.39, 84.47, 83.6, 82.74],
            [68.9, 65.53, 62.38, 59.4, 56.95, 54.92, 53.22, 51.6, 50.12, 48.83],
            [90.92, 90.69, 89.5, 88.21, 87.13, 86.23, 85.29, 84.37, 83.46, 82.66],
        ],
        "MOS - CIL": [
            [90.64, 85.26, 82.29, 79.24, 75.24, 71.41, 68.15, 65.03, 62.3, 59.55],
            [90.92, 87.43, 82.04, 78.25, 74.62, 71.74, 69.02, 65.7, 62.75, 59.77],
        ],
        "MOS - CA": [
            [90.64, 82.86, 79.49, 76.75, 74.63, 72.67, 71.03, 69.47, 68.08, 66.81],
            [90.92, 85.3, 82.15, 79.5, 77.32, 75.58, 74.02, 72.56, 71.17, 69.89],
        ],
        "MOS - LCA": [
            [90.64, 83.82, 80.44, 77.44, 75.0, 72.67, 70.67, 68.76, 67.01, 65.47],
            [90.92, 86.2, 82.97, 80.31, 78.55, 77.02, 75.67, 74.34, 73.09, 71.98],
        ],
        "SLCA": [
            [89.0, 85.3, 80.63, 78.55, 77.51, 76.17, 75.06, 74.07, 73.12, 72.23],
            [84.29, 84.9, 81.8, 80.26, 79.26, 78.28, 77.61, 76.7, 75.76, 75.02],
            [84.82, 85.44, 83.61, 81.67, 80.55, 79.62, 78.79, 77.9, 77.19, 76.48],
        ],
        "SLCA - CIL": [
            [89.0, 83.44, 66.29, 61.6, 59.64, 57.82, 56.48, 55.27, 54.25, 53.09],
            [84.29, 75.18, 72.72, 71.08, 69.94, 68.81, 67.82, 66.8, 65.65, 64.76],
            [84.82, 80.64, 77.83, 75.56, 73.89, 72.84, 71.79, 70.78, 69.94, 69.21],
        ],
        "SLCA - CA": [
            [89.0, 80.41, 75.02, 71.22, 69.95, 68.22, 67.08, 65.62, 64.53, 63.45],
            [84.29, 79.65, 75.28, 73.08, 71.79, 70.77, 70.28, 69.55, 68.31, 67.48],
            [84.82, 81.26, 77.85, 75.87, 74.58, 73.54, 72.57, 71.55, 70.89, 68.23],
        ],
        "SLCA - LCA": [
            [89.0, 81.91, 73.5, 70.47, 69.41, 68.04, 66.75, 65.32, 64.35, 63.2],
            [84.29, 80.82, 75.9, 74.0, 72.73, 71.42, 70.7, 69.47, 68.08, 67.45],
            [84.82, 81.36, 78.19, 76.08, 74.78, 73.76, 72.59, 71.23, 70.33, 69.51],
        ],
        "EASE": [
            [79.05, 68.66, 64.61, 61.51, 58.71, 55.94, 53.77, 51.84, 50.13, 48.48],
            [74.49, 66.71, 62.11, 58.82, 56.25, 53.98, 51.76, 49.8, 48.15, 46.74],
            [74.02, 70.88, 65.07, 61.08, 58.42, 56.18, 54.28, 52.31, 50.7, 49.02],
        ],
        "APER + Adapter": [
            [76.52, 70.46, 66.79, 63.68, 61.09, 58.82, 56.77, 54.92, 53.24, 51.7],
            [73.56, 67.9, 64.04, 60.68, 58.12, 56.0, 54.14, 52.43, 50.85, 49.55],
            [70.89, 67.98, 64.59, 61.38, 59.03, 57.0, 55.27, 53.52, 51.96, 50.54],
        ],
        "APER + Finetune": [
            [78.9, 73.0, 69.55, 66.47, 63.93, 61.69, 59.7, 57.83, 56.16, 54.64],
            [76.05, 70.51, 66.66, 63.24, 60.61, 58.47, 56.56, 54.8, 53.21, 51.86],
            [72.3, 70.5, 67.53, 64.15, 61.65, 59.53, 57.81, 56.08, 54.54, 53.16],
        ],
        "APER + SSF": [
            [76.23, 70.4, 67.06, 64.0, 61.32, 59.03, 57.0, 55.12, 53.45, 51.93],
            [73.72, 67.97, 63.99, 60.59, 58.11, 56.16, 54.47, 52.85, 51.31, 50.04],
            [72.14, 69.56, 66.13, 62.74, 60.25, 58.21, 56.54, 54.83, 53.3, 51.92],
        ],
        "APER + VPT-Deep": [
            [73.55, 66.61, 62.87, 59.59, 56.82, 54.36, 52.23, 50.32, 48.67, 47.17],
            [76.05, 70.54, 66.6, 63.21, 60.77, 58.76, 56.98, 55.36, 53.85, 52.55],
            [73.4, 70.57, 67.14, 63.6, 61.0, 58.83, 57.01, 55.22, 53.64, 52.2],
        ],
        "APER + VPT-Shallow": [
            [76.37, 70.31, 66.66, 63.61, 61.02, 58.72, 56.66, 54.82, 53.15, 51.63],
            [74.03, 68.3, 64.33, 61.01, 58.52, 56.48, 54.68, 52.97, 51.37, 50.03],
            [70.74, 68.28, 65.02, 61.86, 59.44, 57.4, 55.66, 53.9, 52.34, 50.94],
        ],
        "L2P": [
            [79.35, 72.04, 68.49, 65.34, 62.53, 60.48, 58.77, 57.26, 55.96, 54.75],
            [77.14, 70.85, 67.21, 63.88, 61.85, 59.27, 57.24, 55.6, 54.1, 52.85],
            [79.5, 73.8, 68.36, 64.33, 61.12, 58.64, 56.84, 55.26, 53.89, 52.64],
        ],
        "CODA-Prompt": [
            [57.89, 46.5, 42.28, 36.69, 33.25, 30.86, 29.05, 27.69, 26.65, 25.77],
            [56.04, 47.31, 43.12, 37.82, 34.57, 32.04, 30.08, 28.55, 27.29, 26.22],
            [60.22, 48.18, 43.4, 38.7, 35.11, 32.56, 30.63, 29.05, 27.88, 26.88],
        ],
        "DualPrompt": [
            [83.21, 76.16, 71.66, 67.48, 63.86, 60.66, 57.92, 55.62, 53.66, 51.82],
            [77.92, 71.24, 66.68, 62.35, 59.24, 56.42, 53.99, 51.81, 49.89, 48.25],
            [78.09, 73.19, 67.74, 63.39, 59.72, 56.79, 54.24, 51.89, 49.88, 48.13],
        ],
    },
}

alpha_ablation = {  # seed=1993
    "IN-A": {
        "alpha=0.01": {
            "FAA": 65.99,
            "FFM": 12.04,
            "FFD": 30.53,
            "ASA": 75.00,
            "tasks": [
                54.86,
                69.19,
                65.52,
                69.37,
                63.64,
                67.16,
                69.86,
                63.78,
                63.95,
                72.58,
            ],
            "acc": [87.43, 85.93, 84.37, 82.63, 80.88, 79.37, 78.27, 77.11, 76.0, 75.0],
        },
        "alpha=0.1": {
            "FAA": 65.99,
            "FFM": 12.04,
            "FFD": 30.53,
            "ASA": 75.03,
            "tasks": [
                54.86,
                69.19,
                65.52,
                69.37,
                63.64,
                67.16,
                69.86,
                63.78,
                63.95,
                72.58,
            ],
            "acc": [
                87.43,
                86.07,
                84.46,
                82.7,
                80.94,
                79.42,
                78.31,
                77.15,
                76.03,
                75.03,
            ],
        },
        "alpha=0.5": {
            "FAA": 65.99,
            "FFM": 12.04,
            "FFD": 30.53,
            "ASA": 74.98,
            "tasks": [
                54.86,
                69.19,
                65.52,
                69.37,
                63.64,
                67.16,
                69.86,
                63.78,
                63.95,
                72.58,
            ],
            "acc": [87.43, 85.78, 84.4, 82.62, 80.86, 79.4, 78.25, 77.1, 75.98, 74.98],
        },
        "alpha=1.0": {
            "FAA": 65.98,
            "FFM": 11.81,
            "FFD": 31.21,
            "ASA": 74.81,
            "tasks": [
                54.86,
                69.19,
                65.52,
                69.37,
                63.64,
                67.16,
                69.86,
                62.99,
                64.63,
                72.58,
            ],
            "acc": [
                87.43,
                85.22,
                83.84,
                82.12,
                80.47,
                79.09,
                77.99,
                76.88,
                75.79,
                74.81,
            ],
        },
        "alpha=5.0": {
            "FAA": 66.07,
            "FFM": 11.63,
            "FFD": 33.72,
            "ASA": 74.68,
            "tasks": [
                53.71,
                68.65,
                65.52,
                71.17,
                62.94,
                66.42,
                68.49,
                63.78,
                66.67,
                73.39,
            ],
            "acc": [87.43, 85.35, 83.7, 81.92, 80.33, 79.0, 77.82, 76.72, 75.63, 74.68],
        },
        "alpha=10.0": {
            "FAA": 65.82,
            "FFM": 11.94,
            "FFD": 32.00,
            "ASA": 74.20,
            "tasks": [
                55.43,
                65.95,
                66.38,
                69.37,
                62.24,
                65.67,
                69.86,
                63.78,
                65.31,
                74.19,
            ],
            "acc": [87.43, 85.48, 83.6, 81.79, 80.16, 78.76, 77.5, 76.19, 75.13, 74.2],
        },
        "alpha=100.0": {
            "FAA": 60.06,
            "FFM": 17.09,
            "FFD": 38.71,
            "ASA": 70.11,
            "tasks": [
                44.0,
                62.7,
                56.03,
                67.57,
                57.34,
                63.43,
                60.27,
                59.06,
                59.18,
                70.97,
            ],
            "acc": [
                87.43,
                84.23,
                80.73,
                78.43,
                76.71,
                75.16,
                73.71,
                72.4,
                71.23,
                70.11,
            ],
        },
    },
    "VTAB": {
        "alpha=0.01": {
            "FAA": 89.37,
            "FFM": 8.18,
            "FFD": 12.11,
            "ASA": 94.53,
            "tasks": [91.16, 87.63, 82.0, 92.24, 93.8],
            "acc": [99.52, 98.57, 97.0, 95.83, 94.53],
        },
        "alpha=0.1": {
            "FAA": 89.31,
            "FFM": 8.25,
            "FFD": 11.95,
            "ASA": 94.49,
            "tasks": [91.16, 87.58, 81.94, 92.08, 93.8],
            "acc": [99.52, 98.55, 96.94, 95.79, 94.49],
        },
        "alpha=0.5": {
            "FAA": 89.12,
            "FFM": 8.61,
            "FFD": 11.80,
            "ASA": 94.49,
            "tasks": [90.92, 87.04, 81.75, 92.0, 93.91],
            "acc": [99.52, 98.56, 96.95, 95.83, 94.49],
        },
        "alpha=1.0": {
            "FAA": 89.04,
            "FFM": 8.88,
            "FFD": 12.27,
            "ASA": 94.47,
            "tasks": [90.8, 86.74, 81.51, 92.08, 94.05],
            "acc": [99.52, 98.56, 96.91, 95.83, 94.47],
        },
        "alpha=5.0": {
            "FAA": 88.67,
            "FFM": 9.52,
            "FFD": 13.41,
            "ASA": 94.38,
            "tasks": [90.44, 85.7, 80.53, 92.48, 94.22],
            "acc": [99.52, 98.53, 96.88, 95.81, 94.38],
        },
        "alpha=10.0": {
            "FAA": 88.00,
            "FFM": 9.97,
            "FFD": 15.28,
            "ASA": 94.03,
            "tasks": [90.56, 84.46, 79.12, 92.16, 93.7],
            "acc": [99.52, 98.52, 96.79, 95.54, 94.03],
        },
        "alpha=100.0": {
            "FAA": 84.27,
            "FFM": 13.35,
            "FFD": 15.98,
            "ASA": 92.05,
            "tasks": [88.86, 78.67, 74.46, 88.4, 90.97],
            "acc": [99.52, 97.54, 95.5, 94.0, 92.05],
        },
    },
}

backbone_ablation = {  # seed=1993
    "VTAB": {
        "vit_b16_224_ssf_lca": {
            "FAA": 83.92,
            "FFM": 15.77,
            "FFD": 19.09,
            "ASA": 92.08,
            "tasks": [85.47, 73.78, 74.46, 91.44, 94.43],
            "acc": [99.88, 98.59, 95.61, 94.12, 92.08],
        },
        "vit_b16_224_in21k_ssf_lca": {
            "FAA": 71.32,
            "FFM": 31.73,
            "FFD": 5.48,
            "ASA": 88.26,
            "tasks": [65.38, 64.42, 67.42, 64.64, 94.74],
            "acc": [99.64, 98.94, 94.21, 92.5, 88.26],
        },
        "vit_b16_224_vpt_lca": {  # 5 token
            "FAA": 79.14,
            "FFM": 20.38,
            "FFD": 23.61,
            "ASA": 86.92,
            "tasks": [72.15, 67.05, 73.3, 89.2, 93.98],
            "acc": [98.43, 95.89, 90.77, 88.86, 86.92],
        },
        "vit_b16_224_in21k_vpt_lca": {  # 5 token
            "FAA": 66.80,
            "FFM": 31.86,
            "FFD": 41.87,
            "ASA": 79.46,
            "tasks": [68.64, 40.48, 59.77, 78.72, 86.4],
            "acc": [81.96, 88.29, 83.95, 82.63, 79.46],
        },
        "vit_b16_224_adapter_lca": {  # ffn_num=64
            "FAA": 91.93,
            "FFM": 3.74,
            "FFD": 6.98,
            "ASA": 95.75,
            "tasks": [96.13, 92.87, 86.77, 92.4, 91.48],
            "acc": [99.76, 98.82, 97.67, 96.71, 95.75],
        },
        "vit_b16_224_in21k_adapter_lca": {  # ffn_num=64
            "FAA": 92.13,
            "FFM": 4.13,
            "FFD": 4.69,
            "ASA": 95.94,
            "tasks": [94.31, 92.92, 88.0, 93.92, 91.52],
            "acc": [99.88, 99.19, 97.75, 96.89, 95.94],
        },
        "vit_b16_224_lora_lca": {  # r=64, a=128
            "FAA": 88.17,
            "FFM": 9.50,
            "FFD": 18.10,
            "ASA": 93.84,
            "tasks": [90.68, 86.05, 75.93, 94.16, 94.05],
            "acc": [99.52, 98.41, 96.44, 95.26, 93.84],
        },
        "vit_b16_224_in21k_lora_lca": {  # r=64, a=128
            "FAA": 88.58,
            "FFM": 9.28,
            "FFD": 16.04,
            "ASA": 94.03,
            "tasks": [87.65, 86.84, 79.98, 94.88, 93.56],
            "acc": [99.88, 98.71, 96.78, 95.39, 94.03],
        },
        # "vit_b16_224_vpt": { # 64 token
        #     "FAA": 80.97, "FFM": 19.57, "FFD": 32.40, "ASA": 91.57,
        #     "tasks": [87.77, 61.85, 66.07, 94.16, 94.98],
        #     "acc": [99.64, 98.57, 96.06, 94.23, 91.57]
        # },
        # "vit_b16_224_in21k_vpt": { # 64 token
        #     "FAA": 77.63, "FFM": 23.21, "FFD": 24.25, "ASA": 86.80,
        #     "tasks": [67.19, 67.69, 67.67, 89.28, 96.33],
        #     "acc": [96.25, 96.23, 91.68, 89.09, 86.8]
        # },
        "vit_b16_224_ssf_cil": {
            "acc": [99.88, 97.99, 93.71, 90.41, 86.79]
        },
        "vit_b16_224_in21k_ssf_cil": {
            "acc": [99.64, 98.56, 92.93, 89.94, 86.04]
        },
        "vit_b16_224_vpt_cil": {  # 5 token
            "acc": [99.64, 90.28, 82.69, 78.15, 75.11]
        },
        "vit_b16_224_in21k_vpt_cil": {  # 5 token
            "acc": [96.25, 73.5, 68.1, 64.81, 62.08]
        },
        "vit_b16_224_adapter_cil": {  # ffn_num=64
            "acc": [99.76, 98.22, 95.6, 92.02, 88.81]
        },
        "vit_b16_224_in21k_adapter_cil": {  # ffn_num=64
            "acc": [99.88, 98.81, 96.61, 93.57, 90.95]
        },
        "vit_b16_224_lora_cil": {  # r=64,
            "acc": [99.52, 98.38, 94.64, 89.9, 86.53]
        },
        "vit_b16_224_in21k_lora_cil": {  # r=64,
            "acc": [99.88, 97.92, 94.86, 91.91, 89.9]
        }
    }
}

ablation = {
    "CIFAR100": {
        "MM - CIL": {
            "1993": {
                "FAA": 88.33,
                "FFM": 6.82,
                "FFD": 12.40,
                "ASA": 92.75,
                "tasks": [85.5, 86.2, 93.0, 88.4, 83.2, 86.0, 89.4, 86.9, 93.1, 91.6],
                "acc": [
                    98.8,
                    97.82,
                    97.03,
                    96.41,
                    95.67,
                    95.01,
                    94.51,
                    93.83,
                    93.25,
                    92.75,
                ],
            },
            "1994": {
                "FAA": 88.26,
                "FFM": 7.94,
                "FFD": 19.30,
                "ASA": 92.94,
                "tasks": [77.8, 89.8, 89.1, 86.7, 84.8, 94.2, 82.6, 90.0, 93.0, 94.6],
                "acc": [
                    97.7,
                    97.45,
                    97.09,
                    96.44,
                    95.84,
                    95.33,
                    94.76,
                    94.1,
                    93.46,
                    92.94,
                ],
            },
            "1995": {
                "FAA": 89.05,
                "FFM": 6.84,
                "FFD": 14.70,
                "ASA": 92.76,
                "tasks": [86.1, 84.6, 89.4, 82.0, 85.6, 92.4, 93.5, 94.5, 89.9, 92.5],
                "acc": [99.1, 97.65, 96.76, 95.91, 95.16, 94.62, 94.11, 93.62, 93.17, 92.76],
            },
        },
        "MM - CA": {
            "1993": {
                "FAA": 90.77,
                "FFM": 6.06,
                "FFD": 14.10,
                "ASA": 94.40,
                "tasks": [84.7, 90.3, 94.1, 91.7, 87.3, 86.1, 92.9, 90.3, 94.9, 95.4],
                "acc": [
                    98.8,
                    98.1,
                    97.71,
                    97.23,
                    96.74,
                    96.28,
                    95.81,
                    95.27,
                    94.8,
                    94.4,
                ],
            },
            "1994": {
                "FAA": 90.27,
                "FFM": 6.92,
                "FFD": 15.50,
                "ASA": 94.14,
                "tasks": [80.9, 92.9, 85.5, 87.8, 89.7, 94.7, 90.6, 91.0, 93.6, 96.0],
                "acc": [
                    97.7,
                    97.48,
                    97.32,
                    96.93,
                    96.48,
                    96.14,
                    95.62,
                    95.08,
                    94.56,
                    94.14,
                ],
            },
            "1995": {
                "FAA": 89.46,
                "FFM": 7.73,
                "FFD": 19.50,
                "ASA": 93.47,
                "tasks": [79.1, 86.6, 89.8, 84.7, 80.7, 95.2, 93.6, 95.0, 94.0, 95.9],
                "acc": [
                    99.1,
                    97.98,
                    97.22,
                    96.63,
                    96.03,
                    95.41,
                    94.89,
                    94.43,
                    93.92,
                    93.47,
                ],
            },
        },
        "MM - LCA": {
            "1993": {
                "FAA": 91.30,
                "FFM": 5.29,
                "FFD": 10.80,
                "ASA": 94.74,
                "tasks": [87.1, 90.6, 94.4, 92.7, 88.9, 88.1, 93.5, 89.1, 93.9, 94.7],
                "acc": [
                    98.8,
                    98.12,
                    97.76,
                    97.33,
                    96.87,
                    96.44,
                    96.05,
                    95.55,
                    95.12,
                    94.74,
                ],
            }
        },
    },
    "IN-R": {
        "MM - CIL": {
            "1993": {
                "FAA": 80.10,
                "FFM": 7.23,
                "FFD": 15.48,
                "ASA": 85.06,
                "tasks": [
                    77.5,
                    79.43,
                    75.25,
                    76.88,
                    75.56,
                    86.06,
                    80.14,
                    85.73,
                    80.54,
                    83.94,
                ],
                "acc": [
                    94.63,
                    92.47,
                    91.04,
                    89.76,
                    88.48,
                    87.55,
                    86.82,
                    86.18,
                    85.61,
                    85.06,
                ],
            },
            "1994": {
                "FAA": 78.14,
                "FFM": 8.68,
                "FFD": 19.60,
                "ASA": 83.19,
                "tasks": [
                    72.73,
                    75.21,
                    73.96,
                    76.26,
                    83.02,
                    77.48,
                    79.01,
                    79.86,
                    79.33,
                    84.56,
                ],
                "acc": [
                    93.84,
                    91.76,
                    90.31,
                    88.45,
                    87.11,
                    86.07,
                    85.19,
                    84.43,
                    83.75,
                    83.19,
                ],
            },
            "1995": {
                "FAA": 79.27,
                "FFM": 8.71,
                "FFD": 16.77,
                "ASA": 84.58,
                "tasks": [
                    74.39,
                    77.01,
                    75.73,
                    79.01,
                    78.5,
                    82.97,
                    74.56,
                    85.84,
                    76.59,
                    88.09,
                ],
                "acc": [
                    94.0,
                    91.98,
                    90.37,
                    89.2,
                    88.25,
                    87.46,
                    86.56,
                    85.83,
                    85.17,
                    84.58,
                ],
            },
        },
        "MM - CA": {
            "1993": {
                "FAA": 80.47,
                "FFM": 8.92,
                "FFD": 14.80,
                "ASA": 85.47,
                "tasks": [
                    76.49,
                    79.11,
                    75.58,
                    78.11,
                    76.59,
                    80.45,
                    81.37,
                    85.6,
                    82.43,
                    88.95,
                ],
                "acc": [
                    94.63,
                    92.46,
                    91.2,
                    90.22,
                    88.88,
                    87.9,
                    87.16,
                    86.58,
                    86.03,
                    85.47,
                ],
            },
            "1994": {
                "FAA": 80.05,
                "FFM": 9.86,
                "FFD": 14.94,
                "ASA": 84.94,
                "tasks": [
                    75.37,
                    76.35,
                    76.12,
                    72.03,
                    82.44,
                    79.83,
                    82.25,
                    82.43,
                    84.37,
                    89.33,
                ],
                "acc": [
                    93.84,
                    92.1,
                    90.76,
                    89.24,
                    88.25,
                    87.39,
                    86.76,
                    86.12,
                    85.49,
                    84.94,
                ],
            },
            "1995": {
                "FAA": 80.89,
                "FFM": 9.43,
                "FFD": 14.00,
                "ASA": 85.19,
                "tasks": [
                    77.15,
                    74.71,
                    79.13,
                    78.41,
                    78.5,
                    82.34,
                    77.67,
                    86.19,
                    82.94,
                    91.91,
                ],
                "acc": [
                    94.0,
                    92.19,
                    90.63,
                    89.3,
                    88.33,
                    87.58,
                    86.86,
                    86.27,
                    85.66,
                    85.19,
                ],
            },
        },
        "MM - LCA": {
            "1993": {
                "FAA": 81.40,
                "FFM": 7.38,
                "FFD": 12.45,
                "ASA": 85.83,
                "tasks": [
                    79.25,
                    80.22,
                    78.05,
                    78.28,
                    77.41,
                    82.42,
                    82.07,
                    85.46,
                    82.43,
                    88.43,
                ],
                "acc": [
                    94.63,
                    92.57,
                    91.28,
                    90.28,
                    89.05,
                    88.17,
                    87.42,
                    86.85,
                    86.32,
                    85.83,
                ],
            }
        },
    },
    "IN - A": {
        "MM - CIL": {
            "1993": {
                "FAA": 55.64,
                "FFM": 13.50,
                "FFD": 37.50,
                "ASA": 66.52,
                "tasks": [
                    49.14,
                    54.59,
                    57.76,
                    50.45,
                    59.44,
                    60.45,
                    58.22,
                    58.27,
                    47.62,
                    60.48,
                ],
                "acc": [
                    87.43,
                    84.56,
                    81.13,
                    78.0,
                    75.16,
                    72.74,
                    70.74,
                    69.16,
                    67.73,
                    66.52,
                ],
            },
            "1994": {
                "FAA": 55.63,
                "FFM": 13.46,
                "FFD": 30.00,
                "ASA": 67.61,
                "tasks": [
                    52.14,
                    57.94,
                    55.14,
                    54.82,
                    55.7,
                    55.91,
                    56.38,
                    52.56,
                    60.43,
                    55.26,
                ],
                "acc": [
                    82.14,
                    78.91,
                    77.04,
                    75.23,
                    73.71,
                    72.48,
                    71.33,
                    70.2,
                    68.94,
                    67.61,
                ],
            },
            "1995": {
                "FAA": 56.57,
                "FFM": 13.12,
                "FFD": 39.41,
                "ASA": 65.44,
                "tasks": [
                    45.88,
                    48.59,
                    60.31,
                    55.94,
                    64.49,
                    68.59,
                    52.12,
                    56.03,
                    66.36,
                    47.43,
                ],
                "acc": [
                    85.29,
                    79.37,
                    75.84,
                    73.48,
                    71.49,
                    70.13,
                    68.77,
                    67.49,
                    66.42,
                    65.44,
                ],
            },
        },
        "MM - CA": {
            "1993": {
                "FAA": 64.03,
                "FFM": 14.21,
                "FFD": 32.37,
                "ASA": 73.72,
                "tasks": [
                    50.29,
                    67.03,
                    63.79,
                    67.57,
                    60.14,
                    66.42,
                    69.18,
                    61.42,
                    61.9,
                    72.58,
                ],
                "acc": [
                    87.43,
                    86.06,
                    84.14,
                    81.79,
                    80.06,
                    78.4,
                    77.07,
                    75.89,
                    74.79,
                    73.72,
                ],
            },
            "1994": {
                "FAA": 64.61,
                "FFM": 12.09,
                "FFD": 23.88,
                "ASA": 71.92,
                "tasks": [
                    58.57,
                    52.38,
                    64.86,
                    73.49,
                    63.29,
                    63.78,
                    65.1,
                    70.7,
                    65.47,
                    68.42,
                ],
                "acc": [
                    82.14,
                    80.32,
                    78.95,
                    77.94,
                    76.74,
                    75.58,
                    74.63,
                    73.67,
                    72.74,
                    71.92,
                ],
            },
            "1995": {
                "FAA": 63.67,
                "FFM": 13.48,
                "FFD": 15.70,
                "ASA": 71.11,
                "tasks": [
                    63.53,
                    54.23,
                    51.91,
                    58.74,
                    68.22,
                    76.28,
                    69.09,
                    58.62,
                    63.55,
                    72.57,
                ],
                "acc": [
                    85.29,
                    81.34,
                    78.83,
                    77.13,
                    75.53,
                    74.41,
                    73.43,
                    72.68,
                    71.94,
                    71.11,
                ],
            },
        },
        "MM - LCA": {
            "1993": {
                "FAA": 65.89,
                "FFM": 12.78,
                "FFD": 29.17,
                "ASA": 74.77,
                "tasks": [
                    54.86,
                    65.41,
                    65.52,
                    69.82,
                    62.94,
                    69.4,
                    69.18,
                    62.99,
                    64.63,
                    74.19,
                ],
                "acc": [
                    87.43,
                    86.2,
                    84.33,
                    82.68,
                    80.87,
                    79.45,
                    78.13,
                    76.93,
                    75.75,
                    74.77,
                ],
            }
        },
    },
    "CUB": {
        "MM - CIL": {
            "1993": {
                "FAA": 80.28,
                "FFM": 9.56,
                "FFD": 29.11,
                "ASA": 87.16,
                "tasks": [
                    68.83,
                    78.74,
                    84.4,
                    82.59,
                    80.67,
                    78.51,
                    78.22,
                    87.2,
                    81.43,
                    82.17,
                ],
                "acc": [
                    98.38,
                    97.29,
                    95.63,
                    94.21,
                    92.71,
                    90.96,
                    89.69,
                    88.76,
                    87.93,
                    87.16,
                ],
            },
            "1994": {
                "FAA": 79.29,
                "FFM": 10.11,
                "FFD": 24.50,
                "ASA": 87.14,
                "tasks": [
                    72.61,
                    82.73,
                    76.61,
                    72.93,
                    80.65,
                    85.47,
                    70.87,
                    78.6,
                    86.94,
                    85.53,
                ],
                "acc": [
                    97.93,
                    97.44,
                    96.5,
                    95.11,
                    93.07,
                    91.6,
                    90.14,
                    88.94,
                    88.01,
                    87.14,
                ],
            },
            "1995": {
                "FAA": 80.01,
                "FFM": 10.22,
                "FFD": 25.36,
                "ASA": 85.72,
                "tasks": [
                    69.95,
                    77.82,
                    82.38,
                    84.3,
                    80.0,
                    73.21,
                    79.38,
                    78.85,
                    89.92,
                    84.34,
                ],
                "acc": [
                    94.37,
                    93.59,
                    91.61,
                    91.01,
                    90.37,
                    89.24,
                    88.25,
                    87.23,
                    86.35,
                    85.72,
                ],
            },
        },
        "MM - CA": {
            "1993": {
                "FAA": 83.32,
                "FFM": 13.65,
                "FFD": 27.73,
                "ASA": 90.18,
                "tasks": [
                    70.04,
                    67.15,
                    81.19,
                    77.33,
                    84.76,
                    88.6,
                    84.89,
                    90.8,
                    93.25,
                    95.22,
                ],
                "acc": [
                    98.38,
                    97.72,
                    96.47,
                    95.35,
                    94.39,
                    93.37,
                    92.57,
                    91.72,
                    90.94,
                    90.18,
                ],
            },
            "1994": {
                "FAA": 82.37,
                "FFM": 14.62,
                "FFD": 31.99,
                "ASA": 89.66,
                "tasks": [
                    63.9,
                    86.35,
                    76.61,
                    79.04,
                    68.66,
                    79.91,
                    86.22,
                    90.53,
                    94.69,
                    97.81,
                ],
                "acc": [
                    97.93,
                    97.34,
                    96.5,
                    95.53,
                    94.26,
                    93.38,
                    92.37,
                    91.34,
                    90.47,
                    89.66,
                ],
            },
            "1995": {
                "FAA": 84.98,
                "FFM": 11.09,
                "FFD": 22.06,
                "ASA": 89.53,
                "tasks": [
                    73.71,
                    73.79,
                    82.38,
                    83.41,
                    82.22,
                    84.38,
                    94.55,
                    85.02,
                    93.55,
                    96.79,
                ],
                "acc": [
                    94.37,
                    94.21,
                    93.2,
                    92.81,
                    92.41,
                    91.73,
                    91.18,
                    90.56,
                    90.04,
                    89.53,
                ],
            },
        },
        "MM - LCA": {
            "1993": {
                "FAA": 83.85,
                "FFM": 12.94,
                "FFD": 26.34,
                "ASA": 90.67,
                "tasks": [
                    72.87,
                    68.6,
                    83.03,
                    76.92,
                    84.76,
                    88.16,
                    84.89,
                    91.6,
                    92.41,
                    95.22,
                ],
                "acc": [
                    98.38,
                    98.04,
                    96.61,
                    95.68,
                    94.74,
                    93.87,
                    93.06,
                    92.25,
                    91.43,
                    90.67,
                ],
            }
        },
    },
    "OB": {
        "MM - CIL": {
            "1993": {
                "FAA": 70.90,
                "FFM": 18.19,
                "FFD": 29.33,
                "ASA": 80.20,
                "tasks": [
                    61.83,
                    66.61,
                    67.61,
                    59.3,
                    66.61,
                    66.56,
                    77.55,
                    71.79,
                    83.47,
                    87.63,
                ],
                "acc": [
                    94.5,
                    91.92,
                    89.81,
                    87.85,
                    86.21,
                    84.84,
                    83.59,
                    82.39,
                    81.24,
                    80.2,
                ],
            },
            "1994": {
                "FAA": 72.15,
                "FFM": 16.54,
                "FFD": 31.67,
                "ASA": 81.59,
                "tasks": [
                    63.33,
                    64.77,
                    70.28,
                    70.02,
                    65.66,
                    71.5,
                    81.5,
                    73.2,
                    83.95,
                    77.26,
                ],
                "acc": [
                    95.0,
                    92.08,
                    90.4,
                    89.22,
                    87.85,
                    86.27,
                    85.01,
                    83.74,
                    82.64,
                    81.59,
                ],
            },
            "1995": {
                "FAA": 72.06,
                "FFM": 15.39,
                "FFD": 24.71,
                "ASA": 81.57,
                "tasks": [
                    68.56,
                    62.77,
                    73.08,
                    69.45,
                    72.29,
                    69.58,
                    73.96,
                    76.96,
                    75.13,
                    78.83,
                ],
                "acc": [
                    94.82,
                    92.69,
                    91.12,
                    89.27,
                    87.61,
                    86.2,
                    84.95,
                    83.78,
                    82.62,
                    81.57,
                ],
            },
        },
        "MM - CA": {
            "1993": {
                "FAA": 69.02,
                "FFM": 25.75,
                "FFD": 29.16,
                "ASA": 80.50,
                "tasks": [
                    56.83,
                    63.94,
                    64.11,
                    56.11,
                    66.44,
                    56.86,
                    69.85,
                    76.29,
                    85.81,
                    93.98,
                ],
                "acc": [
                    94.5,
                    93.25,
                    91.57,
                    89.34,
                    87.5,
                    85.89,
                    84.52,
                    83.13,
                    81.78,
                    80.5,
                ],
            },
            "1994": {
                "FAA": 71.90,
                "FFM": 22.42,
                "FFD": 28.69,
                "ASA": 81.86,
                "tasks": [
                    61.5,
                    63.44,
                    61.44,
                    63.82,
                    62.98,
                    67.33,
                    84.33,
                    76.05,
                    86.45,
                    91.64,
                ],
                "acc": [
                    95.0,
                    93.16,
                    91.77,
                    90.47,
                    88.84,
                    87.13,
                    85.67,
                    84.21,
                    82.97,
                    81.86,
                ],
            },
            "1995": {
                "FAA": 71.22,
                "FFM": 22.25,
                "FFD": 17.35,
                "ASA": 81.36,
                "tasks": [
                    67.06,
                    56.93,
                    70.74,
                    63.27,
                    66.11,
                    67.73,
                    73.62,
                    77.46,
                    78.46,
                    90.83,
                ],
                "acc": [
                    94.82,
                    92.15,
                    90.6,
                    88.86,
                    87.55,
                    86.24,
                    84.93,
                    83.73,
                    82.49,
                    81.36,
                ],
            },
        },
        "MM - LCA": {
            "1993": {
                "FAA": 70.38,
                "FFM": 23.93,
                "FFD": 25.98,
                "ASA": 81.40,
                "tasks": [
                    60.17,
                    67.45,
                    67.11,
                    58.46,
                    66.11,
                    59.03,
                    70.85,
                    76.13,
                    85.14,
                    93.31,
                ],
                "acc": [
                    94.5,
                    93.16,
                    91.62,
                    89.68,
                    88.02,
                    86.5,
                    85.2,
                    83.89,
                    82.62,
                    81.4,
                ],
            }
        },
    },
    "VTAB": {
        "MM - CIL": {
            "1993": {
                "FAA": 73.03,
                "FFM": 25.38,
                "FFD": 42.71,
                "ASA": 86.53,
                "tasks": [86.2, 52.45, 52.6, 96.0, 77.92],
                "acc": [99.52, 98.38, 94.64, 89.9, 86.53],
            },
            "1994": {
                "FAA": 69.07,
                "FFM": 22.46,
                "FFD": 42.71,
                "ASA": 79.07,
                "tasks": [45.67, 80.86, 76.15, 71.1, 71.55],
                "acc": [97.31, 87.48, 84.3, 81.58, 79.07],
            },
            "1995": {
                "FAA": 79.52,
                "FFM": 15.28,
                "FFD": 44.02,
                "ASA": 88.22,
                "tasks": [50.72, 97.53, 90.33, 73.48, 85.52],
                "acc": [96.96, 97.43, 92.71, 90.4, 88.22],
            },
        },
        "MM - CA": {
            "1993": {
                "FAA": 89.28,
                "FFM": 8.77,
                "FFD": 12.07,
                "ASA": 94.56,
                "tasks": [90.92, 87.14, 81.57, 92.4, 94.39],
                "acc": [99.52, 98.63, 97.0, 95.88, 94.56],
            },
            "1994": {
                "FAA": 86.40,
                "FFM": 11.51,
                "FFD": 27.73,
                "ASA": 92.68,
                "tasks": [68.04, 88.98, 87.72, 92.48, 94.76],
                "acc": [97.31, 96.14, 95.46, 94.26, 92.68],
            },
            "1995": {
                "FAA": 90.22,
                "FFM": 6.91,
                "FFD": 19.02,
                "ASA": 95.56,
                "tasks": [78.18, 98.77, 92.57, 90.08, 91.49],
                "acc": [96.96, 98.03, 97.62, 96.89, 95.56],
            },
        },
        "MM - LCA": {
            "1993": {
                "FAA": 90.37,
                "FFM": 7.34,
                "FFD": 9.90,
                "ASA": 95.21,
                "tasks": [93.1, 88.22, 83.71, 92.48, 94.36],
                "acc": [99.52, 98.69, 97.42, 96.42, 95.21],
            }
        },
    },
    "CARS": {
        "MM - CIL": {
            "1993": {
                "FAA": 58.97,
                "FFM": 16.73,
                "FFD": 49.03,
                "ASA": 69.55,
                "tasks": [
                    38.93,
                    58.53,
                    71.67,
                    43.39,
                    58.6,
                    58.91,
                    66.38,
                    61.21,
                    57.14,
                    74.97,
                ],
                "acc": [
                    90.34,
                    85.74,
                    82.5,
                    80.07,
                    77.69,
                    75.45,
                    73.69,
                    72.09,
                    70.72,
                    69.55,
                ],
            },
            "1994": {
                "FAA": 59.02,
                "FFM": 17.93,
                "FFD": 49.72,
                "ASA": 69.01,
                "tasks": [
                    35.77,
                    42.07,
                    69.02,
                    54.01,
                    72.12,
                    52.42,
                    63.29,
                    63.13,
                    66.51,
                    71.88,
                ],
                "acc": [
                    88.18,
                    84.46,
                    81.06,
                    78.51,
                    76.02,
                    73.98,
                    72.51,
                    71.24,
                    70.12,
                    69.01,
                ],
            },
            "1995": {
                "FAA": 61.33,
                "FFM": 19.55,
                "FFD": 48.56,
                "ASA": 71.84,
                "tasks": [
                    40.22,
                    55.5,
                    59.31,
                    54.83,
                    60.64,
                    62.04,
                    57.86,
                    75.87,
                    75.21,
                    71.77,
                ],
                "acc": [
                    90.45,
                    87.08,
                    83.9,
                    81.46,
                    79.46,
                    77.58,
                    75.92,
                    74.39,
                    73.0,
                    71.84,
                ],
            },
        },
        "MM - CA": {
            "1993": {
                "FAA": 66.20,
                "FFM": 26.63,
                "FFD": 54.69,
                "ASA": 76.77,
                "tasks": [
                    30.76,
                    54.45,
                    61.14,
                    63.47,
                    63.51,
                    64.85,
                    69.3,
                    76.58,
                    85.71,
                    92.22,
                ],
                "acc": [
                    90.34,
                    88.48,
                    86.91,
                    85.48,
                    83.91,
                    82.31,
                    80.83,
                    79.29,
                    77.94,
                    76.77,
                ],
            },
            "1994": {
                "FAA": 61.83,
                "FFM": 31.41,
                "FFD": 57.67,
                "ASA": 73.74,
                "tasks": [
                    23.17,
                    38.75,
                    46.66,
                    52.92,
                    56.61,
                    66.71,
                    78.9,
                    80.02,
                    81.26,
                    93.33,
                ],
                "acc": [
                    88.18,
                    87.27,
                    85.69,
                    83.48,
                    81.72,
                    79.75,
                    78.16,
                    76.57,
                    75.07,
                    73.74,
                ],
            },
            "1995": {
                "FAA": 65.54,
                "FFM": 27.90,
                "FFD": 53.51,
                "ASA": 76.62,
                "tasks": [
                    31.3,
                    43.89,
                    66.05,
                    62.87,
                    63.69,
                    63.35,
                    67.94,
                    80.88,
                    86.25,
                    89.21,
                ],
                "acc": [
                    90.45,
                    89.87,
                    88.02,
                    86.15,
                    84.3,
                    82.57,
                    80.8,
                    79.28,
                    77.85,
                    76.62,
                ],
            },
        },
        "MM - LCA": {
            "1993": {
                "FAA": 66.32,
                "FFM": 27.04,
                "FFD": 52.53,
                "ASA": 77.18,
                "tasks": [
                    31.8,
                    54.69,
                    59.93,
                    63.09,
                    64.5,
                    64.49,
                    69.05,
                    77.2,
                    85.59,
                    92.83,
                ],
                "acc": [
                    90.34,
                    88.5,
                    87.18,
                    85.93,
                    84.46,
                    82.93,
                    81.43,
                    79.81,
                    78.39,
                    77.18,
                ],
            },
            "1994": {
                "FAA": 61.86,
                "FFM": 32.32,
                "FFD": 53.93,
                "ASA": 74.16,
                "tasks": [
                    25.35,
                    38.13,
                    46.42,
                    52.43,
                    58.06,
                    64.72,
                    79.51,
                    79.28,
                    81.15,
                    93.58,
                ],
                "acc": [
                    88.18,
                    87.57,
                    86.17,
                    84.0,
                    82.11,
                    80.24,
                    78.65,
                    77.04,
                    75.53,
                    74.16,
                ],
            },
            "1995": {
                "FAA": 65.48,
                "FFM": 28.57,
                "FFD": 53.04,
                "ASA": 77.14,
                "tasks": [
                    31.3,
                    44.01,
                    67.16,
                    62.5,
                    62.84,
                    62.28,
                    68.06,
                    80.53,
                    86.13,
                    89.96,
                ],
                "acc": [
                    90.45,
                    89.94,
                    88.48,
                    86.93,
                    85.06,
                    83.39,
                    81.57,
                    79.96,
                    78.44,
                    77.14,
                ],
            },
        },
    },
}


def create_lca_performance_bar_chart():
    """
    Create a bar chart showing LCA method performance improvement from CIL -> CA -> LCA.
    Values are calculated as the mean of the last value of each subarray.
    """
    mm_methods = ["MM - CIL", "MM - LCA"]

    # Calculate performance values for each dataset and method
    performance_data = {}
    for method in mm_methods:
        performance_data[method] = []

        for dataset in DATASETS:
            if dataset in results and method in results[dataset]:
                method_data = results[dataset][method]
                if method_data:  # Check if data exists
                    # Handle both single arrays and nested arrays
                    if isinstance(method_data[0], list):
                        # Nested arrays - calculate mean of the last values from each subarray
                        last_values = [subarray[-1] for subarray in method_data]
                        mean_performance = np.mean(last_values)
                    else:
                        # Single array - take the last value
                        mean_performance = method_data[-1]
                    performance_data[method].append(mean_performance)
                else:
                    performance_data[method].append(0)  # No data available
            else:
                performance_data[method].append(0)  # Method not found

    # Set publication-ready style
    set_publication_style()

    # Create the bar chart with larger figure size
    fig, ax = plt.subplots(figsize=(14, 10))

    x = np.arange(len(DATASETS))
    width = 0.25

    # Use consistent colors from the global color scheme
    colors = {method: get_method_color(method) for method in mm_methods}

    # Create clean legend labels without "MM - " prefix
    # legend_labels = {method: method.replace("MM - ", "") for method in mm_methods}
    legend_labels = {
        "MM - CIL": "IM",
        "MM - LCA": "IM+LCA",
    }

    # Create bars for each method with thicker edges
    bars = {}
    for i, method in enumerate(mm_methods):
        bars[method] = ax.bar(
            x + i * width,
            performance_data[method],
            width,
            label=legend_labels[method],
            color=colors[method],
            alpha=0.8,
            edgecolor="black",
            linewidth=1.5,
        )

    # Customize the chart with larger fonts
    ax.set_xlabel("Datasets", fontsize=26)
    ax.set_ylabel("Accuracy (%)", fontsize=26)
    ax.set_xticks(x + width)
    ax.set_xticklabels(DATASETS, rotation=45, ha="right", fontsize=24)
    ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=18, loc="upper right")

    # Add value labels on bars with larger font
    for method in mm_methods:
        for i, bar in enumerate(bars[method]):
            height = bar.get_height()
            if height > 0:  # Only show label if there's data
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.5,
                    f"{height:.1f}",
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=24,
                    fontweight="bold",
                )

    # Add grid for better readability with thicker lines
    ax.grid(True, alpha=0.4, axis="y", linewidth=1.5)
    ax.set_axisbelow(True)

    # Adjust y-axis limits and add tick parameters
    y_min = (
        min(
            [
                min(performance_data[method])
                for method in mm_methods
                if performance_data[method]
            ]
        )
        - 5
    )
    y_max = (
        max(
            [
                max(performance_data[method])
                for method in mm_methods
                if performance_data[method]
            ]
        )
        + 5
    )
    ax.set_ylim(y_min, y_max)
    # ax.set_yticks(np.arange(int(y_min), int(y_max) + 1, 5))
    ax.tick_params(axis="y", labelsize=24)

    # Make tick marks more visible
    ax.tick_params(axis="both", which="major", length=8, width=2)

    plt.tight_layout()

    # Save as PDF in figures folder with high DPI for paper quality
    pdf_path = "figures/fig_lca_performance_comparison.pdf"
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight", dpi=1200)
    print(f"Figure saved as: {pdf_path}")
    plt.close()

    # Reset matplotlib parameters to default
    reset_plot_style()


def create_accuracy_line_plots(
    selected_datasets=None,
    selected_methods=None,
    figsize_per_plot=(4, 3),
    ncols=3,
    show_improvement=True,
    save_path="figures/accuracy_line_plots.pdf",
):
    """
    Create line plots showing accuracy progression for selected datasets and methods.

    Args:
        selected_datasets (list): List of dataset names to include (default: all datasets)
        selected_methods (list): List of method names to include (default: all methods)
        figsize_per_plot (tuple): Size of each subplot (width, height)
        ncols (int): Number of columns in the subplot grid
        show_improvement (bool): Whether to show improvement arrows and values
        save_path (str): Path to save the figure
    """
    if selected_datasets is None:
        selected_datasets = DATASETS

    if selected_methods is None:
        # Get all unique methods across selected datasets
        all_methods = set()
        for dataset in selected_datasets:
            if dataset in results:
                all_methods.update(results[dataset].keys())
        selected_methods = sorted(list(all_methods))

    # Filter to only include datasets that exist in results
    available_datasets = [d for d in selected_datasets if d in results]

    if not available_datasets:
        print("No datasets found in results!")
        return

    # Calculate subplot layout
    nrows = (len(available_datasets) + ncols - 1) // ncols

    # Set publication-ready style
    set_publication_style()

    # Create figure with subplots
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows)
    )

    # Ensure axes is always a 2D array
    if nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    # Plot each dataset
    for idx, dataset in enumerate(available_datasets):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]

        # Get methods available for this dataset
        dataset_methods = [m for m in selected_methods if m in results[dataset]]

        if not dataset_methods:
            ax.text(
                0.5,
                0.5,
                f"No data for\n{dataset}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(f"({chr(97+idx)}) {dataset}")
            continue

        # Plot lines for each method
        for method in dataset_methods:
            method_data = results[dataset][method]
            color = get_method_color(method)

            if method_data:
                if isinstance(method_data[0], list):
                    # Multiple runs - plot mean and optionally std
                    max_length = max(len(run) for run in method_data)

                    # Pad shorter runs with their last value
                    padded_runs = []
                    for run in method_data:
                        padded_run = list(run)
                        while len(padded_run) < max_length:
                            padded_run.append(padded_run[-1])
                        padded_runs.append(padded_run)

                    means = np.mean(padded_runs, axis=0)
                    stds = np.std(padded_runs, axis=0)
                    x = range(1, len(means) + 1)

                    # Clean method name for legend
                    clean_method = method.replace(" - ", "-").replace(" + ", "+")

                    ax.plot(
                        x,
                        means,
                        color=color,
                        linewidth=2,
                        marker="o",
                        markersize=4,
                        label=clean_method,
                    )

                    # Optionally add std bands (uncomment if desired)
                    # ax.fill_between(x, means - stds, means + stds, color=color, alpha=0.2)

                else:
                    # Single run
                    x = range(1, len(method_data) + 1)
                    clean_method = method.replace(" - ", "-").replace(" + ", "+")
                    ax.plot(
                        x,
                        method_data,
                        color=color,
                        linewidth=2,
                        marker="o",
                        markersize=4,
                        label=clean_method,
                    )

        # Customize subplot
        ax.set_xlabel("Number of Classes", fontsize=10)
        ax.set_ylabel("Accuracy (%)", fontsize=10)
        ax.set_title(f"({chr(97+idx)}) {dataset}", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")

        # Show improvement if requested
        if show_improvement and dataset_methods:
            # Calculate improvement (difference between best and worst final performance)
            final_accuracies = []
            for method in dataset_methods:
                method_data = results[dataset][method]
                if method_data:
                    if isinstance(method_data[0], list):
                        # Multiple runs - use mean of final values
                        final_values = [run[-1] for run in method_data]
                        final_acc = np.mean(final_values)
                    else:
                        final_acc = method_data[-1]
                    final_accuracies.append(final_acc)

            if final_accuracies:
                improvement = max(final_accuracies) - min(final_accuracies)
                ax.text(
                    0.02,
                    0.98,
                    f"{improvement:.2f}↑",
                    transform=ax.transAxes,
                    fontsize=12,
                    fontweight="bold",
                    verticalalignment="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                )

    # Hide empty subplots
    for idx in range(len(available_datasets), nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].set_visible(False)

    plt.tight_layout()

    # Save figure
    plt.savefig(save_path, format="pdf", bbox_inches="tight", dpi=800)
    print(f"Figure saved as: {save_path}")
    plt.close()

    # Reset matplotlib parameters to default
    reset_plot_style()


def create_paper_style_line_plots(
    selected_datasets=None,
    selected_methods=None,
    figsize=(15, 10),
    ncols=3,
    save_path="figures/paper_style_plots.pdf",
):
    """
    Create line plots in the exact style of your paper figure with improvement annotations.

    Args:
        selected_datasets (list): List of dataset names to include
        selected_methods (list): List of method names to include
        figsize (tuple): Overall figure size
        ncols (int): Number of columns in the subplot grid
        save_path (str): Path to save the figure
    """
    if selected_datasets is None:
        # Six datasets excluding OB
        selected_datasets = ["CIFAR100", "IN-R", "IN-A", "CUB", "VTAB", "CARS"]

    if selected_methods is None:
        # All methods available, excluding CIL and CA variants
        all_methods = set()
        for dataset in selected_datasets:
            if dataset in results:
                all_methods.update(results[dataset].keys())

        # Filter out CIL, CA, MOS-LCA, SLCA-LCA, and CODA-Prompt methods
        filtered_methods = [
            m
            for m in sorted(all_methods)
            if not (
                m.endswith("- CIL")
                or m.endswith("- CA")
                or m == "MOS - LCA"
                or m == "SLCA - LCA"
                or m == "CODA-Prompt"
            )
        ]

        # Custom ordering: put SLCA, MOS, MM - LCA at the end in that order
        priority_methods = ["SLCA", "MOS", "MM - LCA"]

        # Separate priority methods from others
        other_methods = [m for m in filtered_methods if m not in priority_methods]
        present_priority = [m for m in priority_methods if m in filtered_methods]

        # Combine: other methods first (sorted), then priority methods in specified order
        selected_methods = other_methods + present_priority

    # Filter to only include datasets that exist in results
    available_datasets = [d for d in selected_datasets if d in results]

    if not available_datasets:
        print("No datasets found in results!")
        return

    # Set publication-ready style
    set_publication_style()

    # Calculate subplot layout
    nrows = (len(available_datasets) + ncols - 1) // ncols

    # Create figure with custom styling
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    # Ensure axes is always a 2D array
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    # Custom color mapping for cleaner legend
    method_name_mapping = {
        "L2P": "L2P",
        "DualPrompt": "DualPrompt",
        "EASE": "EASE",
        "SLCA": "SLCA",
        "MOS": "MOS",
        "MM - LCA": "IM+LCA",
        "APER + Adapter": "APER + Adapter",
        "APER + SSF": "APER + SSF",
        "APER + VPT-Deep": "APER + VPT-Deep",
        "APER + VPT-Shallow": "APER + VPT-Shallow",
        "APER + Finetune": "APER + Finetune",
    }

    # Plot each dataset
    for idx, dataset in enumerate(available_datasets):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]

        # Get methods available for this dataset
        dataset_methods = [m for m in selected_methods if m in results[dataset]]

        if not dataset_methods:
            ax.text(
                0.5,
                0.5,
                f"No data for\n{dataset}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            continue

        # Track performance for improvement calculation
        final_performances = []

        # Plot lines for each method
        for method in dataset_methods:
            method_data = results[dataset][method]
            color = get_method_color(method)

            if method_data:
                if isinstance(method_data[0], list):
                    # Multiple runs - plot mean
                    max_length = max(len(run) for run in method_data)

                    # Pad shorter runs with their last value
                    padded_runs = []
                    for run in method_data:
                        padded_run = list(run)
                        while len(padded_run) < max_length:
                            padded_run.append(padded_run[-1])
                        padded_runs.append(padded_run)

                    means = np.mean(padded_runs, axis=0)

                    # Dataset-specific x-axis scaling based on actual number of classes
                    if dataset == "CIFAR100":
                        # 100 classes total, incremental steps
                        x = np.arange(len(means)) * (100 // len(means)) + (
                            100 // len(means)
                        )
                    elif dataset == "VTAB":
                        # 50 classes total
                        x = np.arange(len(means)) * (50 // len(means)) + (
                            50 // len(means)
                        )
                    elif dataset == "CARS":
                        # 196 classes total, starting with 16, then incremental of 20
                        if len(means) > 1:
                            x = [16] + [16 + i * 20 for i in range(1, len(means))]
                        else:
                            x = [16]
                    else:
                        # Default scaling for other datasets
                        x = np.arange(len(means)) * (200 // len(means)) + (
                            200 // len(means)
                        )

                    display_name = method_name_mapping.get(method, method)

                    # Use thicker lines and larger markers for publication
                    ax.plot(
                        x,
                        means,
                        color=color,
                        linewidth=2.5,
                        marker="o",
                        markersize=8,
                        label=display_name,
                        alpha=0.8,
                    )

                    final_performances.append(means[-1])

                else:
                    # Single run
                    # Dataset-specific x-axis scaling for single runs
                    if dataset == "CIFAR100":
                        x = np.arange(len(method_data)) * (100 // len(method_data)) + (
                            100 // len(method_data)
                        )
                    elif dataset == "VTAB":
                        x = np.arange(len(method_data)) * (50 // len(method_data)) + (
                            50 // len(method_data)
                        )
                    elif dataset == "CARS":
                        if len(method_data) > 1:
                            x = [16] + [16 + i * 20 for i in range(1, len(method_data))]
                        else:
                            x = [16]
                    else:
                        x = np.arange(len(method_data)) * (200 // len(method_data)) + (
                            200 // len(method_data)
                        )

                    display_name = method_name_mapping.get(method, method)
                    ax.plot(
                        x,
                        method_data,
                        color=color,
                        linewidth=2.5,
                        marker="o",
                        markersize=8,
                        label=display_name,
                        alpha=0.8,
                    )

                    final_performances.append(method_data[-1])

        # Customize subplot with publication style
        ax.set_ylabel("Accuracy (%)")
        ax.set_xlabel("Number of Classes")

        # Dataset-specific title formatting
        title_mapping = {
            "CIFAR100": "CIFAR100",
            "CUB": "CUB",
            "IN-R": "IN-R",
            "IN-A": "IN-A",
            "OB": "OB",
            "VTAB": "VTAB",
            "CARS": "CARS",
        }

        dataset_title = title_mapping.get(dataset, dataset)

        # Put dataset name at the top with original styling
        ax.text(
            0.5,
            1.05,
            f"({chr(97+idx)}) {dataset_title}",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=20,
        )

        # Add grid matching bar chart style
        ax.grid(True, alpha=0.4, linewidth=1.5)
        ax.set_axisbelow(True)

        # Add legend with bigger text
        ax.legend(
            fontsize=12,
            loc="lower left",
            frameon=True,
            fancybox=True,
            shadow=False,
            framealpha=0.9,
            ncol=2,
            columnspacing=0.8,
            handletextpad=0.5,
        )

        # Make tick marks more visible like bar chart
        ax.tick_params(axis="both", which="major", length=8, width=2)

        # Set dataset-specific x-axis ticks to show correct class numbers
        if dataset == "CIFAR100":
            # Show key milestones for 100 classes
            ax.set_xticks([20, 40, 60, 80, 100])
        elif dataset == "VTAB":
            # Show key milestones for 50 classes
            ax.set_xticks([10, 20, 30, 40, 50])
        elif dataset == "CARS":
            # Show the actual class progression: 16, 36, 56, 76, 96, 116, 136, 156, 176, 196
            ax.set_xticks([36, 76, 116, 156, 196])
        # For other datasets (IN-R, IN-A, CUB), let matplotlib auto-choose

        # Set y-axis limits with proper scaling to show all methods
        if final_performances:
            # Get all y values from all plotted lines for better scaling
            all_y_values = []
            for method in dataset_methods:
                method_data = results[dataset][method]
                if method_data:
                    if isinstance(method_data[0], list):
                        # Multiple runs - get all values
                        for run in method_data:
                            all_y_values.extend(run)
                    else:
                        # Single run
                        all_y_values.extend(method_data)

            if all_y_values:
                y_min = min(all_y_values) - 3
                y_max = max(all_y_values) + 2
                ax.set_ylim(y_min, y_max)

    # Hide empty subplots
    for idx in range(len(available_datasets), nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].set_visible(False)

    # Adjust layout with proper spacing
    plt.tight_layout(pad=2.0)
    plt.subplots_adjust(top=0.92)  # Make room for top titles

    # Save figure with high quality matching bar chart
    plt.savefig(save_path, format="pdf", bbox_inches="tight", dpi=1200)
    print(f"Paper-style figure saved as: {save_path}")
    plt.close()

    # Reset matplotlib parameters to default
    reset_plot_style()


def create_alpha_ablation_plots(
    selected_datasets=None,
    figsize=(12, 5),
    save_path="figures/fig_alpha_ablation.pdf",
):
    """
    Create alpha ablation plots showing the effect of different alpha values.
    
    Args:
        selected_datasets (list): List of datasets to plot (default: ["IN-A", "VTAB"])
        figsize (tuple): Figure size (width, height)
        save_path (str): Path to save the figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Set publication style
    set_publication_style()
    
    if selected_datasets is None:
        selected_datasets = ["IN-A", "VTAB"]
    
    # Define consistent colors for selected lambda values only
    lambda_colors = {
        "alpha=0.1": "#ff7f0e",    # Orange
        "alpha=1.0": "#d62728",    # Red
        "alpha=10.0": "#8c564b",   # Brown
        "alpha=100.0": "#e377c2",  # Pink
    }
    
    # Only show these specific alpha values
    selected_alphas = ["alpha=0.1", "alpha=1.0", "alpha=10.0", "alpha=100.0"]
    
    # Create figure
    fig, axes = plt.subplots(1, len(selected_datasets), figsize=figsize)
    if len(selected_datasets) == 1:
        axes = [axes]
    
    # Plot each dataset
    for idx, dataset in enumerate(selected_datasets):
        ax = axes[idx]
        
        if dataset not in alpha_ablation:
            print(f"Warning: Dataset {dataset} not found in alpha_ablation data")
            continue
        
        # Plot accuracy curves for selected alpha values only
        for alpha_key in selected_alphas:
            if alpha_key not in alpha_ablation[dataset]:
                continue
                
            alpha_data = alpha_ablation[dataset][alpha_key]
            
            if "acc" not in alpha_data:
                continue
                
            accuracies = alpha_data["acc"]
            
            # Dataset-specific x-axis scaling based on actual number of classes
            if dataset == "IN-A":
                # 200 classes total (default scaling for IN-A)
                x = np.arange(len(accuracies)) * (200 // len(accuracies)) + (200 // len(accuracies))
            elif dataset == "VTAB":
                # 50 classes total
                x = np.arange(len(accuracies)) * (50 // len(accuracies)) + (50 // len(accuracies))
            else:
                # Default scaling for other datasets
                x = np.arange(len(accuracies)) * (200 // len(accuracies)) + (200 // len(accuracies))
            
            # Get color for this lambda value
            color = lambda_colors.get(alpha_key, "#000000")  # Default to black if not found
            
            # Create lambda label for display
            lambda_label = alpha_key.replace("alpha=", "λ=")
            
            # Plot the accuracy curve
            ax.plot(x, accuracies, 
                   color=color, 
                   linewidth=2.5, 
                   marker='o', 
                   markersize=4, 
                   label=lambda_label,
                   alpha=0.8)
        
        # Dataset-specific formatting (no bold)
        if dataset == "IN-A":
            ax.set_title("IN-A", fontsize=14, pad=15)
            ax.set_ylabel("Accuracy (%)", fontsize=12)
        elif dataset == "VTAB":
            ax.set_title("VTAB", fontsize=14, pad=15)
            ax.set_ylabel("Accuracy (%)", fontsize=12)
        
        # Common formatting (no bold)
        ax.set_xlabel("Number of Classes", fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Set appropriate x-axis limits and ticks based on dataset
        if dataset == "IN-A":
            # Start from first data point, not 0
            first_x = x[0]
            last_x = x[-1]
            ax.set_xlim(first_x - 5, last_x + 5)
            # Let matplotlib auto-choose ticks for IN-A (like in paper_style_line_plots)
        elif dataset == "VTAB":
            # Start from first data point, not 0  
            first_x = x[0]
            last_x = x[-1]
            ax.set_xlim(first_x - 5, last_x + 5)
            # Show key milestones for 50 classes
            # ax.set_xticks([10, 20, 30, 40, 50])
        
        # Tighten y-axis limits to remove extra space and prevent stick out
        if dataset == "IN-A":
            ax.set_ylim(68, 88)  # 20-point range for IN-A
        elif dataset == "VTAB":
            ax.set_ylim(90, 100)  # 8-point range for VTAB with more room
        
        # Add legend to both subplots (lower left)
        ax.legend(loc='lower left', fontsize=10, framealpha=0.9)
    
    # Adjust layout
    plt.tight_layout(pad=2.0)
    
    # Save figure
    plt.savefig(save_path, format="pdf", bbox_inches="tight", dpi=1200)
    print(f"Alpha ablation plots saved as: {save_path}")
    plt.close()
    
    # Reset matplotlib parameters to default
    reset_plot_style()


def create_backbone_ablation_plots(
    selected_datasets=None,
    figsize=(8, 5),
    save_path="figures/fig_backbone_ablation.pdf",
):
    """
    Create backbone ablation plots showing the effect of different backbone models.
    
    Args:
        selected_datasets (list): List of datasets to plot (default: ["VTAB"])
        figsize (tuple): Figure size (width, height)
        save_path (str): Path to save the figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Set publication style
    set_publication_style()
    
    if selected_datasets is None:
        selected_datasets = ["VTAB"]  # Can be expanded to ["VTAB", "CIFAR100", "CUB"] etc.
    
    # Define backbone name mapping
    backbone_name_mapping = {
        "vit_b16_224_ssf_cil": "ViT-B16/IN1K - SSF - IM",
        "vit_b16_224_ssf_lca": "ViT-B16/IN1K - SSF - IM+LCA",
        "vit_b16_224_in21k_ssf_cil": "ViT-B16/IN21K - SSF - IM",
        "vit_b16_224_in21k_ssf_lca": "ViT-B16/IN21K - SSF - IM+LCA",

        "vit_b16_224_vpt_cil": "ViT-B16/IN1K - VPT - IM",
        "vit_b16_224_vpt_lca": "ViT-B16/IN1K - VPT - IM+LCA",
        "vit_b16_224_in21k_vpt_cil": "ViT-B16/IN21K - VPT - IM",
        "vit_b16_224_in21k_vpt_lca": "ViT-B16/IN21K - VPT - IM+LCA",

        "vit_b16_224_adapter_cil": "ViT-B16/IN1K - Adapter - IM",
        "vit_b16_224_adapter_lca": "ViT-B16/IN1K - Adapter - IM+LCA",
        "vit_b16_224_in21k_adapter_cil": "ViT-B16/IN21K - Adapter - IM",
        "vit_b16_224_in21k_adapter_lca": "ViT-B16/IN21K - Adapter - IM+LCA",

        "vit_b16_224_lora_cil": "ViT-B16/IN1K - LoRA - IM",
        "vit_b16_224_lora_lca": "ViT-B16/IN1K - LoRA - IM+LCA",
        "vit_b16_224_in21k_lora_cil": "ViT-B16/IN21K - LoRA - IM",
        "vit_b16_224_in21k_lora_lca": "ViT-B16/IN21K - LoRA - IM+LCA",
    }
    
    # Define consistent colors for backbone variants - grouped by method
    backbone_colors = {
        # SSF variants - Blue family
        "vit_b16_224_ssf_lca": "#1f77b4",        # Dark Blue - IM+LCA
        "vit_b16_224_ssf_cil": "#87CEEB",        # Light Blue - IM
        "vit_b16_224_in21k_ssf_lca": "#4682B4",  # Steel Blue - IM+LCA
        "vit_b16_224_in21k_ssf_cil": "#B0E0E6",  # Powder Blue - IM
        
        # VPT variants - Green family
        "vit_b16_224_vpt_lca": "#2ca02c",        # Dark Green - IM+LCA
        "vit_b16_224_vpt_cil": "#90EE90",        # Light Green - IM
        "vit_b16_224_in21k_vpt_lca": "#228B22",  # Forest Green - IM+LCA
        "vit_b16_224_in21k_vpt_cil": "#98FB98",  # Pale Green - IM
        
        # Adapter variants - Purple family
        "vit_b16_224_adapter_lca": "#9467bd",    # Dark Purple - IM+LCA
        "vit_b16_224_adapter_cil": "#DDA0DD",    # Plum - IM
        "vit_b16_224_in21k_adapter_lca": "#8B008B", # Dark Magenta - IM+LCA
        "vit_b16_224_in21k_adapter_cil": "#DA70D6", # Orchid - IM
        
        # LoRA variants - Orange/Red family
        "vit_b16_224_lora_lca": "#ff7f0e",       # Orange - IM+LCA
        "vit_b16_224_lora_cil": "#FFA07A",       # Light Salmon - IM
        "vit_b16_224_in21k_lora_lca": "#FF4500", # Orange Red - IM+LCA
        "vit_b16_224_in21k_lora_cil": "#FFB07A", # Light Orange - IM
    }
    
    # Create figure
    fig, axes = plt.subplots(1, len(selected_datasets), figsize=figsize)
    if len(selected_datasets) == 1:
        axes = [axes]
    
    # Plot each dataset
    for idx, dataset in enumerate(selected_datasets):
        ax = axes[idx]
        
        if dataset not in backbone_ablation:
            print(f"Warning: Dataset {dataset} not found in backbone_ablation data")
            continue
        
        # Get backbone variants
        backbone_variants = list(backbone_ablation[dataset].keys())
        
        # Plot accuracy curves for each backbone variant
        for backbone_key in backbone_variants:
            backbone_data = backbone_ablation[dataset][backbone_key]
            
            if "acc" not in backbone_data:
                continue
                
            accuracies = backbone_data["acc"]
            
            # Dataset-specific x-axis scaling based on actual number of classes
            if dataset == "VTAB":
                # 50 classes total
                x = np.arange(len(accuracies)) * (50 // len(accuracies)) + (50 // len(accuracies))
            else:
                # Default scaling for other datasets
                x = np.arange(len(accuracies)) * (200 // len(accuracies)) + (200 // len(accuracies))
            
            # Get color for this backbone variant
            color = backbone_colors.get(backbone_key, "#000000")  # Default to black if not found
            
            # Get display name
            display_name = backbone_name_mapping.get(backbone_key, backbone_key)
            
            # Plot the accuracy curve
            ax.plot(x, accuracies, 
                   color=color, 
                   linewidth=2.5, 
                   marker='o', 
                   markersize=4, 
                   label=display_name,
                   alpha=0.8)
        
        # Dataset-specific formatting (no bold)
        if dataset == "VTAB":
            ax.set_title("VTAB", fontsize=14, pad=15)
            ax.set_ylabel("Accuracy (%)", fontsize=12)
        elif dataset == "CIFAR100":
            ax.set_title("CIFAR100", fontsize=14, pad=15)
            ax.set_ylabel("Accuracy (%)", fontsize=12)
        elif dataset == "CUB":
            ax.set_title("CUB", fontsize=14, pad=15)
            ax.set_ylabel("Accuracy (%)", fontsize=12)
        else:
            ax.set_title(dataset, fontsize=14, pad=15)
            ax.set_ylabel("Accuracy (%)", fontsize=12)
        
        # Common formatting (no bold)
        ax.set_xlabel("Number of Classes", fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Set appropriate x-axis limits and ticks based on dataset
        if dataset == "VTAB":
            # Start from first data point, not 0  
            first_x = x[0]
            last_x = x[-1]
            ax.set_xlim(first_x - 2, last_x + 2)
            # Show key milestones for 50 classes
            ax.set_xticks([10, 20, 30, 40, 50])
        elif dataset == "CIFAR100":
            # Start from first data point, not 0  
            first_x = x[0]
            last_x = x[-1] 
            ax.set_xlim(first_x - 5, last_x + 5)
            # Show key milestones for 100 classes
            ax.set_xticks([20, 40, 60, 80, 100])
        elif dataset == "CUB":
            # Start from first data point, not 0  
            first_x = x[0]
            last_x = x[-1]
            ax.set_xlim(first_x - 5, last_x + 5)
            # Show key milestones for 200 classes  
            ax.set_xticks([40, 80, 120, 160, 200])
        
        # Adjust y-axis limits to provide more space and prevent value blocking
        if dataset == "VTAB":
            ax.set_ylim(60, 102)  # Extended range for VTAB to accommodate methods falling to 65%
        elif dataset == "CIFAR100":
            ax.set_ylim(60, 102)  # Extended range for CIFAR100
        elif dataset == "CUB": 
            ax.set_ylim(60, 102)  # Extended range for CUB
        else:
            # Auto-scale for other datasets with extra margin
            all_accuracies = []
            for backbone_key in backbone_variants:
                backbone_data = backbone_ablation[dataset][backbone_key]
                if "acc" in backbone_data:
                    all_accuracies.extend(backbone_data["acc"])
            if all_accuracies:
                margin = (max(all_accuracies) - min(all_accuracies)) * 0.15
                y_min = max(60, min(all_accuracies) - margin)  # Don't go below 60%
                ax.set_ylim(y_min, max(all_accuracies) + margin + 2)
        
        # Add legend to subplot with better spacing for CIL variants
        num_methods = len([k for k in backbone_variants if k in backbone_name_mapping])
        
        # Group legend entries by method type for better visualization
        legend_handles = []
        legend_labels = []
        
        # Sort backbone variants by method type for grouped legend
        method_order = ['ssf', 'vpt', 'adapter', 'lora']
        sorted_variants = []
        for method in method_order:
            for variant in backbone_variants:
                if method in variant and variant in backbone_name_mapping:
                    sorted_variants.append(variant)
        
        # Create custom legend entries
        for variant in sorted_variants:
            if variant in backbone_name_mapping:
                # Find the line object for this variant
                for line in ax.get_lines():
                    if line.get_label() == backbone_name_mapping[variant]:
                        legend_handles.append(line)
                        # Simplify labels for better readability
                        simplified_label = backbone_name_mapping[variant].replace("ViT-B16/", "").replace(" - ", "-")
                        legend_labels.append(simplified_label)
                        break
        
        # Position legend at bottom left inside the figure
        if len(legend_handles) > 0:
            ax.legend(legend_handles, legend_labels, 
                     loc='lower left', 
                     fontsize=8, framealpha=0.9, ncol=2)
    
    # Adjust layout to accommodate external legend
    plt.tight_layout(pad=2.0)
    
    # Save figure with bbox_inches to include external legend
    plt.savefig(save_path, format="pdf", bbox_inches="tight", dpi=800)
    print(f"Backbone ablation plots saved as: {save_path}")
    plt.close()
    
    # Reset matplotlib parameters to default
    reset_plot_style()


def create_mos_slca_comparison_bar_plot(
    selected_datasets=None,
    figsize=(14, 10),
    save_path="figures/fig_mos_slca_comparison.pdf",
):
    """
    Create bar plot comparing MOS and SLCA variants across all datasets.
    
    Args:
        selected_datasets (list): List of datasets to plot (default: all datasets)
        figsize (tuple): Figure size (width, height)
        save_path (str): Path to save the figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Set publication style
    set_publication_style()
    
    if selected_datasets is None:
        selected_datasets = DATASETS
    
    # Define the methods to compare (omitting original MOS and SLCA, and CA variants)
    methods_to_compare = ["MOS - CIL", "MOS - LCA", 
                         "SLCA - CIL", "SLCA - LCA"]
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Prepare data
    dataset_names = []
    method_data = {method: [] for method in methods_to_compare}
    
    for dataset in selected_datasets:
        if dataset not in results:
            continue
            
        dataset_names.append(dataset)
        
        for method in methods_to_compare:
            if method in results[dataset]:
                # Calculate mean performance from the results
                method_results = results[dataset][method]
                if method_results:
                    if isinstance(method_results[0], list):
                        # Multiple runs - calculate mean of final accuracies
                        final_accs = [run[-1] for run in method_results]
                        mean_acc = np.mean(final_accs)
                    else:
                        # Single run - take final accuracy
                        mean_acc = method_results[-1]
                    method_data[method].append(mean_acc)
                else:
                    method_data[method].append(0)  # No data available
            else:
                method_data[method].append(0)  # Method not found
    
    # Set up bar positions
    x = np.arange(len(dataset_names))
    bar_width = 0.2
    n_methods = len(methods_to_compare)
    
    # Define legend name mapping
    legend_names = {
        "MOS - CIL": "MOS",
        "MOS - LCA": "MOS - LCA",
        "SLCA - CIL": "SLCA",
        "SLCA - LCA": "SLCA - LCA",
    }
    
    # Create bars for each method
    for i, method in enumerate(methods_to_compare):
        offset = (i - n_methods/2 + 0.5) * bar_width
        color = get_method_color(method)
        
        # Use mapped name for legend
        legend_label = legend_names.get(method, method)
        
        bars = ax.bar(x + offset, method_data[method], bar_width, 
                     label=legend_label, color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Add value labels on top of bars (horizontal and bold)
        for j, (bar, value) in enumerate(zip(bars, method_data[method])):
            if value > 0:  # Only show labels for non-zero values
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                       f'{value:.1f}', ha='center', va='bottom', fontsize=20, 
                       rotation=90, fontweight='bold')
    
    # Customize the plot
    ax.set_xlabel("Datasets", fontsize=22)
    ax.set_ylabel("Accuracy (%)", fontsize=22)
    # ax.set_title("MOS vs SLCA Variants Comparison Across Datasets", fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names, fontsize=20)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
    ax.set_axisbelow(True)
    
    # Add legend with custom names
    ax.legend(loc='upper right', fontsize=16, framealpha=0.9)
    
    # Set y-axis limits with some margin
    all_values = []
    for method_values in method_data.values():
        all_values.extend([v for v in method_values if v > 0])
    
    if all_values:
        y_min = min(all_values) - 5
        y_max = max(all_values) + 5
        ax.set_ylim(y_min, 100)
    ax.tick_params(axis="y", labelsize=20)
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, format="pdf", bbox_inches="tight", dpi=800)
    print(f"MOS vs SLCA comparison bar plot saved as: {save_path}")
    plt.close()
    
    # Reset matplotlib parameters to default
    reset_plot_style()


if __name__ == "__main__":
    # Demo 1: Create bar chart
    # create_lca_performance_bar_chart()

    # create_paper_style_line_plots(
    #     selected_datasets=["CIFAR100", "IN-R", "IN-A", "CUB", "VTAB", "CARS"],
    #     selected_methods=None,  # Will auto-filter to exclude unwanted methods
    #     figsize=(18, 12),
    #     ncols=3,
    #     save_path="figures/fig_performance_curve.pdf",
    # )

    # # Create alpha ablation plots
    # create_alpha_ablation_plots(
    #     selected_datasets=["IN-A", "VTAB"],
    #     figsize=(12, 5),
    #     save_path="figures/fig_lambda_ablation.pdf",
    # )

    # # Create backbone ablation plots
    # create_backbone_ablation_plots(
    #     selected_datasets=["VTAB"],
    #     figsize=(8, 5),
    #     save_path="figures/fig_backbone_ablation.pdf",
    # )

    # Create MOS vs SLCA comparison bar plot
    create_mos_slca_comparison_bar_plot(
        selected_datasets=DATASETS,
        save_path="figures/fig_mos_slca_comparison.pdf",
    )

    # # Demo 3: Print LaTeX table
    # print_latex_table(datasets=None, methods=None, stat_type="final")
