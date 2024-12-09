def measure_performance(query_results, optimized_results):
    """
    Calculate Percent Optimized (OPT) and Speedup Rate (SP) metrics.

    Args:
        query_results (list of dict): Results from the original code.
        optimized_results (list of dict): Results from the optimized code.

    Returns:
        dict: A dictionary containing:
            - 'OPT': Percent of optimized programs that meet criteria.
            - 'SP': Speedup rate across all test cases.
    """
    n = len(query_results)
    if n == 0:
        return {"OPT": 0, "SP": 0}

    optimized_count = 0
    total_speedup = 0

    for query_result, optimized_result in zip(query_results, optimized_results):
        correct_query = query_result["correct"]
        correct_optimized = optimized_result["correct"]
        runtime_query = query_result["runtime_ms"]
        runtime_optimized = optimized_result["runtime_ms"]

        # Check if optimized code is correct and at least 10% faster
        if correct_optimized and runtime_optimized < runtime_query * 0.9:
            optimized_count += 1

        # Speedup calculation
        if correct_optimized and runtime_optimized <= runtime_query:
            speedup = runtime_query / runtime_optimized
        else:
            speedup = 1.0  # Worst-case scenario, no improvement

        total_speedup += speedup

    # Calculate metrics
    OPT = (optimized_count / n) * 100  # Percent Optimized
    SP = total_speedup / n  # Average Speedup Rate

    return {"OPT": OPT, "SP": SP}