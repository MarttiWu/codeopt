def measure_performance(results):
    """
    Calculate Percent Optimized (OPT) and Speedup Rate (SP) metrics.

    Args:
        results (list of dict): A list of dictionaries, where each dictionary contains:
            - 'correct': Boolean indicating if the test case passed.
            - 'original_runtime': Runtime of the original code (in ms).
            - 'optimized_runtime': Runtime of the optimized code (in ms).

    Returns:
        dict: A dictionary with the metrics:
            - 'OPT': Percent of optimized programs that meet criteria.
            - 'SP': Speedup rate across all test cases.
    """
    n = len(results)
    if n == 0:
        return {"OPT": 0, "SP": 0}

    total_speedup = 0
    optimized_count = 0

    for result in results:
        original_runtime = result["original_runtime"]
        optimized_runtime = result["optimized_runtime"]
        correct = result["correct"]

        # Calculate Percent Optimized (OPT)
        if correct and optimized_runtime < original_runtime * 0.9:
            optimized_count += 1

        # Calculate Speedup Rate (SP)
        if correct and optimized_runtime <= original_runtime:
            speedup = original_runtime / optimized_runtime
        else:
            speedup = 1.0  # Worst-case speedup

        total_speedup += speedup

    # Compute metrics
    OPT = (optimized_count / n) * 100
    SP = total_speedup / n

    return {"OPT": OPT, "SP": SP}