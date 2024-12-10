# executor.py
import logging
import multiprocessing
import tracemalloc
import time
import os
from unittest.mock import patch
from io import StringIO
import sys

class TimeoutException(Exception):
    pass

def execute_code(code_str, test_input, queue):
    try:
        # Redirect stdout to capture output
        captured_output = StringIO()
        sys.stdout = captured_output

        # Mock input
        with patch("builtins.input", side_effect=test_input):
            exec(code_str, {}, {})  # Empty globals and locals

        # Capture output
        result = captured_output.getvalue().strip()
        sys.stdout = sys.__stdout__
        queue.put((result, True, None))
    except Exception as e:
        sys.stdout = sys.__stdout__
        queue.put((None, False, str(e)))

def execute_code_with_test_cases(code_str, problem_id, test_cases_path, globals_dict=None, locals_dict=None, timeout=5):
    """
    Executes the given code with provided test cases and validates the outputs.

    Args:
        code_str (str): Code to execute.
        problem_id (str): Problem folder ID under test_cases_path.
        test_cases_path (str): Path to the test cases directory.
        globals_dict (dict): Global variables for the exec environment.
        locals_dict (dict): Local variables for the exec environment.
        timeout (int): Timeout for code execution in seconds.

    Returns:
        list: A list of dictionaries with test case results.
    """
    if globals_dict is None:
        globals_dict = {}
    if locals_dict is None:
        locals_dict = {}

    problem_folder = os.path.join(test_cases_path, problem_id)
    if not os.path.exists(problem_folder):
        raise FileNotFoundError(f"Problem folder {problem_folder} does not exist.")

    # Collect input and output files
    input_files = sorted([f for f in os.listdir(problem_folder) if f.startswith("input")])
    output_files = sorted([f for f in os.listdir(problem_folder) if f.startswith("output")])

    results = []

    for input_file, output_file in zip(input_files, output_files):
        input_path = os.path.join(problem_folder, input_file)
        output_path = os.path.join(problem_folder, output_file)

        # Read the input and output
        with open(input_path, "r") as f:
            test_input = f.read().strip().splitlines()

        with open(output_path, "r") as f:
            expected_output = f.read().strip()

        # Prepare for execution
        start_time = time.time()
        tracemalloc.start()

        # Set up the multiprocessing Queue
        queue = multiprocessing.Queue()
        process = multiprocessing.Process(target=execute_code, args=(code_str, test_input, queue))
        process.start()
        process.join(timeout)

        if process.is_alive():
            process.terminate()
            process.join()
            correct = False
            result = None
            error = "Timeout: Code execution exceeded the time limit!"
            logging.error(error)
        else:
            if not queue.empty():
                result, correct, error = queue.get()
                if correct:
                    logging.info("Execution successful.")
                else:
                    logging.error(f"Execution failed: {error}")
            else:
                correct = False
                result = None
                error = "No output received."

        # Stop tracking memory usage
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Calculate runtime
        end_time = time.time()
        runtime = min((end_time - start_time) * 1000, timeout * 1000)  # Convert to milliseconds

        # Append result
        results.append({
            "input_file": input_file,
            "output_file": output_file,
            "correct": correct,
            "runtime_ms": runtime,
            "memory_peak_kb": peak / 1024,
            "result": result,  # Captured printed output
            "expected_output": expected_output  # Expected output
        })

    return results
