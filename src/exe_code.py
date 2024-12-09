import logging
import signal
import time
import tracemalloc
import os
from unittest.mock import patch
from io import StringIO
import sys


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Code execution exceeded the time limit!")


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

        # Set up the timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)  # Set the timeout to the specified seconds

        try:
            logging.info(f"Executing with input from {input_file}...")

            # Redirect stdout to capture printed output
            captured_output = StringIO()
            sys.stdout = captured_output

            # Mock the input function for the exec environment
            with patch("builtins.input", side_effect=test_input):
                exec(code_str, globals_dict, locals_dict)

            # Capture printed output
            result = captured_output.getvalue().strip()

            # Clear the alarm if execution finishes within the time limit
            signal.alarm(0)

            # Compare the result with the expected output
            correct = (result == expected_output)

        except TimeoutException:
            correct = False
            logging.error("Timeout: Code execution exceeded the time limit!")
            result = None
        except Exception as e:
            correct = False
            logging.error(
                f"Error executing code:\n{code_str}\n"
                f"With input from {input_file}.\n"
                f"Input: {test_input}\nExpected output: {expected_output}\nError: {e}",
                exc_info=True
            )
            result = None
        finally:
            # Stop tracking memory usage
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # Restore original stdout
            sys.stdout = sys.__stdout__

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