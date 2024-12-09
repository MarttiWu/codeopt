import logging
import signal
import time
import tracemalloc

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Code execution exceeded the time limit!")

def execute_code(code_str, globals_dict=None, locals_dict=None, timeout=30):
    if globals_dict is None:
        globals_dict = {}
    if locals_dict is None:
        locals_dict = {}

    # Start tracking
    start_time = time.time()
    tracemalloc.start()

    # Set up the timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)  # Set the timeout to the specified seconds

    try:
        print("executing...")
        # Execute the code
        exec(code_str, globals_dict, locals_dict)
        # Get the result of the last executed statement, if relevant
        result = locals_dict.get("_result", None)
        # Clear the alarm if execution finishes within the time limit
        signal.alarm(0)
        
    except TimeoutException:
        result = "Timeout: Code execution exceeded the time limit!"
    except Exception as e:
        result = f"Error executing code: {e}"

    # Stop tracking
    current, peak = tracemalloc.get_traced_memory()
    end_time = time.time()

    # Calculate runtime
    runtime = min((end_time - start_time) * 1000, timeout * 1000)  # Convert to milliseconds
    tracemalloc.stop()

    # Return the results
    return result, runtime, current

import json

def normalize_code(code):
    """Normalize the code by removing leading/trailing spaces and ensuring consistent newlines."""
    return '\n'.join(line.strip() for line in code.splitlines()).strip()

def find_matching_code(file_path, target_input):
    """Find the matching 'problem_id' for the given code input."""
    normalized_target = normalize_code(target_input)
    results = []
    
    try:
        with open(file_path, 'r') as file:
            for line in file:
                # Parse each JSON line
                data = json.loads(line)
                # Normalize the code for comparison
                for key in ['code_v0_no_empty_lines', 'code_v1_no_empty_lines']:
                    if key in data:
                        if normalize_code(data[key]) == normalized_target:
                            results.append(data.get('problem_id'))
    except FileNotFoundError:
        return "Error: File not found!"
    except json.JSONDecodeError:
        return "Error: Malformed JSON in file!"
    except Exception as e:
        return f"Error: {e}"

    return results or "No matches found"