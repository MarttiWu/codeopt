

def speed_up_rate(measured_runtime_v0, measured_runtime_v1):
    return max(measured_runtime_v0 / measured_runtime_v1, 1.0)


def measure_performance(result, original_runtime , runtime, memory):
    
    
    return result, runtime, memory