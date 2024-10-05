# utils/data_generation.py
import numpy as np

def generate_data(num_samples=1000, seed=42):
    np.random.seed(seed)
    throughput_data = np.sin(np.linspace(0, 50, num_samples)) + np.random.normal(0, 0.2, num_samples)
    return throughput_data
