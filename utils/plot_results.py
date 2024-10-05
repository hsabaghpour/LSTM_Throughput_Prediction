# utils/plot_results.py
import matplotlib.pyplot as plt

def plot_results(actual, predicted):
    plt.plot(actual, label='Actual Throughput')
    plt.plot(predicted, label='Predicted Throughput')
    plt.legend()
    plt.title("Throughput Prediction with Attention-based LSTM")
    plt.show()
