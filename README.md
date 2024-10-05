# LSTM_Throughput_Prediction

# LSTM-Based Throughput Prediction for LTE Networks

## Overview

This repository contains an implementation of an LSTM-based model with an attention mechanism to predict LTE network throughput. The code is structured into modular components, focusing on data preparation, model training, evaluation, and results visualization.

## Features
- Sliding window data preparation for time-series throughput data.
- Attention-based LSTM model for better prediction accuracy.
- Modular design with separate files for dataset handling, model definition, and result visualization.

## Project Structure

LSTM_Throughput_Prediction
│
├── /models              # Contains the LSTM model with attention mechanism
│   └── attention_lstm.py
│
├── /utils               # Helper functions and data processing
│   └── dataset.py
│   └── data_generation.py
│   └── plot_results.py
│
├── main.py              # Main script for training and evaluation
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation


## Usage

1. Clone the repository:

git clone https://github.com/yourusername/LSTM_Throughput_Prediction.git
cd LSTM_Throughput_Prediction

2. Install dependencies:

pip install -r requirements.txt

