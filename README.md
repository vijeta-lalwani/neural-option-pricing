
# Neural Network Approximation for Option Pricing

This project demonstrates how a simple neural network can approximate the
Black-Scholes pricing function for European call options.

## What it does

- Implements the Black-Scholes formula
- Generates synthetic option data with NumPy and Pandas
- Trains a feedforward neural network using PyTorch
- Evaluates the model with test-set MSE
- Plots predicted vs actual prices
- Benchmarks Black-Scholes pricing vs neural-network inference

## Files

- `option_pricing_nn.py`: main script
- `requirements.txt`: Python dependencies

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 option_pricing_nn.py
```

The script saves a scatter plot to `predicted_vs_actual.png` and also shows
the training loss curve in `training_loss.png`.
=======
# neural-option-pricing
A PyTorch project that uses a feedforward neural network to approximate Black-Scholes European call option pricing.
