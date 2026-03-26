"""
Simple neural-network approximation of Black-Scholes option pricing.

This script does five things:
1. Computes European call prices with the Black-Scholes formula.
2. Generates a synthetic dataset of option parameters and prices.
3. Trains a small PyTorch neural network to learn that pricing function.
4. Evaluates the model on a held-out test set.
5. Benchmarks Black-Scholes pricing against neural-network inference.

The code is intentionally simple so it is easy to explain in an interview.
"""

import os
from math import erf
from time import perf_counter

# Point Matplotlib config to a writable local folder so the script runs cleanly
# in restricted environments such as sandboxes and CI.
os.environ.setdefault("MPLCONFIGDIR", ".matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# -----------------------------
# Configuration
# -----------------------------
# Keeping configuration in one place makes the script easy to tweak.
RANDOM_SEED = 42
N_SAMPLES = 20_000
TEST_SIZE = 0.2
BATCH_SIZE = 256
EPOCHS = 150
LEARNING_RATE = 1e-3
HIDDEN_DIM = 64
PLOT_FILE = "predicted_vs_actual.png"
LOSS_PLOT_FILE = "training_loss.png"


np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# -----------------------------
# Data generation
# -----------------------------
def normal_cdf(x: np.ndarray) -> np.ndarray:
    """
    Standard normal cumulative distribution function.

    Black-Scholes uses the normal CDF, so we implement it directly here.
    """
    return 0.5 * (1.0 + np.vectorize(erf)(x / np.sqrt(2.0)))


def black_scholes_call_price(
    stock_price: np.ndarray,
    strike_price: np.ndarray,
    time_to_maturity: np.ndarray,
    risk_free_rate: np.ndarray,
    volatility: np.ndarray,
) -> np.ndarray:
    """
    Compute European call option prices using the Black-Scholes formula.

    Inputs can be NumPy arrays, which lets us price many options at once.
    """
    # Small floor values avoid division-by-zero or log-of-zero issues.
    stock_price = np.maximum(stock_price, 1e-8)
    strike_price = np.maximum(strike_price, 1e-8)
    time_to_maturity = np.maximum(time_to_maturity, 1e-8)
    volatility = np.maximum(volatility, 1e-8)

    sqrt_t = np.sqrt(time_to_maturity)
    d1 = (
        np.log(stock_price / strike_price)
        + (risk_free_rate + 0.5 * volatility**2) * time_to_maturity
    ) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t

    call_price = (
        stock_price * normal_cdf(d1)
        - strike_price * np.exp(-risk_free_rate * time_to_maturity) * normal_cdf(d2)
    )
    return call_price


def generate_option_dataset(n_samples: int) -> pd.DataFrame:
    """
    Create a synthetic dataset of option inputs and Black-Scholes prices.

    The parameter ranges are broad enough to give the model varied examples.
    """
    stock_price = np.random.uniform(50, 150, n_samples)
    strike_price = np.random.uniform(50, 150, n_samples)
    time_to_maturity = np.random.uniform(0.1, 2.0, n_samples)
    risk_free_rate = np.random.uniform(0.0, 0.1, n_samples)
    volatility = np.random.uniform(0.1, 0.6, n_samples)

    option_price = black_scholes_call_price(
        stock_price,
        strike_price,
        time_to_maturity,
        risk_free_rate,
        volatility,
    )

    return pd.DataFrame(
        {
            "S": stock_price,
            "K": strike_price,
            "T": time_to_maturity,
            "r": risk_free_rate,
            "sigma": volatility,
            "price": option_price,
        }
    )


# -----------------------------
# Model definition
# -----------------------------
class OptionPricingNN(nn.Module):
    """
    Small feedforward neural network.

    It takes five inputs: S, K, T, r, sigma
    and returns one output: the predicted option price.
    """

    def __init__(self, input_dim: int = 5, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# -----------------------------
# Training loop
# -----------------------------
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    learning_rate: float,
) -> list:
    """
    Train the neural network and return the history of training losses.
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_history = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for batch_features, batch_targets in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_features)
            loss = criterion(predictions, batch_targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_features.size(0)

        average_loss = epoch_loss / len(train_loader.dataset)
        loss_history.append(average_loss)

        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1:3d}/{epochs} - Training Loss: {average_loss:.6f}")

    return loss_history


# -----------------------------
# Evaluation
# -----------------------------
def evaluate_model(
    model: nn.Module,
    features_test: np.ndarray,
    targets_test: np.ndarray,
) -> tuple:
    """
    Run the model on the test set and compute mean squared error.
    """
    model.eval()
    with torch.no_grad():
        test_tensor = torch.tensor(features_test, dtype=torch.float32)
        predicted_prices = model(test_tensor).cpu().numpy().ravel()

    mse = mean_squared_error(targets_test, predicted_prices)
    return predicted_prices, mse


def plot_results(actual_prices: np.ndarray, predicted_prices: np.ndarray, output_path: str) -> None:
    """
    Plot predicted prices against actual Black-Scholes prices.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(actual_prices, predicted_prices, alpha=0.5, s=14, label="Predictions")

    min_price = min(actual_prices.min(), predicted_prices.min())
    max_price = max(actual_prices.max(), predicted_prices.max())
    plt.plot([min_price, max_price], [min_price, max_price], "r--", label="Perfect fit")

    plt.xlabel("Actual Black-Scholes Price")
    plt.ylabel("Neural Network Predicted Price")
    plt.title("Predicted vs Actual Option Prices")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_training_loss(loss_history: list, output_path: str) -> None:
    """
    Plot how the training loss changes over time.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss Over Time")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# -----------------------------
# Benchmarking
# -----------------------------
def benchmark_black_scholes(
    stock_price: np.ndarray,
    strike_price: np.ndarray,
    time_to_maturity: np.ndarray,
    risk_free_rate: np.ndarray,
    volatility: np.ndarray,
) -> float:
    """
    Measure how long it takes to price the test set with Black-Scholes.
    """
    start_time = perf_counter()
    _ = black_scholes_call_price(
        stock_price,
        strike_price,
        time_to_maturity,
        risk_free_rate,
        volatility,
    )
    end_time = perf_counter()
    return end_time - start_time


def benchmark_neural_network(model: nn.Module, features_test: np.ndarray) -> float:
    """
    Measure how long it takes for the neural network to make predictions.
    """
    model.eval()
    test_tensor = torch.tensor(features_test, dtype=torch.float32)

    start_time = perf_counter()
    with torch.no_grad():
        _ = model(test_tensor)
    end_time = perf_counter()

    return end_time - start_time


def main() -> None:
    """
    Run the full workflow end-to-end.
    """
    print("Generating synthetic option pricing data...")
    data = generate_option_dataset(N_SAMPLES)

    feature_columns = ["S", "K", "T", "r", "sigma"]
    target_column = "price"

    features = data[feature_columns].values
    targets = data[target_column].values

    # We scale inputs because neural networks train better when features
    # are on similar ranges.
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        targets,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    train_dataset = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32),
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    print("Training neural network...")
    model = OptionPricingNN(input_dim=5, hidden_dim=HIDDEN_DIM)
    loss_history = train_model(
        model=model,
        train_loader=train_loader,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
    )

    print("Evaluating model on test set...")
    predicted_prices, test_mse = evaluate_model(model, X_test_scaled, y_test)
    print(f"Test MSE: {test_mse:.6f}")

    plot_results(y_test, predicted_prices, PLOT_FILE)
    print(f"Saved predicted-vs-actual plot to: {PLOT_FILE}")
    plot_training_loss(loss_history, LOSS_PLOT_FILE)
    print(f"Saved training loss plot to: {LOSS_PLOT_FILE}")

    # Use the original unscaled test inputs for Black-Scholes timing,
    # because the formula expects raw financial inputs.
    bs_runtime = benchmark_black_scholes(
        stock_price=X_test[:, 0],
        strike_price=X_test[:, 1],
        time_to_maturity=X_test[:, 2],
        risk_free_rate=X_test[:, 3],
        volatility=X_test[:, 4],
    )
    nn_runtime = benchmark_neural_network(model, X_test_scaled)

    print(f"Black-Scholes runtime on test set: {bs_runtime:.8f} seconds")
    print(f"Neural network inference runtime on test set: {nn_runtime:.8f} seconds")

    # Printing a few example predictions is useful in interviews because it
    # shows the network is learning the mapping, not just reporting one score.
    results = pd.DataFrame(
        {
            "actual_price": y_test[:10],
            "predicted_price": predicted_prices[:10],
        }
    )
    print("\nSample predictions:")
    print(results.round(4).to_string(index=False))

if __name__ == "__main__":
    main()
