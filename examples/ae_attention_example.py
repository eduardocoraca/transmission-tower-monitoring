"""
Example usage of AERegularizedAttention model.

This script demonstrates how to use the attention-based autoencoder with masking
for reconstruction tasks.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

from training.models.ae_regularized_attention import AERegularizedAttention


def create_dummy_data(
    batch_size: int = 100,
    num_cables: int = 4,
    input_dim: int = 256,
    num_samples: int = 500,
):
    """
    Create dummy data that simulates the expected input format.

    Args:
        batch_size: Number of samples per batch
        num_cables: Number of cables (expected to be 4)
        input_dim: Dimension of each cable's feature vector
        num_samples: Total number of samples to generate

    Returns:
        DataLoader with dummy data
    """
    # Generate synthetic data with some structure
    np.random.seed(42)
    torch.manual_seed(42)

    # Create some underlying "modes" that cables should share
    num_modes = 5
    mode_weights = torch.randn(num_samples, num_modes)

    # Each cable has slightly different representation of the same modes
    cable_data = []
    for cable in range(num_cables):
        # Each cable has its own linear combination of modes plus noise
        cable_transform = torch.randn(num_modes, input_dim) * 0.5
        cable_bias = torch.randn(input_dim) * 0.1

        cable_features = torch.matmul(mode_weights, cable_transform) + cable_bias
        cable_features += torch.randn(num_samples, input_dim) * 0.05  # Add noise
        cable_data.append(cable_features)

    # Stack data: (num_samples, num_cables, input_dim)
    data = torch.stack(cable_data, dim=1)

    # Create dummy labels and additional data (not used in AE but expected by interface)
    labels = torch.randint(0, 3, (num_samples,))  # 3 classes
    extra_data = torch.zeros(num_samples, 1)  # Placeholder

    # Create dataset and dataloader
    dataset = TensorDataset(data, labels, extra_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader, data.shape


def demonstrate_masking():
    """
    Demonstrate the masking functionality of the attention model.
    """
    print("=== Masking Demonstration ===")

    # Create a simple model
    model = AERegularizedAttention(
        input_dim=50,
        lr=1e-3,
        latent_dim=16,
        hidden_dim=64,
        num_layers=2,
        activation="relu",
        beta=1.0,
        num_heads=4,
        dropout=0.1,
        mask_ratio=0.3,  # Mask 30% of input
    )

    # Create sample input
    x = torch.randn(8, 50)  # Batch of 8 samples, 50 features each

    # Show masking in training mode
    model.train()
    masked_x, mask = model.apply_masking(x)

    print(f"Original input shape: {x.shape}")
    print(f"Masked input shape: {masked_x.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Fraction of elements masked: {(mask == 0).float().mean():.3f}")
    print(f"Mask token value: {model.mask_token.item():.3f}")

    # Show that no masking occurs in eval mode
    model.eval()
    masked_x_eval, mask_eval = model.apply_masking(x)
    print(
        f"Masking in eval mode (should be none): {(mask_eval == 0).float().mean():.3f}"
    )

    print()


def train_simple_example():
    """
    Train the attention autoencoder on dummy data.
    """
    print("=== Training Example ===")

    # Set up model
    input_dim = 128
    model = AERegularizedAttention(
        input_dim=input_dim,
        lr=1e-3,
        latent_dim=32,
        hidden_dim=128,
        num_layers=3,
        activation="relu",
        beta=0.5,
        num_heads=8,
        dropout=0.1,
        mask_ratio=0.15,
    )

    # Create data
    train_loader, data_shape = create_dummy_data(
        batch_size=32, num_cables=4, input_dim=input_dim, num_samples=200
    )

    print(f"Data shape: {data_shape}")
    print(f"Model architecture:")
    print(f"  Input dim: {input_dim}")
    print(f"  Latent dim: {model.latent_dim}")
    print(f"  Hidden dim: {model.hidden_dim}")
    print(f"  Num layers: {model.num_layers}")
    print(f"  Num heads: {model.num_heads}")
    print(f"  Mask ratio: {model.mask_ratio}")

    # Simple training loop (without PyTorch Lightning for simplicity)
    optimizer = torch.optim.Adam(model.parameters(), lr=model.lr)
    model.train()

    losses = []
    for epoch in range(20):
        epoch_loss = 0.0
        for batch_idx, (x, _, _) in enumerate(train_loader):
            optimizer.zero_grad()

            # Manual training step (similar to model.training_step)
            batch_size = x.shape[0]
            num_cables = x.shape[1]

            rec_loss = 0.0
            z_list = []

            # Process each cable
            for i in range(num_cables):
                x_rec, _, z, _ = model.forward(x[:, i])
                rec_loss += model.rec_loss_fn(x_rec, x[:, i])
                z_list.append(z)

            # Regularization loss
            z_stack = torch.stack(z_list, dim=0).transpose(0, 1)
            z_dist = torch.cdist(z_stack, z_stack, p=2)
            upper_triangular = torch.triu(z_dist, diagonal=1)
            dist_loss = upper_triangular.sum() / (batch_size * num_cables)

            loss = rec_loss + model.beta * dist_loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)

        if epoch % 5 == 0:
            print(f"Epoch {epoch:2d}: Loss = {avg_loss:.4f}")

    print(f"Final loss: {losses[-1]:.4f}")
    print()

    return model, train_loader, losses


def demonstrate_reconstruction():
    """
    Show reconstruction capabilities after training.
    """
    print("=== Reconstruction Demonstration ===")

    # Train model
    model, data_loader, losses = train_simple_example()

    # Get a sample for reconstruction
    model.eval()
    with torch.no_grad():
        x_sample, _, _ = next(iter(data_loader))
        x_sample = x_sample[:4]  # Take first 4 samples

        print(f"Sample shape: {x_sample.shape}")

        # Reconstruct each cable
        for cable_idx in range(x_sample.shape[1]):
            x_cable = x_sample[:, cable_idx]  # Shape: (4, input_dim)

            # Get reconstruction
            x_rec, _, z, _ = model.forward(x_cable)

            # Calculate reconstruction error
            rec_error = torch.mean((x_rec - x_cable) ** 2, dim=1)

            print(f"Cable {cable_idx + 1}:")
            print(f"  Latent representation shape: {z.shape}")
            print(f"  Reconstruction error per sample: {rec_error.tolist()}")
            print(f"  Average reconstruction error: {rec_error.mean():.4f}")

    print()


def compare_with_without_attention():
    """
    Compare the attention model with a simple MLP-based approach.
    """
    print("=== Attention vs MLP Comparison ===")

    input_dim = 64
    latent_dim = 16

    # Create the same data for both models
    train_loader, _ = create_dummy_data(
        batch_size=16, num_cables=4, input_dim=input_dim, num_samples=100
    )

    # Attention model
    attention_model = AERegularizedAttention(
        input_dim=input_dim,
        lr=1e-3,
        latent_dim=latent_dim,
        hidden_dim=64,
        num_layers=2,
        activation="relu",
        beta=0.5,
        num_heads=4,
        mask_ratio=0.15,
    )

    print(
        "Attention model parameters:",
        sum(p.numel() for p in attention_model.parameters()),
    )

    # Test forward pass
    attention_model.eval()
    with torch.no_grad():
        x_sample, _, _ = next(iter(train_loader))
        x_cable = x_sample[:, 0]  # First cable

        # Forward pass
        x_rec, _, z, _ = attention_model.forward(x_cable)

        print(f"Input shape: {x_cable.shape}")
        print(f"Latent shape: {z.shape}")
        print(f"Reconstruction shape: {x_rec.shape}")
        print(f"Reconstruction error: {torch.mean((x_rec - x_cable) ** 2):.4f}")


if __name__ == "__main__":
    print("AERegularizedAttention Example")
    print("=" * 50)

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Run demonstrations
    demonstrate_masking()
    demonstrate_reconstruction()
    compare_with_without_attention()

    print("Example completed successfully!")
    print("\nKey features of AERegularizedAttention:")
    print("1. Uses attention mechanisms instead of MLPs")
    print("2. Applies random masking during training")
    print("3. Maintains same interface as AERegularized")
    print("4. Includes cable embedding distance regularization")
    print("5. Supports configurable attention parameters (heads, dropout, etc.)")
