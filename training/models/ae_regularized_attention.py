import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import math
from typing import Optional, Tuple


def get_activation(activation_fn: str) -> torch.nn.Module:
    """
    Get the activation function based on the input string.

    Args:
        activation_fn (str): The activation function name. Valid options are "relu", "leakyrelu", "sigmoid", "tanh".

    Returns:
        torch.nn.Module: The corresponding activation function.
    """
    activation_dict = {
        "relu": torch.nn.ReLU,
        "leakyrelu": torch.nn.LeakyReLU,
        "sigmoid": torch.nn.Sigmoid,
        "tanh": torch.nn.Tanh,
        "relu6": torch.nn.ReLU6,
        "gelu": torch.nn.GELU,
        "elu": torch.nn.ELU,
        "selu": torch.nn.SELU,
        "softmax": torch.nn.Softmax,
        "hardtanh": torch.nn.Hardtanh,
        "tanhshrink": torch.nn.Tanhshrink,
    }

    # Convert to lowercase and check the activation function dictionary
    activation_fn = activation_fn.lower()

    if activation_fn in activation_dict:
        return activation_dict[activation_fn]
    else:
        raise ValueError(f"Unsupported activation function: {activation_fn}")


class PositionalEncoding(torch.nn.Module):
    """
    Positional encoding for transformer-style attention.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (seq_len, batch_size, d_model)
        """
        return x + self.pe[: x.size(0), :]


class MultiHeadAttention(torch.nn.Module):
    """
    Multi-head attention mechanism.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = torch.nn.Linear(d_model, d_model)
        self.w_k = torch.nn.Linear(d_model, d_model)
        self.w_v = torch.nn.Linear(d_model, d_model)
        self.w_o = torch.nn.Linear(d_model, d_model)

        self.dropout = torch.nn.Dropout(dropout)

    def scaled_dot_product_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scaled dot-product attention.
        """
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        output = torch.matmul(attention_weights, value)
        return output, attention_weights

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = query.size(0)

        # Linear transformations and split into heads
        Q = (
            self.w_q(query)
            .view(batch_size, -1, self.num_heads, self.d_k)
            .transpose(1, 2)
        )
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = (
            self.w_v(value)
            .view(batch_size, -1, self.num_heads, self.d_k)
            .transpose(1, 2)
        )

        # Apply attention
        attention_output, _ = self.scaled_dot_product_attention(Q, K, V, mask)

        # Concatenate heads
        attention_output = (
            attention_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.d_model)
        )

        # Final linear transformation
        output = self.w_o(attention_output)
        return output


class TransformerBlock(torch.nn.Module):
    """
    A single transformer block with self-attention and feed-forward layers.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)

        self.feed_forward = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_ff),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(d_ff, d_model),
        )

        self.dropout = torch.nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention with residual connection
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class AttentionEncoder(torch.nn.Module):
    """
    Attention-based encoder for the autoencoder.
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        dropout: float = 0.1,
        max_seq_len: int = 1000,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.d_model = d_model

        # Input projection
        self.input_projection = torch.nn.Linear(
            1, d_model
        )  # Each feature is projected separately

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)

        # Transformer blocks
        self.transformer_blocks = torch.nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ]
        )

        # Output projection to embedding space
        self.output_projection = torch.nn.Linear(d_model, embedding_dim)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the attention encoder.

        Args:
            x: Input tensor of shape (batch_size, input_dim)
            mask: Optional mask tensor

        Returns:
            Encoded tensor of shape (batch_size, embedding_dim)
        """
        batch_size, input_dim = x.shape

        # Reshape input to treat each feature as a sequence element
        # (batch_size, input_dim) -> (batch_size, input_dim, 1)
        x = x.unsqueeze(-1)

        # Project each feature to d_model dimensions
        # (batch_size, input_dim, 1) -> (batch_size, input_dim, d_model)
        x = self.input_projection(x)

        # Add positional encoding
        # Transpose for positional encoding: (input_dim, batch_size, d_model)
        x = x.transpose(0, 1)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # Back to (batch_size, input_dim, d_model)

        x = self.dropout(x)

        # Apply transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, mask)

        # Global average pooling to get single representation
        # (batch_size, input_dim, d_model) -> (batch_size, d_model)
        x = x.mean(dim=1)

        # Project to embedding dimension
        # (batch_size, d_model) -> (batch_size, embedding_dim)
        x = self.output_projection(x)

        return x


class AttentionDecoder(torch.nn.Module):
    """
    Attention-based decoder for the autoencoder.
    """

    def __init__(
        self,
        output_dim: int,
        embedding_dim: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        dropout: float = 0.1,
        max_seq_len: int = 1000,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.embedding_dim = embedding_dim
        self.d_model = d_model

        # Embedding projection
        self.embedding_projection = torch.nn.Linear(embedding_dim, d_model)

        # Learnable queries for each output position
        self.output_queries = torch.nn.Parameter(torch.randn(output_dim, d_model))

        # Positional encoding for queries
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)

        # Cross-attention layers
        self.cross_attention_layers = torch.nn.ModuleList(
            [MultiHeadAttention(d_model, num_heads, dropout) for _ in range(num_layers)]
        )

        # Layer norms
        self.layer_norms = torch.nn.ModuleList(
            [torch.nn.LayerNorm(d_model) for _ in range(num_layers)]
        )

        # Output projection
        self.output_projection = torch.nn.Linear(d_model, 1)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the attention decoder.

        Args:
            z: Encoded tensor of shape (batch_size, embedding_dim)

        Returns:
            Reconstructed tensor of shape (batch_size, output_dim)
        """
        batch_size = z.shape[0]

        # Project embedding to d_model
        # (batch_size, embedding_dim) -> (batch_size, 1, d_model)
        z_proj = self.embedding_projection(z).unsqueeze(1)

        # Prepare queries for each output position
        # (output_dim, d_model) -> (batch_size, output_dim, d_model)
        queries = self.output_queries.unsqueeze(0).expand(batch_size, -1, -1)

        # Add positional encoding to queries
        queries = queries.transpose(0, 1)  # (output_dim, batch_size, d_model)
        queries = self.pos_encoding(queries)
        queries = queries.transpose(0, 1)  # (batch_size, output_dim, d_model)

        queries = self.dropout(queries)

        # Apply cross-attention layers
        for cross_attn, layer_norm in zip(
            self.cross_attention_layers, self.layer_norms
        ):
            # Cross-attention: queries attend to the encoded representation
            attn_output = cross_attn(queries, z_proj, z_proj)
            queries = layer_norm(queries + self.dropout(attn_output))

        # Project to output
        # (batch_size, output_dim, d_model) -> (batch_size, output_dim, 1) -> (batch_size, output_dim)
        output = self.output_projection(queries).squeeze(-1)

        return output


class AERegularizedAttention(pl.LightningModule):
    """
    Attention-based autoencoder with regularization.

    This model applies masking to the input during training, removing some elements,
    and attempts to reconstruct the original input using attention mechanisms.
    """

    def __init__(
        self,
        input_dim: int,
        lr: float,
        latent_dim: int,
        hidden_dim: int,  # d_model for attention
        num_layers: int,
        activation: str,  # Not used in attention but kept for compatibility
        beta: float,
        num_heads: int = 8,
        dropout: float = 0.1,
        mask_ratio: float = 0.15,  # Fraction of input to mask during training
        d_ff_ratio: int = 4,  # Feed-forward dimension ratio
    ):
        """
        A LightningModule that bundles together:
          - An attention-based encoder/decoder for autoencoder
          - Input masking during training
          - Regularization in terms of cable embedding distance
          - Loss functions for reconstruction and distance
        """
        super().__init__()

        # Calculate feed-forward dimension
        d_ff = hidden_dim * d_ff_ratio

        self.encoder = AttentionEncoder(
            input_dim=input_dim,
            embedding_dim=latent_dim,
            d_model=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_seq_len=input_dim,
        )

        self.decoder = AttentionDecoder(
            output_dim=input_dim,
            embedding_dim=latent_dim,
            d_model=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_seq_len=input_dim,
        )

        # Store loss functions, hyperparameters
        self.rec_loss_fn = torch.nn.MSELoss(reduction="mean")
        self.lr = lr
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.activation = activation
        self.beta = beta
        self.mask_ratio = mask_ratio
        self.num_heads = num_heads
        self.dropout = dropout

        # Mask token (learnable parameter)
        self.mask_token = torch.nn.Parameter(torch.randn(1))

    def apply_masking(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply random masking to input during training.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            masked_x: Input with some elements masked
            mask: Binary mask indicating which elements were masked
        """
        if not self.training:
            return x, torch.ones_like(x)

        batch_size, input_dim = x.shape

        # Create random mask
        mask = torch.rand(batch_size, input_dim, device=x.device) > self.mask_ratio

        # Apply mask (replace masked elements with mask token)
        masked_x = x.clone()
        masked_x[~mask] = self.mask_token

        return masked_x, mask.float()

    def encode(self, x: torch.Tensor):
        """
        Encode input with optional masking.
        """
        masked_x, mask = self.apply_masking(x)
        h = self.encoder(masked_x)
        return h, h

    def forward(self, x: torch.Tensor):
        """
        Forward pass for inference:
         1) apply masking (during training)
         2) encode input into latent representation
         3) decode to reconstruction
        """
        z, _ = self.encode(x)
        x_rec = self.decoder(z)
        return x_rec, None, z, None

    def training_step(self, batch, batch_idx):
        """
        Runs each optimization step:
         - forward pass with masking
         - compute combined loss (reconstruction + regularization)
         - log metrics
         - return loss for backprop
        """
        x, _, _ = batch
        batch_size = x.shape[0]
        x_rec_list = []
        z_list = []
        rec_loss = 0.0
        num_cables = x.shape[1]

        for i in range(num_cables):
            x_rec, _, z, _ = self.forward(x[:, i])
            rec_loss += self.rec_loss_fn(
                x_rec, x[:, i]
            )  # Reconstruct original (unmasked) input
            x_rec_list.append(x_rec)
            z_list.append(z)

        # Compute the pairwise distance between the embeddings
        z_stack = torch.stack(z_list, dim=0).transpose(0, 1)
        z_dist = torch.cdist(z_stack, z_stack, p=2)
        upper_triangular = torch.triu(z_dist, diagonal=1)
        dist_loss = upper_triangular.sum() / (batch_size * num_cables)

        loss = rec_loss + self.beta * dist_loss

        # Log for visualization in Lightning
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_rec_loss", rec_loss, on_step=False, on_epoch=True)
        self.log("train_dist_loss", dist_loss, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step:
        - Forward pass without masking
        - Compute losses (reconstruction, distance)
        - Log metrics
        - No weight updates (handled automatically by Lightning)
        """
        x, _, _ = batch
        batch_size = x.shape[0]
        x_rec_list = []
        z_list = []
        rec_loss = 0.0
        num_cables = x.shape[1]

        for i in range(num_cables):
            x_rec, _, z, _ = self.forward(x[:, i])
            rec_loss += self.rec_loss_fn(x_rec, x[:, i])
            x_rec_list.append(x_rec)
            z_list.append(z)

        # Compute the pairwise distance between the embeddings
        z_stack = torch.stack(z_list, dim=0).transpose(0, 1)
        z_dist = torch.cdist(z_stack, z_stack, p=2)
        upper_triangular = torch.triu(z_dist, diagonal=1)
        dist_loss = upper_triangular.sum() / (batch_size * num_cables)

        loss = rec_loss + self.beta * dist_loss

        # Log validation metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_rec_loss", rec_loss, on_step=False, on_epoch=True)
        self.log("val_dist_loss", dist_loss, on_step=False, on_epoch=True)

        return {"val_loss": loss}

    def configure_optimizers(self):
        """
        Set up optimizers/schedulers for training.
        """
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + [self.mask_token],
            lr=self.lr,
        )
        return optimizer
