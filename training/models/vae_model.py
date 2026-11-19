from pytorch_lightning.loggers import CometLogger
import torch
import pytorch_lightning as pl


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


class MLPEncoder(torch.nn.Module):
    """
    Multi-Layer Perceptron (MLP) encoder for the autoencoder.

    Args:
        input_dim (int): Dimension of the input data.
        embedding_dim (int): Dimension of the encoded embedding.
        hidden_dim (int): Number of neurons in the hidden layers.
        num_layers (int): Number of hidden layers in the encoder.
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        activation_fn: str = "relu",
    ) -> None:
        super().__init__()
        self.input_dim: int = input_dim
        self.embedding_dim: int = embedding_dim
        self.hidden_dim: int = hidden_dim
        self.activation_fn: str = activation_fn

        # Get the activation function based on the string
        activation = get_activation(activation_fn)

        layers: list[torch.nn.Module] = [
            torch.nn.Linear(input_dim, hidden_dim),
            activation(),
        ]
        for _ in range(num_layers):
            layers += [
                torch.nn.Linear(hidden_dim, hidden_dim),
                activation(),
            ]
        layers.append(torch.nn.Linear(hidden_dim, embedding_dim))
        self.model: torch.nn.Sequential = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder.

        Args:
            x (torch.Tensor): Input tensor of shape `(batch_size, input_dim)`.

        Returns:
            torch.Tensor: Encoded tensor of shape `(batch_size, embedding_dim)`.
        """
        return self.model(x)


class MLPDecoder(torch.nn.Module):
    """
    Multi-Layer Perceptron (MLP) decoder for the autoencoder.

    Args:
        input_dim (int): Dimension of the input data to be reconstructed.
        embedding_dim (int): Dimension of the encoded embedding.
        hidden_dim (int): Number of neurons in the hidden layers.
        num_layers (int): Number of hidden layers in the decoder.
    """

    def __init__(
        self,
        output_dim: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        activation_fn: str = "relu",
    ) -> None:
        super().__init__()
        self.output_dim: int = output_dim
        self.embedding_dim: int = embedding_dim
        self.hidden_dim: int = hidden_dim
        self.activation_fn: str = activation_fn

        # Get the activation function based on the string
        activation = get_activation(activation_fn)

        layers: list[torch.nn.Module] = [
            torch.nn.Linear(embedding_dim, hidden_dim),
            activation(),
        ]
        for _ in range(num_layers):
            layers += [
                torch.nn.Linear(hidden_dim, hidden_dim),
                activation(),
            ]
        layers += [torch.nn.Linear(hidden_dim, output_dim)]
        self.model: torch.nn.Sequential = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the decoder.

        Args:
            x (torch.Tensor): Encoded tensor of shape `(batch_size, embedding_dim)`.

        Returns:
            torch.Tensor: Reconstructed tensor of shape `(batch_size, input_dim)`.
        """
        return self.model(x)


class VAE(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        lr: float,
        beta: float,
        latent_dim: int,
        hidden_dim: int,
        num_layers: int,
        activation: str,
        num_classes: int,
    ):
        """
        A LightningModule that bundles together:
          - An encoder/decoder for (variational) autoencoder
          - A classifier for latent representations
          - Loss functions for reconstruction, classification, and KL divergence
        """
        super().__init__()
        self.encoder = MLPEncoder(
            input_dim=input_dim,
            embedding_dim=latent_dim * 2,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation_fn=activation,
        )
        self.decoder = MLPDecoder(
            output_dim=input_dim,
            embedding_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation_fn=activation,
        )
        self.clf = Clf(latent_dim, num_classes)

        # Store loss functions, hyperparameters
        self.rec_loss_fn = torch.nn.MSELoss(reduction="mean")
        self.beta = beta
        self.lr = lr
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.activation = activation

    def encode(self, x: torch.Tensor):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=1)
        return mu, logvar

    def forward(self, x: torch.Tensor):
        """
        Forward pass for inference:
         1) encode input into (mu, logvar)
         2) reparameterize to latent z
         3) decode z -> reconstruction
         4) classify z -> classes
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        x_rec = self.decoder(z)
        y_pred = self.clf(mu)
        return x_rec, y_pred, mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, log_var: torch.Tensor):
        """
        Reparameterization trick to sample from N(mu, var) using N(0,1).
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def kl_divergence_normals(mu1, logvar1, mu2, logvar2):
        """
        KL divergence between two normals N(mu1, exp(logvar1)) and N(mu2, exp(logvar2)).
           KL(N1 || N2) = 0.5 * [ (logvar2 - logvar1) + (exp(logvar1)/exp(logvar2))
                                  + ((mu1 - mu2)^2 / exp(logvar2)) - 1 ]
        """
        var1 = torch.exp(logvar1)
        var2 = torch.exp(logvar2)
        kl = 0.5 * (logvar2 - logvar1 + var1 / var2 + (mu1 - mu2).pow(2) / var2 - 1.0)
        return kl

    def training_step(self, batch, batch_idx):
        """
        Runs each optimization step:
         - forward pass
         - compute combined loss (reconstruction + classification + KL)
         - log metrics
         - return loss for backprop
        """
        x, y, _ = batch
        x_rec, y_pred, mu, logvar = self.forward(x)

        # Desired distribution: N(0, 1) for each dimension
        desired_mu = torch.zeros_like(mu)
        desired_logvar = torch.zeros_like(mu)

        # Compute losses
        rec_loss_val = self.rec_loss_fn(x_rec, x)
        kl_loss_val = self.kl_divergence_normals(
            mu, logvar, desired_mu, desired_logvar
        ).mean()

        loss = rec_loss_val + self.beta * kl_loss_val

        # Log for visualization in Lightning
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_rec_loss", rec_loss_val, on_step=False, on_epoch=True)
        self.log("train_kl_loss", kl_loss_val, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the VAE+classifier:
        - Forward pass
        - Compute losses (reconstruction, classification, KL)
        - Log metrics
        - No weight updates (handled automatically by Lightning)
        """
        x, y, _ = batch

        # Make sure model is in eval mode during validation
        with torch.no_grad():
            x_rec, y_pred, mu, logvar = self.forward(x)

            # Desired distribution: N(0, 1) for each dimension
            desired_mu = torch.zeros_like(mu)
            desired_logvar = torch.zeros_like(mu)

            # Compute the same losses as in training
            rec_loss_val = self.rec_loss_fn(x_rec, x)
            kl_loss_val = self.kl_divergence_normals(
                mu, logvar, desired_mu, desired_logvar
            ).mean()

            loss = rec_loss_val + self.beta * kl_loss_val

        # Log validation metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_rec_loss", rec_loss_val, on_step=False, on_epoch=True)
        self.log("val_kl_loss", kl_loss_val, on_step=False, on_epoch=True)

        return {"val_loss": loss}

    def configure_optimizers(self):
        """
        Set up optimizers/schedulers for training.
        """
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.clf.parameters()),
            lr=self.lr,
        )
        return optimizer
