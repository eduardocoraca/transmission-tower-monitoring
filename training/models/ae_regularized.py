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


class VAERegularized(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        lr: float,
        latent_dim: int,
        hidden_dim: int,
        num_layers: int,
        activation: str,
        beta: float,
        kl_weight: float = 1.0,
    ):
        """
        A LightningModule that bundles together:
          - An encoder/decoder for variational autoencoder
          - Regularization in terms of cable embedding distance
          - Loss functions for reconstruction, KL divergence, and distance
        """
        super().__init__()

        # Encoder outputs both mu and logvar
        self.encoder_mu = MLPEncoder(
            input_dim=input_dim,
            embedding_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation_fn=activation,
        )
        self.encoder_logvar = MLPEncoder(
            input_dim=input_dim,
            embedding_dim=latent_dim,
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
        self.dropout = torch.nn.Dropout(p=0.2)

        # Store loss functions, hyperparameters
        self.rec_loss_fn = torch.nn.BCEWithLogitsLoss(reduction="mean")
        self.lr = lr
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.activation = activation
        self.beta = beta
        self.kl_weight = kl_weight

    def encode(self, x: torch.Tensor):
        """
        Encode input to latent distribution parameters.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            tuple: (mu, logvar) - mean and log variance of latent distribution
        """
        x_drop = self.dropout(x)
        mu = self.encoder_mu(x_drop)
        logvar = self.encoder_logvar(x_drop)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        """
        Reparameterization trick to sample from latent distribution.

        Args:
            mu (torch.Tensor): Mean of latent distribution
            logvar (torch.Tensor): Log variance of latent distribution

        Returns:
            torch.Tensor: Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor):
        """
        Compute KL divergence between latent distribution and standard normal.

        Args:
            mu (torch.Tensor): Mean of latent distribution
            logvar (torch.Tensor): Log variance of latent distribution

        Returns:
            torch.Tensor: KL divergence loss
        """
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    def forward(self, x: torch.Tensor):
        """
        Forward pass for inference:
         1) encode input into (mu, logvar)
         2) reparameterize to latent z
         3) decode z -> reconstruction
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_rec = self.decoder(z)
        x_rec = torch.nn.functional.sigmoid(x_rec)
        return x_rec, mu, z, logvar

    def training_step(self, batch, batch_idx):
        """
        Runs each optimization step:
         - forward pass
         - compute combined loss (reconstruction + KL + distance)
         - log metrics
         - return loss for backprop
        """
        x = batch.float()
        batch_size = x.shape[0]
        x_rec_list = []
        z_list = []
        rec_loss = 0.0
        kl_loss = 0.0
        num_cables = x.shape[1]

        for i in range(num_cables):
            mu, logvar = self.encode(x[:, i])
            z = self.reparameterize(mu, logvar)
            x_rec = self.decoder(z)

            rec_loss += self.rec_loss_fn(x_rec, x[:, i])
            kl_loss += self.kl_divergence(mu, logvar)
            x_rec_list.append(x_rec)
            z_list.append(z)

        # Compute the pairwise distance between the embeddings
        z_stack = torch.stack(z_list, dim=0).transpose(0, 1)
        z_dist = torch.cdist(z_stack, z_stack, p=2)
        upper_triangular = torch.triu(z_dist, diagonal=1)
        dist_loss = upper_triangular.sum() / (batch_size + num_cables)

        # Average KL loss over batch and cables
        kl_loss = kl_loss / (batch_size * num_cables)

        loss = rec_loss + self.kl_weight * kl_loss + self.beta * dist_loss

        # Log for visualization in Lightning
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_rec_loss", rec_loss, on_step=False, on_epoch=True)
        self.log("train_kl_loss", kl_loss, on_step=False, on_epoch=True)
        self.log("train_dist_loss", dist_loss, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the VAE+regularization:
        - Forward pass
        - Compute losses (reconstruction, KL, distance)
        - Log metrics
        - No weight updates (handled automatically by Lightning)
        """
        x = batch.float()
        batch_size = x.shape[0]
        x_rec_list = []
        z_list = []
        rec_loss = 0.0
        kl_loss = 0.0
        num_cables = x.shape[1]

        for i in range(num_cables):
            mu, logvar = self.encode(x[:, i])
            z = self.reparameterize(mu, logvar)
            x_rec = self.decoder(z)

            rec_loss += self.rec_loss_fn(x_rec, x[:, i])
            kl_loss += self.kl_divergence(mu, logvar)
            x_rec_list.append(x_rec)
            z_list.append(z)

        # Compute the pairwise distance between the embeddings
        z_stack = torch.stack(z_list, dim=0).transpose(0, 1)
        z_dist = torch.cdist(z_stack, z_stack, p=2)
        upper_triangular = torch.triu(z_dist, diagonal=1)
        dist_loss = upper_triangular.sum() / (batch_size + num_cables)

        # Average KL loss over batch and cables
        kl_loss = kl_loss / (batch_size * num_cables)

        loss = rec_loss + self.kl_weight * kl_loss + self.beta * dist_loss

        # Log validation metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_rec_loss", rec_loss, on_step=False, on_epoch=True)
        self.log("val_kl_loss", kl_loss, on_step=False, on_epoch=True)
        self.log("val_dist_loss", dist_loss, on_step=False, on_epoch=True)

        return {"val_loss": loss}

    def configure_optimizers(self):
        """
        Set up optimizers/schedulers for training.
        """
        optimizer = torch.optim.Adam(
            list(self.encoder_mu.parameters())
            + list(self.encoder_logvar.parameters())
            + list(self.decoder.parameters()),
            lr=self.lr,
        )
        return optimizer


class AECosineRegularized(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        lr: float,
        latent_dim: int,
        hidden_dim: int,
        num_layers: int,
        activation: str,
        beta: float,
    ):
        """
        A LightningModule that bundles together:
          - An encoder/decoder for variational autoencoder
          - Regularization in terms of cable embedding distance
          - Loss functions for reconstruction, KL divergence, and distance
        """
        super().__init__()

        # Encoder outputs both mu and logvar
        self.encoder = MLPEncoder(
            input_dim=input_dim,
            embedding_dim=latent_dim,
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
        self.dropout = torch.nn.Dropout(p=0.2)

        # Store loss functions, hyperparameters
        self.rec_loss_fn = torch.nn.MSELoss(reduction="mean")
        self.lr = lr
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.activation = activation
        self.beta = beta

    def encode(self, x: torch.Tensor):
        """
        Encode input to latent distribution parameters.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            tuple: (mu, logvar) - mean and log variance of latent distribution
        """
        x_drop = self.dropout(x)
        z = self.encoder(x_drop)
        return z

    def forward(self, x: torch.Tensor):
        """
        Forward pass for inference:
         1) encode input into (mu, logvar)
         2) reparameterize to latent z
         3) decode z -> reconstruction
        """
        z = self.encode(x)
        z = torch.nn.functional.tanh(z)
        x_rec = self.decoder(z)
        x_rec = torch.nn.functional.sigmoid(x_rec)

        return x_rec, z

    def training_step(self, batch, batch_idx):
        """
        Runs each optimization step:
         - forward pass
         - compute combined loss (reconstruction + KL + distance)
         - log metrics
         - return loss for backprop
        """
        x = batch.float()
        x_rec_list = []
        z_list = []
        rec_loss = 0.0
        num_cables = x.shape[1]

        for i in range(num_cables):
            x_rec, z = self.forward(x[:, i])
            rec_loss += self.rec_loss_fn(x_rec, x[:, i])
            x_rec_list.append(x_rec)
            z_list.append(z)

        # Compute the pairwise distance between the embeddings
        # z_stack = torch.stack(z_list, dim=0).transpose(0, 1)
        # z_dist = torch.cdist(z_stack, z_stack, p=2)
        # upper_triangular = torch.triu(z_dist, diagonal=1)
        # dist_loss = upper_triangular.sum() / (batch_size + num_cables)
        for i in range(num_cables):
            for j in range(i + 1, num_cables):
                cos_sim = torch.nn.functional.cosine_similarity(
                    z_list[i], z_list[j], dim=1, eps=1e-8
                )
                if i == 0 and j == 1:
                    dist_loss = (1 - cos_sim).sum()
                else:
                    dist_loss += (1 - cos_sim).sum()
        loss = rec_loss + self.beta * dist_loss

        # Log for visualization in Lightning
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_rec_loss", rec_loss, on_step=False, on_epoch=True)
        self.log("train_dist_loss", dist_loss, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the VAE+regularization:
        - Forward pass
        - Compute losses (reconstruction, KL, distance)
        - Log metrics
        - No weight updates (handled automatically by Lightning)
        """
        x = batch.float()
        x_rec_list = []
        z_list = []
        rec_loss = 0.0
        num_cables = x.shape[1]

        for i in range(num_cables):
            x_rec, z = self.forward(x[:, i])
            rec_loss += self.rec_loss_fn(x_rec, x[:, i])
            x_rec_list.append(x_rec)
            z_list.append(z)

        for i in range(num_cables):
            for j in range(i + 1, num_cables):
                cos_sim = torch.nn.functional.cosine_similarity(
                    z_list[i], z_list[j], dim=1
                )
                if i == 0 and j == 1:
                    dist_loss = (1 - cos_sim).sum()
                else:
                    dist_loss += (1 - cos_sim).sum()
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
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=self.lr,
        )
        return optimizer
