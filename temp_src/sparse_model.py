"""
Module that contains the encoder-decoder model for mimicking a
V-Cycle in a Multigrid solver. We focus on sparse and semi-sparse
models.

- SemiSparseVcycle: SemiSparse Encoder-Decoder model for mimicking a V-Cycle in a Multigrid solver.
- SemiSparseMG: SemiSparse Encoder-Decoder model for mimicking a Multigrid solver.

We base these models on unpublished work by Dr. Panayot Vassilevski and old code
written by Gabriel Pinochet-Soto.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as functional
import torchvision
import scipy.sparse as sp
import internal


class SemiSparseVcycle(nn.Module):
    """
    Sparse Encoder-Decoder model for mimicking a V-Cycle in a Multigrid solver.
    All tensors are assumed to be sparse.
    """

    def __init__(self, input_shape=None, coarse_shape=None, bool_matrix=None, **kwargs):
        """
        Constructor for the SemiSparseVcycle model.

        Required parameters:
        - input_shape (tuple): Shape of the input tensor.
        - coarse_shape (tuple): Shape of the coarse (smallest) tensor.
        - bool_matrix (torch.Tensor): Boolean matrix tensor. Tall matrix: input_shape[0] x coarse_shape[0].

        Other parameters are stored in a dictionary.
        """
        assert input_shape is not None, "Input shape must be provided."
        assert coarse_shape is not None, "Coarse shape must be provided."
        assert bool_matrix is not None, "Matrix must be provided."

        shape_bool_matrix = bool_matrix.shape
        assert (
            shape_bool_matrix[0] == input_shape[-1]
        ), f"Matrix must be tall. {shape_bool_matrix[0]} != {input_shape[-1]}"
        assert (
            shape_bool_matrix[1] == coarse_shape[-1]
        ), f"Matrix must be tall. {shape_bool_matrix[1]} != {coarse_shape[-1]}"

        super().__init__()

        self._name = "SemiSparseVcycle"
        self._input_shape = input_shape
        self._coarse_shape = coarse_shape
        self._sparsity_projector = bool_matrix

        self._dict = kwargs

        self.encoder = nn.Linear(
            in_features=self._input_shape[-1],
            out_features=self._coarse_shape[-1],
            bias=False,
        )

        self.decoder = nn.Linear(
            in_features=self._coarse_shape[-1],
            out_features=self._input_shape[-1],
            bias=False,
        )

        if "device" in self._dict:
            self.encoder.to(self._dict["device"])
            self.decoder.to(self._dict["device"])

        # Apply the initial sparsity to the weight matrices
        self.encoder.weight.data *= self._sparsity_projector.T
        self.decoder.weight.data *= self._sparsity_projector

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def apply_sparsity(self):
        """
        Apply sparsity to the weight matrices.
        """
        self.encoder.weight.grad *= self._sparsity_projector.T
        self.decoder.weight.grad *= self._sparsity_projector


class SemiSparseMG(nn.Module):
    """
    SemiSparse Encoder-Decoder model for mimicking a Multigrid solver.
    All tensors are assumed to be sparse.
    """

    def __init__(
        self,
        input_shape=None,
        coarse_shape=None,
        matrix=None,
        bool_matrix=None,
        **kwargs,
    ):
        """
        Constructor for the SparseMG model.

        Required parameters:
        - matrix (torch.Tensor): Matrix tensor.
        - input_shape (tuple): Shape of the input tensor.
        - coarse_shape (tuple): Shape of the coarse (smallest) tensor.
        - bool_matrix (torch.Tensor): Boolean matrix tensor.

        Other parameters are stored in a dictionary.
        """
        assert input_shape is not None, "Input shape must be provided."
        assert coarse_shape is not None, "Coarse shape must be provided."
        assert matrix is not None, "Matrix must be provided."
        assert bool_matrix is not None, "Matrix must be provided."

        shape_bool_matrix = bool_matrix.shape
        assert (
            shape_bool_matrix[0] == input_shape[-1]
        ), f"Matrix must be tall. {shape_bool_matrix[0]} != {input_shape[-1]}"
        assert (
            shape_bool_matrix[1] == coarse_shape[-1]
        ), f"Matrix must be tall. {shape_bool_matrix[1]} != {coarse_shape[-1]}"
        super().__init__()

        self._name = "SemiSparseMG"
        self._input_shape = input_shape
        self._coarse_shape = coarse_shape
        self._sparsity_projector = bool_matrix

        self._dict = kwargs

        self.encoder = nn.Linear(
            in_features=self._input_shape[-1],
            out_features=self._coarse_shape[-1],
            bias=False,
        )

        self.decoder = nn.Linear(
            in_features=self._coarse_shape[-1],
            out_features=self._input_shape[-1],
            bias=False,
        )

        self.range_space = nn.Linear(
            in_features=self._input_shape[-1],
            out_features=self._input_shape[-1],
            bias=False,
        )
        self.range_space.weight = nn.Parameter(matrix, requires_grad=False)

        if "device" in self._dict:
            self.encoder.to(self._dict["device"])
            self.decoder.to(self._dict["device"])
            self.range_space.to(self._dict["device"])

        # Apply the initial sparsity to the weight matrices
        self.encoder.weight.data *= self._sparsity_projector.T
        self.decoder.weight.data *= self._sparsity_projector

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.range_space(x)
        return x

    def apply_sparsity(self):
        """
        Apply sparsity to the weight matrices.
        """
        self.encoder.weight.grad *= self._sparsity_projector.T
        self.decoder.weight.grad *= self._sparsity_projector


# TESTS

LOCAL_FOLDER = "./tmp"


def check_directories():
    """
    Check if the directories exist.
    """
    import os

    if not os.path.exists(LOCAL_FOLDER + "/results"):
        os.makedirs(LOCAL_FOLDER + "/results")
        print(f"Created directories: {LOCAL_FOLDER}/results")


def make_bool_matrix_from_identity(input_shape):
    device = internal.get_device()
    id = sp.eye(input_shape[0]).tocsr()
    projector = internal.pyamg_get_sparcity_pattern_projector(id, "lloyd")
    projector = torch.tensor(projector[0].toarray()).to(device)
    return projector


def test_semisparse_vcycle():
    """
    Test the SemiSparseVcycle model.
    """
    import matplotlib.pyplot as plt
    from torchvision.utils import save_image

    device = internal.get_device()

    bool_matrix = make_bool_matrix_from_identity((784,))

    model = SemiSparseVcycle(
        input_shape=(784,),
        coarse_shape=(bool_matrix.shape[1],),
        bool_matrix=bool_matrix,
        device=device,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    loss_fn = nn.MSELoss()

    print(model)

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            LOCAL_FOLDER,
            train=True,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=256,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            LOCAL_FOLDER,
            train=False,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=256,
        shuffle=True,
    )

    def train(epoch):
        """
        Train the model.
        """
        model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch = model(data.view(-1, 784))
            loss = loss_fn(recon_batch, data.view(-1, 784))
            loss.backward()
            model.apply_sparsity()
            train_loss += loss.item()
            optimizer.step()
        print(f"Epoch: {epoch}, Loss: {train_loss / len(train_loader.dataset)}")

    def test(epoch):
        """
        Test the model.
        """
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for i, (data, _) in enumerate(test_loader):
                data = data.to(device)
                recon_batch = model(data.view(-1, 784))
                test_loss += loss_fn(recon_batch, data.view(-1, 784)).item()
                if i == 0:
                    n = min(data.size(0), 8)
                    comparison = torch.cat(
                        [
                            data.view(-1, 1, 28, 28)[:n],
                            recon_batch.view(-1, 1, 28, 28)[:n],
                        ]
                    )
                    save_image(
                        comparison.cpu(),
                        LOCAL_FOLDER
                        + "/results/reconstr_semisparse_vcycle_"
                        + str(epoch)
                        + ".png",
                        nrow=n,
                    )

        test_loss /= len(test_loader.dataset)
        print(f"Test set loss: {test_loss}")

    for epoch in range(1, 11):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            sample = torch.randn(64, bool_matrix.shape[1]).to(device)
            sample = model.decoder(sample).cpu()
            save_image(
                sample.view(64, 1, 28, 28),
                LOCAL_FOLDER + "/results/sample_vcycle_" + str(epoch) + ".png",
            )


def test_semisparse_mg():
    """
    Test the SemiSparseMG model.
    """
    import matplotlib.pyplot as plt
    from torchvision.utils import save_image

    device = internal.get_device()

    bool_matrix = make_bool_matrix_from_identity((784,))

    matrix = torch.eye(784).to(device)

    model = SemiSparseMG(
        input_shape=(784,),
        coarse_shape=(bool_matrix.shape[1],),
        matrix=matrix,
        bool_matrix=bool_matrix,
        device=device,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    loss_fn = nn.MSELoss()

    print(model)

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            LOCAL_FOLDER,
            train=True,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=256,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            LOCAL_FOLDER,
            train=False,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=256,
        shuffle=True,
    )

    def train(epoch):
        """
        Train the model.
        """
        model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch = model(data.view(-1, 784))
            loss = loss_fn(recon_batch, data.view(-1, 784))
            loss.backward()
            model.apply_sparsity()
            optimizer.step()
            train_loss += loss.item()
        print(f"Epoch: {epoch}, Loss: {train_loss / len(train_loader.dataset)}")

    def test(epoch):
        """
        Test the model.
        """
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for i, (data, _) in enumerate(test_loader):
                data = data.to(device)
                recon_batch = model(data.view(-1, 784))
                test_loss += loss_fn(recon_batch, data.view(-1, 784)).item()
                if i == 0:
                    n = min(data.size(0), 8)
                    comparison = torch.cat(
                        [
                            data.view(-1, 1, 28, 28)[:n],
                            recon_batch.view(-1, 1, 28, 28)[:n],
                        ]
                    )
                    save_image(
                        comparison.cpu(),
                        LOCAL_FOLDER
                        + "/results/reconstr_semisparse_mg_"
                        + str(epoch)
                        + ".png",
                        nrow=n,
                    )

        test_loss /= len(test_loader.dataset)
        print(f"Test set loss: {test_loss}")

    for epoch in range(1, 11):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            sample = torch.randn(64, bool_matrix.shape[1]).to(device)
            sample = model.decoder(sample).cpu()
            save_image(
                sample.view(64, 1, 28, 28),
                LOCAL_FOLDER + "/results/sample_mg_" + str(epoch) + ".png",
            )


if __name__ == "__main__":
    test_semisparse_vcycle()
    test_semisparse_mg()
