"""
Module that contains the encoder-decoder model for mimicking a
V-Cycle in a Multigrid solver.

We have four models:
- DenseVcycle: Encoder-Decoder model for mimicking a V-Cycle in a Multigrid solver.
- DenseMG: Encoder-Decoder model for mimicking a Multigrid solver.
- SparseVcycle: Sparse Encoder-Decoder model for mimicking a V-Cycle in a Multigrid solver.
- SparseMG: Sparse Encoder-Decoder model for mimicking a Multigrid solver.

We base these models on unpublished work by Dr. Panayot Vassilevski and old code
written by Gabriel Pinochet-Soto.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as functional
import torchvision

def get_device():
    """
    Get the device (GPU, MPS, or CPU).
    """
    device = (
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"Using device: {device}")
    return device

class DenseVcycle(nn.Module):
    """
    Dense Encoder-Decoder model for mimicking a V-Cycle in a Multigrid solver.
    All tensors are assumed to be dense.
    """
    def __init__(
        self,
        **kwargs
    ):
        """
        Constructor for the DenseVcycle model.

        Required parameters:
        - input_shape (tuple): Shape of the input tensor.
        - coarse_shape (tuple): Shape of the coarse (smallest) tensor.
        
        Other parameters are stored in a dictionary.
        """
        super().__init__()

        required_params = [
            "input_shape",
            "coarse_shape",
        ]

        for param in required_params:
            assert param in kwargs, f"{param} must be provided."

        self._name = "DenseVcycle"
        for param, value in kwargs.items():
            setattr(self, f"_{param}", value)

        self._dict = {}
        for param, value in kwargs.items():
            if param not in required_params:
                self._dict[param] = value

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

class DenseMG(nn.Module):
    """
    Dense Encoder-Decoder model for mimicking a Multigrid solver.
    All tensors are assumed to be dense.
    """
    def __init__(
        self,
        matrix=None,
        **kwargs
    ):
        """
        Constructor for the DenseMG model.

        Required parameters:
        - matrix (torch.Tensor): Matrix tensor.
        - input_shape (tuple): Shape of the input tensor.
        - coarse_shape (tuple): Shape of the coarse (smallest) tensor.
        
        Other parameters are stored in a dictionary.
        """
        assert matrix is not None, "Matrix must be provided."
        super().__init__()

        required_params = [
            "input_shape",
            "coarse_shape",
        ]

        for param in required_params:
            assert param in kwargs, f"{param} must be provided."

        self._name = "DenseMG"
        for param, value in kwargs.items():
            setattr(self, f"_{param}", value)

        self._dict = {}
        for param, value in kwargs.items():
            if param not in required_params:
                self._dict[param] = value

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
            out_features=self._coarse_shape[-1],
            bias=False,
        )
        self.range_space.weight = nn.Parameter(matrix, requires_grad=False)

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

class SparseVcycle(nn.Module):
    """
    Sparse Encoder-Decoder model for mimicking a V-Cycle in a Multigrid solver.
    All tensors are assumed to be sparse.
    """
    def __init__(
        self,
        **kwargs
    ):
        """
        Constructor for the SparseVcycle model.

        Required parameters:
        - input_shape (tuple): Shape of the input tensor.
        - coarse_shape (tuple): Shape of the coarse (smallest) tensor.
        
        Other parameters are stored in a dictionary.
        """
        raise NotImplementedError

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        raise NotImplementedError

class SparseMG(nn.Module):
    """
    Sparse Encoder-Decoder model for mimicking a Multigrid solver.
    All tensors are assumed to be sparse.
    """
    def __init__(
        self,
        matrix=None,
        **kwargs
    ):
        """
        Constructor for the SparseMG model.

        Required parameters:
        - matrix (torch.Tensor): Matrix tensor.
        - input_shape (tuple): Shape of the input tensor.
        - coarse_shape (tuple): Shape of the coarse (smallest) tensor.
        
        Other parameters are stored in a dictionary.
        """
        raise NotImplementedError

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        raise NotImplementedError

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

def test_dense_vcycle():
    """
    Test the DenseVcycle model.
    Encoding-decoding MNIST data.
    """
    import torchvision
    import matplotlib.pyplot as plt
    from torchvision.utils import save_image

    device = get_device()

    model = DenseVcycle(
        input_shape=(784,),
        coarse_shape=(32,),
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    loss_fn = nn.MSELoss()

    print(model)

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            LOCAL_FOLDER,
            train=True,
            download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ])),
        batch_size=256,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            LOCAL_FOLDER,
            train=False,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ])),
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
            train_loss += loss.item()
            optimizer.step()
        print(f"Epoch: {epoch} Loss: {train_loss / len(train_loader.dataset)}")

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
                    comparison = torch.cat([data.view(-1, 1, 28, 28)[:n],
                                            recon_batch.view(-1, 1, 28, 28)[:n]])
                    save_image(
                        comparison.cpu(),
                        LOCAL_FOLDER + '/results/reconstruction_' + str(epoch) + '.png',
                        nrow=n
                    )

        test_loss /= len(test_loader.dataset)
        print(f"Test set loss: {test_loss}")

    for epoch in range(1, 11):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            sample = torch.zeros(64, 32).to(device)
            # draw a square
            for i in range(32):
                if i < 8:
                    sample[:, i] = 1
                elif i < 16:
                    sample[:, i] = 0
                elif i < 24:
                    sample[:, i] = 1
                else:
                    sample[:, i] = 0
            sample = model.decoder(sample).cpu()
            save_image(
                sample.view(64, 1, 28, 28),
                LOCAL_FOLDER + '/results/sample_' + str(epoch) + '.png'
            )

def test_dense_mg():
    raise NotImplementedError

if __name__ == "__main__":
    check_directories()
    test_dense_vcycle()
    # test_dense_mg()
