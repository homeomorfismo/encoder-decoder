"""
TOML parser and dataclasses for the TOML data model.
"""

from dataclasses import dataclass
from typing import Any, Dict
import argparse
import toml


@dataclass
class MeshConfig:
    maxh: float


@dataclass
class DataGenConfig:
    tol: float
    iterations: int
    is_complex: bool
    n_samples: int
    use_restricted_operator: bool


@dataclass
class ModelConfig:
    compression_factor: float
    seed: int
    init_encoder_type: str
    init_decoder_type: str
    init_encoder_kwargs: Dict[str, Any]
    init_decoder_kwargs: Dict[str, Any]


@dataclass
class OptimizationConfig:
    optimizer_type: str
    optimizer_kwargs: Dict[str, Any]
    ord: int
    reg: float


@dataclass
class TrainingConfig:
    n_epochs: int
    freq: int
    batch_size: int


@dataclass
class CoarseningConfig:
    coarsening_type: str
    use_restricted_operator: bool
    regularization: float


@dataclass
class SolverConfig:
    solver_tol: float
    solver_max_iter: int


@dataclass
class SmootherConfig:
    smoother_tol: float
    smoother_max_iter: int


@dataclass
class OutputConfig:
    save_weights: bool
    plot_weights: bool
    strict_assert: bool
    use_progress_bar: bool


@dataclass
class Config:
    mesh: MeshConfig
    data_gen: DataGenConfig
    model: ModelConfig
    optimization: OptimizationConfig
    training: TrainingConfig
    coarsening: CoarseningConfig
    solver: SolverConfig
    smoother: SmootherConfig
    output: OutputConfig


class ConfigLoader:
    @staticmethod
    def parse_args() -> argparse.Namespace:
        """
        Parse command-line arguments.
        """
        parser = argparse.ArgumentParser(
            description="Test the encoder-decoder and two-level solver models."
        )
        parser.add_argument(
            "--config",
            type=str,
            required=True,
            help="Path to the TOML configuration file.",
        )
        return parser.parse_args()

    @staticmethod
    def parse_config(config_path: str) -> Config:
        """
        Parse the TOML configuration file.
        """
        with open(config_path, "r") as f:
            config_dict = toml.load(f)
        ConfigLoader.assert_minimal_config(config_dict)
        return Config(
            mesh=MeshConfig(**config_dict["mesh"]),
            data_gen=DataGenConfig(**config_dict["data_gen"]),
            model=ModelConfig(**config_dict["model"]),
            optimization=OptimizationConfig(**config_dict["optimization"]),
            training=TrainingConfig(**config_dict["training"]),
            coarsening=CoarseningConfig(**config_dict["coarsening"]),
            solver=SolverConfig(**config_dict["solver"]),
            smoother=SmootherConfig(**config_dict["smoother"]),
            output=OutputConfig(**config_dict["output"]),
        )

    @staticmethod
    def assert_minimal_config(config: dict) -> None:
        """
        Assert that the minimal configuration is present.
        """
        required_keys = [
            "mesh",
            "data_gen",
            "model",
            "optimization",
            "training",
            "coarsening",
            "solver",
            "smoother",
            "output",
        ]
        for key in required_keys:
            if key not in config:
                raise ValueError(
                    f"Missing required configuration section: {key}"
                )
