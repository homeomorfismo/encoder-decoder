# encoder-decoder

FEM encoder-decoder, a la V-cycle

We use JAX to implement the encoder-decoder neural network for solving PDEs.
On the other hand, we use NGSolve to construct the finite element spaces and matrices.

In order to perform the aggregation steps (so we can define the sparsity pattern
of the matrices), we use PyAMG aggregation module.

# Structure of the repository

Currently, the repository is structured as follows:

```
.
├── LICENSE
├── README.md
├── src
│   ├── data_gen.py
│   ├── default.toml
│   ├── geo2d.py
│   ├── loss.py
│   ├── models.py
│   ├── pyproject.toml
│   ├── solver.py
│   ├── sparse.py
│   ├── test.py
│   └── utilities.py
└── write-up

```

Here is a brief description of the files in the repository:

## data_gen.py -- Data generation

This file contains the `ABC` class `DataGenerator` which is used to generate the data for the training of the neural network.
It constructs finite element matrices for different discretizations and PDEs, in NGSolve.

The new simplified data generator wraps and contains the NGSolve elements and forms.

### ABC class `DataGenerator`

This class is an abstract base class for the data generation.
Currently, all linear algebra data types are using `jax.numpy` as the backend.

### Current classes derived from `DataGenerator`


## model.py -- encoder-decoder implementation

JIT compiled functions for the encoder-decoder neural network.
We use the `jax` library to implement the neural network.
Cf. unpublished notes by [P. S. Vassilevski](https://web.pdx.edu/~panayot/).

## geo2d.py -- Geometry

This file constructs different geometries.
They are employed in `data_gen.py` to construct the finite element spaces.

### Available geometries

- `make_unit_square` -- a unit square `[0, 1] x [0, 1]`.
- `make_l_shape` -- an L-shaped domain `[-1, 1] x [-1, 1] \ [0, 1] x [-1, 0]`.

## loss.py -- Losses

This file contains the losses for the neural network, wrapped in `get_loss` and
`get_mg_loss` functions.
These functions return a `Callable` object that can be differentiated by `jax`.
Their argument is a description of the norm employed for the regularization
term.

It allows different regularization terms to be used in the loss function, such as
L2, L1, L0.

## test.py -- Testing and wrapping

This file is used to test the different classes and functions in the repository.
