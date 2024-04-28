# encoder-decoder

FEM encoder-decoder, a la V-cycle

# Structure of the repository

```
.
├── LICENSE
├── README.md
└── src
    ├── data_gen.py
    ├── encoder.py
    ├── fes.py
    ├── geo2d.py
    ├── initializers.py
    ├── layers.py
    ├── losses.py
    ├── metrics.py
    ├── playground.py
    └── regularizers.py
```

# Description

Here is a brief description of the files in the repository:

## data_gen.py -- Data generation

This file contains the `ABC` class `DataGenerator` which is used to generate the data for the training of the neural network.
It constructs finite element matrices for different stable equations in NGSolve.

### ABC class `DataGenerator`

This class is an abstract base class for the data generation.
So far, it requires the implementation of the following methods:

- `make_operator` -- use the FEM matrices to construct a linear operator, e.g., a `scipy.linalg.LinearOperator`.
- `make_sparse_operator` -- use the FEM matrices to construct a sparse matrix, e.g., a `scipy.sparse.csr_matrix`.
- `make_solver` -- use the FEM matrices to construct a direct solver. No example provided yet.
- `make_sparse_solver` -- use the FEM matrices to construct a sparse solver, e.g., a `scipy.linalg.spilu`.
- `make_data` -- small utility function to generate the data for the training. It uses the previous methods (members) to construct the data.
- `from_smooth` -- a class method to construct the data from a smooth function. It uses the previous methods (members) to construct the data.
- `from_random` -- a class method to construct the data from a random function. It uses the previous methods (members) to construct the data.
- `get_gf` -- a utility function to get the grid function with matching finite element space and dimensions as the data.

### Current classes derived from `DataGenerator`

Current classes derived from `DataGenerator`:

- `LaplaceDGen` -- data generation for the Laplace equation.

## encoder.py -- encoder-decoder implementation

We implement the encoder-decoder-like neural network in this file.
There classes are derived from Keras' `Model` class.
In particular, we have the `PseudoVcycle` and the `PseudoMG` classes.
Cf. unpublished notes by P. S. Vassilevski.

## fes.py -- Finite element spaces

This class implements the bilinear form for different equations in NGSolve.
So far, we have a general convection-diffusion equation.

## geo2d.py -- Geometry

This file constructs different geometries.
They are employed in `fes.py` to construct the finite element spaces.

### Available geometries

- `make_unit_square` -- a unit square.
- `make_l_shape` -- an L-shaped domain.

## initializers.py -- Initializers

This file contains the initializers for some variables in the neural network.
This is just made to provide a constant matrix to one of the layers.

## layers.py -- Layers

This file contains the layers for the neural network.

### Available layers

- `LinearLayer` -- a linear layer: no activation function, no bias.

## losses.py -- Losses

This file contains the losses for the neural network.

### Available losses

- `projected_l2_loss` -- a projected L2 loss into the image of a linear operator. It requires using partial functions.
- `l2_l1_loss` -- a L2-L1 loss.

## metrics.py -- Metrics

This file contains the metrics for the neural network.
Metrics are used to evaluate the performance of the neural network, but they are not used in the training.

## playground.py -- Playground

This file is used to test the different classes and functions in the repository.
