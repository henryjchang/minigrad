# minigrad
A mini autograd engine and neural network library with a PyTorch-like API.

The autograd engine implements backpropagation over a dynamically built DAG for a subset of `Tensor` operations. The order in which gradients are computed is determined through a topological sort of the computational graph nodes. 

The neural network library is primarily for demonstrating usage of the autograd engine, with implementations for a `nn.Linear` layer, `nn.ReLU` activation, `SGD` optimizer, and `cross_entropy` loss. It also has some nuggets of insight into how neural network layers are nested in a `Module` and what a `Parameter` is.

The code in this repo is primarily a "fork" of [the backprop chapter from ARENA 3.0](https://github.com/callummcdougall/ARENA_3.0/tree/main/chapter0_fundamentals/exercises/part4_backprop). I highly recommend the material in ARENA 3.0 overall. This repo was inspired by https://github.com/karpathy/micrograd; it's worth comparing the implementations to see the similarities and differences.

### Autograd Example Usage

```
from engine.tensor import Tensor

a = Tensor([1, 2, 3], requires_grad=True)
b = 2
c = a * b
d = 2
e = d * c
e.backward(end_grad=np.array([1.0, 1.0, 1.0]))

# Expected grads
assert e.grad is None
assert c.grad is None
assert a.grad is not None
assert np.allclose(a.grad.array, np.array([4.0, 4.0, 4.0]))
```

### Training A Neural Net

Check out `demo.ipynb` for an example of training a 2-layer neural network (MLP) on the MNIST dataset. We borrow PyTorch's `DataLoader` for an iterator through the dataset. All other training operations come from this repo.

### Running tests

To run the unit tests simply run:

```
pytest test
```
