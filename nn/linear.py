import numpy as np
from typing import Optional

from .core.module import Module
from .core.parameter import Parameter
from engine.tensor import Tensor

class Linear(Module):
    weight: Parameter
    bias: Optional[Parameter]

    def __init__(self, in_features: int, out_features: int, bias=True):
        '''
        A simple linear (technically, affine) transformation.

        The fields should be named `weight` and `bias` for compatibility with PyTorch.
        If `bias` is False, set `self.bias` to None.
        '''
        super().__init__()
        # SOLUTION
        self.in_features = in_features
        self.out_features = out_features

        # sf needs to be a float
        sf = in_features ** -0.5

        weight = sf * Tensor(2 * np.random.rand(out_features, in_features) - 1)
        self.weight = Parameter(weight)

        if bias:
            bias = sf * Tensor(2 * np.random.rand(out_features,) - 1)
            self.bias = Parameter(bias)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        '''
        x: shape (*, in_features)
        Return: shape (*, out_features)
        '''
        # SOLUTION
        out = x @ self.weight.T
        # Note, transpose has been defined as .permute(-1, -2) in the Tensor class
        if self.bias is not None:
            out = out + self.bias
        return out

    def extra_repr(self) -> str:
        # note, we need to use `self.bias is not None`, because `self.bias` is either a tensor or None, not bool
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


# linear = Linear(3, 4)
# assert isinstance(linear.weight, Tensor)
# assert linear.weight.requires_grad

# input = Tensor([[1.0, 2.0, 3.0]])
# output = linear(input)
# assert output.requires_grad

# expected_output = input @ linear.weight.T + linear.bias
# np.testing.assert_allclose(output.array, expected_output.array)

# print("All tests for `Linear` passed!")