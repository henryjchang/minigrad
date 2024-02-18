import numpy as np
from engine.tensor import Tensor
from nn.linear import Linear

def test_linear():
    linear = Linear(3, 4)
    assert isinstance(linear.weight, Tensor)
    assert linear.weight.requires_grad

    input = Tensor([[1.0, 2.0, 3.0]])
    output = linear(input)
    assert output.requires_grad

    expected_output = input @ linear.weight.T + linear.bias
    np.testing.assert_allclose(output.array, expected_output.array)
