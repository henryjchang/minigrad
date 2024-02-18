import numpy as np
import warnings

from engine.tensor import Tensor
from nn.functional.cross_entropy import cross_entropy

def test_cross_entropy():
    logits = Tensor([
        [float("-inf"), float("-inf"), float("-inf"), 0], 
        [1/4, 1/4, 1/4, 1/4], 
        [float("-inf"), 0, 0, 0]
    ])
    true_labels = Tensor([3, 0, 0])
    expected = Tensor([0.0, np.log(4), float("inf")])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        actual = cross_entropy(logits, true_labels)
    np.testing.assert_allclose(actual.array, expected.array)
