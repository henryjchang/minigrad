from engine.tensor import Tensor

class Parameter(Tensor):
    def __init__(self, tensor: Tensor, requires_grad=True):
        '''Share the array with the provided tensor.'''
        # SOLUTION
        return super().__init__(tensor.array, requires_grad=requires_grad)

    def __repr__(self):
        # SOLUTION
        return f"Parameter containing:\n{super().__repr__()}"


# x = Tensor([1.0, 2.0, 3.0])
# p = Parameter(x)
# assert p.requires_grad
# assert p.array is x.array
# assert repr(p) == "Parameter containing:\nTensor(array([1., 2., 3.], dtype=float32), requires_grad=True)"
# x.add_(Tensor(np.array(2.0)))
# assert np.allclose(
#     p.array, np.array([3.0, 4.0, 5.0])
# ), "in-place modifications to the original tensor should affect the parameter"