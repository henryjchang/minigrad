from typing import Any, Callable
from collections import defaultdict
import numpy as np

Arr = np.ndarray

grad_tracking_enabled = True #TODO: refactor

# forward declarations
class Tensor:
    def __init__(self):
        pass
class Recipe:
    def __init__(self):
        pass


def unbroadcast(broadcasted: Arr, original: Arr) -> Arr:
    '''
    Sum 'broadcasted' until it has the shape of 'original'.

    broadcasted: An array that was formerly of the same shape of 'original' and was expanded by broadcasting rules.
    '''
    # SOLUTION

    # Step 1: sum and remove prepended dims, so both arrays have same number of dims
    n_dims_to_sum = len(broadcasted.shape) - len(original.shape)
    broadcasted = broadcasted.sum(axis=tuple(range(n_dims_to_sum)))

    # Step 2: sum over dims which were originally 1 (but don't remove them)
    dims_to_sum = tuple([
        i for i, (o, b) in enumerate(zip(original.shape, broadcasted.shape))
        if o == 1 and b > 1
    ])
    broadcasted = broadcasted.sum(axis=dims_to_sum, keepdims=True)

    return broadcasted


def wrap_forward_fn(numpy_func: Callable, is_differentiable=True) -> Callable:
    '''
    numpy_func: Callable
        takes any number of positional arguments, some of which may be NumPy arrays, and
        any number of keyword arguments which we aren't allowing to be NumPy arrays at
        present. It returns a single NumPy array.

    is_differentiable:
        if True, numpy_func is differentiable with respect to some input argument, so we
        may need to track information in a Recipe. If False, we definitely don't need to
        track information.

    Return: Callable
        It has the same signature as numpy_func, except wherever there was a NumPy array,
        this has a Tensor instead.
    '''

    def tensor_func(*args: Any, **kwargs: Any) -> "Tensor":
        # SOLUTION

        # Get all function arguments as non-tensors (i.e. either ints or arrays)
        arg_arrays = tuple([(a.array if isinstance(a, Tensor) else a) for a in args])

        # Calculate the output (which is a numpy array)
        out_arr = numpy_func(*arg_arrays, **kwargs)

        # Find whether the tensor requires grad (need to check if ANY of the inputs do)
        requires_grad = grad_tracking_enabled and is_differentiable and any([
            (isinstance(a, Tensor) and (a.requires_grad or a.recipe is not None)) for a in args
        ])

        # Create the output tensor from the underlying data and the requires_grad flag
        out = Tensor(out_arr, requires_grad)

        # If requires_grad, then create a recipe
        if requires_grad:
            parents = {idx: a for idx, a in enumerate(args) if isinstance(a, Tensor)}
            out.recipe = Recipe(numpy_func, arg_arrays, kwargs, parents)

        return out

    return tensor_func


# Used for registering backward functions
class BackwardFuncLookup:
    def __init__(self) -> None:
        # SOLUTION
        self.back_funcs: defaultdict[Callable, dict[int, Callable]] = defaultdict(dict)

    def add_back_func(self, forward_fn: Callable, arg_position: int, back_fn: Callable) -> None:
        # SOLUTION
        self.back_funcs[forward_fn][arg_position] = back_fn

    def get_back_func(self, forward_fn: Callable, arg_position: int) -> Callable:
        # SOLUTION
        return self.back_funcs[forward_fn][arg_position]