from typing import Any, Callable, Optional, Union, Dict, Tuple
from dataclasses import dataclass
import numpy as np

# custom
from engine.function_utils import unbroadcast, BackwardFuncLookup
from engine.topological_sort import sorted_computational_graph

Arr = np.ndarray

grad_tracking_enabled = True # TODO: refactor

@dataclass(frozen=True)
class Recipe:
    '''Extra information necessary to run backpropagation.'''

    func: Callable
    "The 'inner' NumPy function that does the actual forward computation."
    "Note, we call it 'inner' to distinguish it from the wrapper we'll use for it later on."

    args: tuple
    "The input arguments passed to func."
    "For instance, if func was np.sum then args would be a length-1 tuple containing the tensor to be summed."

    kwargs: Dict[str, Any]
    "Keyword arguments passed to func."
    "For instance, if func was np.sum then kwargs might contain 'dim' and 'keepdims'."

    parents: Dict[int, "Tensor"]
    "Map from positional argument index to the Tensor at that position, in order to be able to pass gradients back along the computational graph."

class Tensor:
    '''
    A drop-in replacement for torch.Tensor supporting a subset of features.
    '''

    array: Arr
    "The underlying array. Can be shared between multiple Tensors."
    requires_grad: bool
    "If True, calling functions or methods on this tensor will track relevant data for backprop."
    grad: Optional["Tensor"]
    "Backpropagation will accumulate gradients into this field."
    recipe: Optional[Recipe]
    "Extra information necessary to run backpropagation."

    def __init__(self, array: Union[Arr, list], requires_grad=False):
        self.array = array if isinstance(array, Arr) else np.array(array)
        if self.array.dtype == np.float64:
            self.array = self.array.astype(np.float32)
        self.requires_grad = requires_grad
        self.grad = None
        self.recipe = None
        "If not None, this tensor's array was created via recipe.func(*recipe.args, **recipe.kwargs)."

    def __neg__(self) -> "Tensor":
        return negative(self)

    def __add__(self, other) -> "Tensor":
        return add(self, other)

    def __radd__(self, other) -> "Tensor":
        return add(other, self)

    def __sub__(self, other) -> "Tensor":
        return subtract(self, other)

    def __rsub__(self, other) -> "Tensor":
        return subtract(other, self)

    def __mul__(self, other) -> "Tensor":
        return multiply(self, other)

    def __rmul__(self, other):
        return multiply(other, self)

    def __truediv__(self, other):
        return true_divide(self, other)

    def __rtruediv__(self, other):
        return true_divide(other, self)

    def __matmul__(self, other):
        return matmul(self, other)

    def __rmatmul__(self, other):
        return matmul(other, self)

    def __eq__(self, other):
        return eq(self, other)

    def __repr__(self) -> str:
        return f"Tensor({repr(self.array)}, requires_grad={self.requires_grad})"

    def __len__(self) -> int:
        if self.array.ndim == 0:
            raise TypeError
        return self.array.shape[0]

    def __hash__(self) -> int:
        return id(self)

    def __getitem__(self, index) -> "Tensor":
        return getitem(self, index)

    def add_(self, other: "Tensor", alpha: float = 1.0) -> "Tensor":
       add_(self, other, alpha=alpha)
       return self

    @property
    def T(self) -> "Tensor":
        return permute(self, axes=(-1, -2))

    def item(self):
        return self.array.item()

    def sum(self, dim=None, keepdim=False) -> "Tensor":
        return sum(self, dim=dim, keepdim=keepdim)

    def log(self) -> "Tensor":
        return log(self)

    def exp(self) -> "Tensor":
        return exp(self)

    def reshape(self, new_shape) -> "Tensor":
        return reshape(self, new_shape)

    def expand(self, new_shape) -> "Tensor":
        return expand(self, new_shape)

    def permute(self, dims) -> "Tensor":
        return permute(self, dims)

    def maximum(self, other) -> "Tensor":
        return maximum(self, other)

    def relu(self) -> "Tensor":
        return maximum(self, 0.0)

    def argmax(self, dim=None, keepdim=False) -> "Tensor":
        return argmax(self, dim=dim, keepdim=keepdim)

    def uniform_(self, low: float, high: float) -> "Tensor":
        self.array[:] = np.random.uniform(low, high, self.array.shape)
        return self

    def backward(self, end_grad: Union[Arr, "Tensor", None] = None) -> None:
        if isinstance(end_grad, Arr):
            end_grad = Tensor(end_grad)
        return backprop(self, end_grad)

    def size(self, dim: Optional[int] = None):
        if dim is None:
            return self.shape
        return self.shape[dim]

    @property
    def shape(self):
        return self.array.shape

    @property
    def ndim(self):
        return self.array.ndim

    @property
    def is_leaf(self):
        '''Same as https://pytorch.org/docs/stable/generated/torch.Tensor.is_leaf.html'''
        if self.requires_grad and self.recipe and self.recipe.parents:
            return False
        return True

    def __bool__(self):
        if np.array(self.shape).prod() != 1:
            raise RuntimeError("bool value of Tensor with more than one value is ambiguous")
        return bool(self.item())

def empty(*shape: int) -> Tensor:
    '''Like torch.empty.'''
    return Tensor(np.empty(shape))

def zeros(*shape: int) -> Tensor:
    '''Like torch.zeros.'''
    return Tensor(np.zeros(shape))

def arange(start: int, end: int, step=1) -> Tensor:
    '''Like torch.arange(start, end).'''
    return Tensor(np.arange(start, end, step=step))

def tensor(array: Arr, requires_grad=False) -> Tensor:
    '''Like torch.tensor.'''
    return Tensor(array, requires_grad=requires_grad)

def backprop(end_node: Tensor, end_grad: Optional[Tensor] = None) -> None:
    '''Accumulates gradients in the grad field of each leaf node.

    tensor.backward() is equivalent to backprop(tensor).

    end_node:
        The rightmost node in the computation graph.
        If it contains more than one element, end_grad must be provided.
    end_grad:
        A tensor of the same shape as end_node.
        Set to 1 if not specified and end_node has only one element.
    '''
    # Get value of end_grad_arr
    end_grad_arr = np.ones_like(end_node.array) if end_grad is None else end_grad.array

    # Create dict to store gradients
    grads: Dict[Tensor, Arr] = {end_node: end_grad_arr}

    # Iterate through the computational graph
    for node in sorted_computational_graph(end_node):

        # Get the outgradient from the grads dict
        outgrad = grads.pop(node)
        # We only store the gradients if this node is a leaf & requires_grad is true
        if node.is_leaf and node.requires_grad:
            # Add the gradient to this node's grad (need to deal with special case grad=None)
            if node.grad is None:
                node.grad = Tensor(outgrad)
            else:
                node.grad.array += outgrad

        # If node has no parents, then the backtracking through the computational
        # graph ends here
        if node.recipe is None or node.recipe.parents is None:
            continue

        # If node has a recipe, then we iterate through parents (which is a dict of {arg_posn: tensor})
        for argnum, parent in node.recipe.parents.items():

            # Get the backward function corresponding to the function that created this node
            back_fn = BACK_FUNCS.get_back_func(node.recipe.func, argnum)

            # Use this backward function to calculate the gradient
            in_grad = back_fn(outgrad, node.array, *node.recipe.args, **node.recipe.kwargs)

            # Add the gradient to this node in the dictionary `grads`
            # Note that we only set node.grad (from the grads dict) in the code block above
            if parent not in grads:
                grads[parent] = in_grad
            else:
                grads[parent] += in_grad

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
        # Get all function arguments as non-tensors (i.e. either ints or arrays)
        arg_arrays = tuple([(a.array if isinstance(a, Tensor) else a) for a in args])

        # Calculate the output (which is a numpy array)
        out_arr = numpy_func(*arg_arrays, **kwargs)

        # Find whether the tensor requires grad (need to check if ANY of the inputs do)
        global grad_tracking_enabled
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

# create a lookup table for backward functions
BACK_FUNCS = BackwardFuncLookup()

'''
ELEMENT-WISE OPERATIONS
'''

'''add'''
add = wrap_forward_fn(np.add)
BACK_FUNCS.add_back_func(np.add, 0, lambda grad_out, out, x, y: unbroadcast(grad_out, x))
BACK_FUNCS.add_back_func(np.add, 1, lambda grad_out, out, x, y: unbroadcast(grad_out, y))

'''subtract'''
subtract = wrap_forward_fn(np.subtract)
BACK_FUNCS.add_back_func(np.subtract, 0, lambda grad_out, out, x, y: unbroadcast(grad_out, x))
BACK_FUNCS.add_back_func(np.subtract, 1, lambda grad_out, out, x, y: unbroadcast(-grad_out, y))

'''true_divide'''
true_divide = wrap_forward_fn(np.true_divide)
BACK_FUNCS.add_back_func(np.true_divide, 0, lambda grad_out, out, x, y: unbroadcast(grad_out/y, x))
BACK_FUNCS.add_back_func(np.true_divide, 1, lambda grad_out, out, x, y: unbroadcast(grad_out*(-x/y**2), y))

'''multiply'''
def multiply_back0(grad_out: Arr, out: Arr, x: Arr, y: Union[Arr, float]) -> Arr:
    '''Backwards function for x * y wrt argument 0 aka x.'''
    if not isinstance(y, Arr):
        y = np.array(y)
    return unbroadcast(y * grad_out, x)

def multiply_back1(grad_out: Arr, out: Arr, x: Union[Arr, float], y: Arr) -> Arr:
    '''Backwards function for x * y wrt argument 1 aka y.'''
    if not isinstance(x, Arr):
        x = np.array(x)
    return unbroadcast(x * grad_out, y)

multiply = wrap_forward_fn(np.multiply)
BACK_FUNCS.add_back_func(np.multiply, 0, multiply_back0)
BACK_FUNCS.add_back_func(np.multiply, 1, multiply_back1)

'''
SINGLE-TENSOR DIFFERENTIABLE FUNCTIONS
'''

'''negative'''
def negative_back(grad_out: Arr, out: Arr, x: Arr) -> Arr:
    '''Backward function for f(x) = -x elementwise.'''
    return unbroadcast(-grad_out, x)

negative = wrap_forward_fn(np.negative)
BACK_FUNCS.add_back_func(np.negative, 0, negative_back)

'''exp'''
def exp_back(grad_out: Arr, out: Arr, x: Arr) -> Arr:
    return out * grad_out

exp = wrap_forward_fn(np.exp)
BACK_FUNCS.add_back_func(np.exp, 0, exp_back)

'''reshape'''
def reshape_back(grad_out: Arr, out: Arr, x: Arr, new_shape: tuple) -> Arr:
    return np.reshape(grad_out, x.shape)

reshape = wrap_forward_fn(np.reshape)
BACK_FUNCS.add_back_func(np.reshape, 0, reshape_back)

'''permute'''
def invert_transposition(axes: tuple) -> tuple:
    '''
    axes: tuple indicating a transition

    Returns: inverse of this transposition, i.e. the array `axes_inv` s.t. we have:
        np.transpose(np.transpose(x, axes), axes_inv) == x

    Some examples:
        (1, 0)    --> (1, 0)     # this is reversing a simple 2-element transposition
        (0, 2, 1) --> (0, 1, 2)
        (1, 2, 0) --> (2, 0, 1)  # this is reversing the order of a 3-cycle
    '''
    # Slick solution:
    return tuple(np.argsort(axes))

    # Slower solution, which makes it clearer what operation is happening:
    reversed_transposition_map = {num: idx for (idx, num) in enumerate(axes)}
    reversed_transposition = [reversed_transposition_map[idx] for idx in range(len(axes))]
    return tuple(reversed_transposition)

def permute_back(grad_out: Arr, out: Arr, x: Arr, axes: tuple) -> Arr:
    return np.transpose(grad_out, invert_transposition(axes))

BACK_FUNCS.add_back_func(np.transpose, 0, permute_back)
permute = wrap_forward_fn(np.transpose)

'''expand'''
def expand_back(grad_out: Arr, out: Arr, x: Arr, new_shape: tuple) -> Arr:
    return unbroadcast(grad_out, x)

def _expand(x: Arr, new_shape) -> Arr:
    '''
    Like torch.expand, calling np.broadcast_to internally.

    Note torch.expand supports -1 for a dimension size meaning "don't change the size".
    np.broadcast_to does not natively support this.
    '''
    n_added = len(new_shape) - x.ndim
    shape_non_negative = tuple([x.shape[i - n_added] if s == -1 else s for i, s in enumerate(new_shape)])
    return np.broadcast_to(x, shape_non_negative)

expand = wrap_forward_fn(_expand)
BACK_FUNCS.add_back_func(_expand, 0, expand_back)

'''sum'''
def _sum(x: Arr, dim=None, keepdim=False) -> Arr:
    '''Like torch.sum, calling np.sum internally.'''
    # need to be careful with sum, because kwargs have different names in torch and numpy
    return np.sum(x, axis=dim, keepdims=keepdim)

def sum_back(grad_out: Arr, out: Arr, x: Arr, dim=None, keepdim=False):
    '''Basic idea: repeat grad_out over the dims along which x was summed'''
    # If grad_out is a scalar, we need to make it a tensor (so we can expand it later)
    if not isinstance(grad_out, Arr):
        grad_out = np.array(grad_out)

    # If dim=None, this means we summed over all axes, and we want to repeat back to input shape
    if dim is None:
        dim = list(range(x.ndim))

    # If keepdim=False, then we need to add back in dims, so grad_out and x have same number of dims
    if keepdim == False:
        grad_out = np.expand_dims(grad_out, dim)

    # Finally, we repeat grad_out along the dims over which x was summed
    return np.broadcast_to(grad_out, x.shape)

sum = wrap_forward_fn(_sum)
BACK_FUNCS.add_back_func(_sum, 0, sum_back)

'''getitem'''
Index = Union[int, Tuple[int, ...], Tuple[Arr], Tuple['Tensor']]

def coerce_index(index: Index) -> Union[int, Tuple[int, ...], Tuple[Arr]]:
    '''
    If index is of type signature `Tuple[Tensor]`, converts it to `Tuple[Arr]`.
    '''
    if isinstance(index, tuple) and all(isinstance(i, Tensor) for i in index):
        return tuple([i.array for i in index])
    else:
        return index
    
def _getitem(x: Arr, index: Index) -> Arr:
    '''Like x[index] when x is a torch.Tensor.'''
    return x[coerce_index(index)]

def getitem_back(grad_out: Arr, out: Arr, x: Arr, index: Index):
    '''
    Backwards function for _getitem.

    Hint: use np.add.at(a, indices, b)
    This function works just like a[indices] += b, except that it allows for repeated indices.
    '''
    new_grad_out = np.full_like(x, 0)
    np.add.at(new_grad_out, coerce_index(index), grad_out)
    return new_grad_out

getitem = wrap_forward_fn(_getitem)
BACK_FUNCS.add_back_func(_getitem, 0, getitem_back)

'''log'''
def log_back(grad_out: Arr, out: Arr, x: Arr) -> Arr:
    '''Backwards function for f(x) = log(x)

    grad_out: Gradient of some loss wrt out
    out: the output of np.log(x).
    x: the input of np.log.

    Return: gradient of the given loss wrt x
    '''
    return grad_out / x

log = wrap_forward_fn(np.log)
BACK_FUNCS.add_back_func(np.log, 0, log_back)

'''
NON-DIFFERENTIABLE FUNCTIONS
'''

'''argmax'''
def _argmax(x: Arr, dim=None, keepdim=False):
    '''Like torch.argmax.'''
    return np.expand_dims(np.argmax(x, axis=dim), axis=([] if dim is None else dim))

argmax = wrap_forward_fn(_argmax, is_differentiable=False)

'''eq'''
eq = wrap_forward_fn(np.equal, is_differentiable=False)


'''
MIXED SCALAR-TENSOR OPERATIONS
'''

'''
maximum
'''
def maximum_back0(grad_out: Arr, out: Arr, x: Arr, y: Arr):
    '''Backwards function for max(x, y) wrt x.'''
    bool_sum = ((x > y) + 0.5 * (x == y))
    return unbroadcast(grad_out * bool_sum, x)

def maximum_back1(grad_out: Arr, out: Arr, x: Arr, y: Arr):
    '''Backwards function for max(x, y) wrt y.'''
    bool_sum = ((x < y) + 0.5 * (x == y))
    return unbroadcast(grad_out * bool_sum, y)

maximum = wrap_forward_fn(np.maximum)

BACK_FUNCS.add_back_func(np.maximum, 0, maximum_back0)
BACK_FUNCS.add_back_func(np.maximum, 1, maximum_back1)


'''
matmul
'''
def _matmul2d(x: Arr, y: Arr) -> Arr:
    '''Matrix multiply restricted to the case where both inputs are exactly 2D.'''
    return x @ y

def matmul2d_back0(grad_out: Arr, out: Arr, x: Arr, y: Arr) -> Arr:
    return grad_out @ y.T

def matmul2d_back1(grad_out: Arr, out: Arr, x: Arr, y: Arr) -> Arr:
    return x.T @ grad_out

matmul = wrap_forward_fn(_matmul2d)
BACK_FUNCS.add_back_func(_matmul2d, 0, matmul2d_back0)
BACK_FUNCS.add_back_func(_matmul2d, 1, matmul2d_back1)

'''
IN-PLACE OPERATIONS

Supporting in-place operations introduces substantial complexity and generally
doesn't help performance that much. The problem is that if any of the inputs 
used in the backward function have been modified in-place since the forward
pass, then the backward function will incorrectly calculate using the modified
version.

PyTorch will warn you when this causes a problem with the error "RuntimeError: 
a leaf Variable that requires grad is being used in an in-place operation.".

However here we skip the warning implementation.
'''
def add_(x: Tensor, other: Tensor, alpha: float = 1.0) -> Tensor:
    '''Like torch.add_. Compute x += other * alpha in-place and return tensor.'''
    np.add(x.array, other.array * alpha, out=x.array)
    return x
