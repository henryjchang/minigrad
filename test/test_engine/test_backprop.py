import numpy as np
from engine.tensor import Tensor

def test_backprop():
    a = Tensor([np.e, np.e**np.e], requires_grad=True)
    b = a.log()
    c = b.log()
    c.backward(end_grad=np.array([1.0, 1.0]))
    assert c.grad is None
    assert b.grad is None
    assert a.grad is not None
    assert np.allclose(a.grad.array, 1 / b.array / a.array)

def test_backprop_branching():
    a = Tensor([1, 2, 3], requires_grad=True)
    b = Tensor([1, 2, 3], requires_grad=True)
    c = a * b
    c.backward(end_grad=np.array([1.0, 1.0, 1.0]))
    assert np.allclose(a.grad.array, b.array)
    assert np.allclose(b.grad.array, a.array)

def test_backprop_requires_grad_false():
    a = Tensor([1, 2, 3], requires_grad=True)
    b = Tensor([1, 2, 3], requires_grad=False)
    c = a * b
    c.backward(end_grad=np.array([1.0, 1.0, 1.0]))
    assert np.allclose(a.grad.array, b.array)
    assert b.grad is None

def test_backprop_float_arg():
    a = Tensor([1, 2, 3], requires_grad=True)
    b = 2
    c = a * b
    d = 2
    e = d * c
    e.backward(end_grad=np.array([1.0, 1.0, 1.0]))
    assert e.grad is None
    assert c.grad is None
    assert a.grad is not None
    assert np.allclose(a.grad.array, np.array([4.0, 4.0, 4.0]))

def test_backprop_shared_parent():
    a = 2
    b = Tensor([1, 2, 3], requires_grad=True)
    c = 3
    d = a * b
    e = b * c
    f = d * e
    f.backward(end_grad=np.array([1.0, 1.0, 1.0]))
    assert f.grad is None
    assert b.grad is not None
    assert np.allclose(b.grad.array, np.array([12.0, 24.0, 36.0])), "Multiple nodes may have the same parent."
