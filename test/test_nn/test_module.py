from nn.core.module import Module
from nn.core.parameter import Parameter
from engine.tensor import Tensor

def test_module():
    inner_mod = Module()
    inner_mod.param1 = Parameter(Tensor([1.0]))
    inner_mod.param2 = Parameter(Tensor([2.0]))

    mod = Module()
    mod.inner = inner_mod
    mod.param3 = Parameter(Tensor([3.0]))

    assert list(mod.modules()) == [mod.inner]
    assert list(mod.parameters()) == [
        mod.param3,
        mod.inner.param1,
        mod.inner.param2,
    ], "parameters should come before submodule parameters"
