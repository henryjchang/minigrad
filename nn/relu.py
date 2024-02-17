from engine.tensor import Tensor
from engine.functions import maximum
from .core.module import Module

class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return maximum(x, 0.0)