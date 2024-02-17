from engine.tensor import Tensor, maximum
from .core.module import Module

class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return maximum(x, 0.0)