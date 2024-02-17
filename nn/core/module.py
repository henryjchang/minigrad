from typing import Any, Dict, Iterator, Union

from nn.core.parameter import Parameter

class Module:
    _modules: Dict[str, "Module"]
    _parameters: Dict[str, 'Parameter']

    def __init__(self):
        self._modules = {}
        self._parameters = {}

    def modules(self):
        '''Return the direct child modules of this module.'''
        return self.__dict__["_modules"].values()

    def parameters(self, recurse: bool = True) -> Iterator['Parameter']:
        '''
        Return an iterator over Module parameters.

        recurse: if True, the iterator includes parameters of submodules, recursively.
        '''
        # SOLUTION
        parameters_list = list(self.__dict__["_parameters"].values())
        if recurse:
            for mod in self.modules():
                parameters_list.extend(list(mod.parameters(recurse=True)))
        return iter(parameters_list)

    def __setattr__(self, key: str, val: Any) -> None:
        '''
        If val is a Parameter or Module, store it in the appropriate _parameters or _modules dict.
        Otherwise, call __setattr__ from the superclass.
        '''
        # SOLUTION
        if isinstance(val, Parameter):
            self.__dict__["_parameters"][key] = val
        elif isinstance(val, Module):
            self.__dict__["_modules"][key] = val
        else:
            super().__setattr__(key, val)

    def __getattr__(self, key: str) -> Union['Parameter', "Module"]:
        '''
        If key is in _parameters or _modules, return the corresponding value.
        Otherwise, raise KeyError.
        '''
        # SOLUTION
        if key in self.__dict__["_parameters"]:
            return self.__dict__["_parameters"][key]

        if key in self.__dict__["_modules"]:
            return self.__dict__["_modules"][key]

        raise KeyError(key)


    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self):
        raise NotImplementedError("Subclasses must implement forward!")

    def __repr__(self):
        def _indent(s_, numSpaces):
            return re.sub("\n", "\n" + (" " * numSpaces), s_)
        lines = [f"({key}): {_indent(repr(module), 2)}" for key, module in self._modules.items()]
        return "".join([
            self.__class__.__name__ + "(",
            "\n  " + "\n  ".join(lines) + "\n" if lines else "", ")"
        ])


# class TestInnerModule(Module):
#     def __init__(self):
#         super().__init__()
#         self.param1 = Parameter(Tensor([1.0]))
#         self.param2 = Parameter(Tensor([2.0]))

# class TestModule(Module):
#     def __init__(self):
#         super().__init__()
#         self.inner = TestInnerModule()
#         self.param3 = Parameter(Tensor([3.0]))

# mod = TestModule()
# assert list(mod.modules()) == [mod.inner]
# assert list(mod.parameters()) == [
#     mod.param3,
#     mod.inner.param1,
#     mod.inner.param2,
# ], "parameters should come before submodule parameters"
# print("Manually verify that the repr looks reasonable:")
# print(mod)