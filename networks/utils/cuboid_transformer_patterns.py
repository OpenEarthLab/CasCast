"""Patterns for cuboid self-attention / cross attention"""

import functools
"""Create a registry."""

from typing import Optional, List
import json
from json import JSONDecodeError


class Registry:
    """Create the registry that will map name to object. This facilitates the users to create
    custom registry.

    Parameters
    ----------
    name
        The name of the registry

    Examples
    --------

    >>> from earthformer.utils.registry import Registry
    >>> # Create a registry
    >>> MODEL_REGISTRY = Registry('MODEL')
    >>>
    >>> # To register a class/function with decorator
    >>> @MODEL_REGISTRY.register()
...     class MyModel:
...         pass
    >>> @MODEL_REGISTRY.register()
...     def my_model():
...         return
    >>>
    >>> # To register a class object with decorator and provide nickname:
    >>> @MODEL_REGISTRY.register('test_class')
...     class MyModelWithNickName:
...         pass
    >>> @MODEL_REGISTRY.register('test_function')
...     def my_model_with_nick_name():
...         return
    >>>
    >>> # To register a class/function object by function call
...     class MyModel2:
...         pass
    >>> MODEL_REGISTRY.register(MyModel2)
    >>> # To register with a given name
    >>> MODEL_REGISTRY.register('my_model2', MyModel2)
    >>> # To list all the registered objects:
    >>> MODEL_REGISTRY.list_keys()

['MyModel', 'my_model', 'test_class', 'test_function', 'MyModel2', 'my_model2']

    >>> # To get the registered object/class
    >>> MODEL_REGISTRY.get('test_class')

__main__.MyModelWithNickName

    """

    def __init__(self, name: str) -> None:
        self._name: str = name
        self._obj_map: dict[str, object] = dict()

    def _do_register(self, name: str, obj: object) -> None:
        assert (
            name not in self._obj_map
        ), "An object named '{}' was already registered in '{}' registry!".format(
            name, self._name
        )
        self._obj_map[name] = obj

    def register(self, *args):
        """
        Register the given object under either the nickname or `obj.__name__`. It can be used as
         either a decorator or not. See docstring of this class for usage.
        """
        if len(args) == 2:
            # Register an object with nick name by function call
            nickname, obj = args
            self._do_register(nickname, obj)
        elif len(args) == 1:
            if isinstance(args[0], str):
                # Register an object with nick name by decorator
                nickname = args[0]
                def deco(func_or_class: object) -> object:
                    self._do_register(nickname, func_or_class)
                    return func_or_class
                return deco
            else:
                # Register an object by function call
                self._do_register(args[0].__name__, args[0])
        elif len(args) == 0:
            # Register an object by decorator
            def deco(func_or_class: object) -> object:
                self._do_register(func_or_class.__name__, func_or_class)
                return func_or_class
            return deco
        else:
            raise ValueError('Do not support the usage!')

    def get(self, name: str) -> object:
        ret = self._obj_map.get(name)
        if ret is None:
            raise KeyError(
                "No object named '{}' found in '{}' registry!".format(
                    name, self._name
                )
            )
        return ret

    def list_keys(self) -> List:
        return list(self._obj_map.keys())

    def __repr__(self) -> str:
        s = '{name}(keys={keys})'.format(name=self._name,
                                         keys=self.list_keys())
        return s

    def create(self, name: str, *args, **kwargs) -> object:
        """Create the class object with the given args and kwargs

        Parameters
        ----------
        name
            The name in the registry
        args
        kwargs

        Returns
        -------
        ret
            The created object
        """
        obj = self.get(name)
        try:
            return obj(*args, **kwargs)
        except Exception as exp:
            print('Cannot create name="{}" --> {} with the provided arguments!\n'
                  '   args={},\n'
                  '   kwargs={},\n'
                  .format(name, obj, args, kwargs))
            raise exp

    def create_with_json(self, name: str, json_str: str):
        """

        Parameters
        ----------
        name
        json_str

        Returns
        -------

        """
        try:
            args = json.loads(json_str)
        except JSONDecodeError:
            raise ValueError('Unable to decode the json string: json_str="{}"'
                             .format(json_str))
        if isinstance(args, (list, tuple)):
            return self.create(name, *args)
        elif isinstance(args, dict):
            return self.create(name, **args)
        else:
            raise NotImplementedError('The format of json string is not supported! We only support '
                                      'list/dict. json_str="{}".'
                                      .format(json_str))
        
###############################################################################################################
###############################################################################################################
###############################################################################################################


CuboidSelfAttentionPatterns = Registry('CuboidSelfAttentionPattern')
CuboidCrossAttentionPatterns = Registry('CuboidCrossAttentionPatterns')

# basic patterns

def full_attention(input_shape):
    T, H, W, _ = input_shape
    cuboid_size = [(T, H, W)]
    strategy = [('l', 'l', 'l')]
    shift_size = [(0, 0, 0)]
    return cuboid_size, strategy, shift_size

def self_axial(input_shape):
    """Axial attention proposed in https://arxiv.org/abs/1912.12180

    Parameters
    ----------
    input_shape
        T, H, W

    Returns
    -------
    cuboid_size
    strategy
    shift_size
    """
    T, H, W, _ = input_shape
    cuboid_size = [(T, 1, 1), (1, H, 1), (1, 1, W)]
    strategy = [('l', 'l', 'l'), ('l', 'l', 'l'), ('l', 'l', 'l')]
    shift_size = [(0, 0, 0), (0, 0, 0), (0, 0, 0)]
    return cuboid_size, strategy, shift_size

def self_video_swin(input_shape, P=2, M=4):
    """Adopt the strategy in Video SwinTransformer https://arxiv.org/pdf/2106.13230.pdf"""
    T, H, W, _ = input_shape
    P = min(P, T)
    M = min(M, H, W)
    cuboid_size = [(P, M, M), (P, M, M)]
    strategy = [('l', 'l', 'l'), ('l', 'l', 'l')]
    shift_size = [(0, 0, 0), (P // 2, M // 2, M // 2)]

    return cuboid_size, strategy, shift_size

def self_divided_space_time(input_shape):
    T, H, W, _ = input_shape
    cuboid_size = [(T, 1, 1), (1, H, W)]
    strategy = [('l', 'l', 'l'), ('l', 'l', 'l')]
    shift_size = [(0, 0, 0), (0, 0, 0)]
    return cuboid_size, strategy, shift_size

# basic patterns
CuboidSelfAttentionPatterns.register('full', full_attention)
CuboidSelfAttentionPatterns.register('axial', self_axial)
CuboidSelfAttentionPatterns.register('video_swin', self_video_swin)
CuboidSelfAttentionPatterns.register('divided_st', self_divided_space_time)
# video_swin_PxM
for p in [1, 2, 4, 8, 10]:
    for m in [1, 2, 4, 8, 16, 32]:
        CuboidSelfAttentionPatterns.register(
            f'video_swin_{p}x{m}',
            functools.partial(self_video_swin,
                              P=p, M=m))

# our proposals
def self_spatial_lg_v1(input_shape, M=4):
    T, H, W, _ = input_shape

    if H <= M and W <= M:
        cuboid_size = [(T, 1, 1), (1, H, W)]
        strategy = [('l', 'l', 'l'), ('l', 'l', 'l')]
        shift_size = [(0, 0, 0), (0, 0, 0)]
    else:
        cuboid_size = [(T, 1, 1), (1, M, M), (1, M, M)]
        strategy = [('l', 'l', 'l'), ('l', 'l', 'l'), ('d', 'd', 'd')]
        shift_size = [(0, 0, 0), (0, 0, 0), (0, 0, 0)]
    return cuboid_size, strategy, shift_size


# Following are our proposed new patterns based on the CuboidSelfAttention design.
CuboidSelfAttentionPatterns.register('spatial_lg_v1', self_spatial_lg_v1)
# spatial_lg
for m in [1, 2, 4, 8, 16, 32]:
    CuboidSelfAttentionPatterns.register(
        f'spatial_lg_{m}',
        functools.partial(self_spatial_lg_v1,
                          M=m))

def self_axial_space_dilate_K(input_shape, K=2):
    T, H, W, _ = input_shape
    K = min(K, H, W)
    cuboid_size = [(T, 1, 1),
                   (1, H // K, 1), (1, H // K, 1),
                   (1, 1, W // K), (1, 1, W // K)]
    strategy = [('l', 'l', 'l'),
                ('d', 'd', 'd'), ('l', 'l', 'l'),
                ('d', 'd', 'd'), ('l', 'l', 'l'),]
    shift_size = [(0, 0, 0),
                  (0, 0, 0), (0, 0, 0),
                  (0, 0, 0), (0, 0, 0)]
    return cuboid_size, strategy, shift_size
for k in [2, 4, 8]:
    CuboidSelfAttentionPatterns.register(
        f'axial_space_dilate_{k}',
        functools.partial(self_axial_space_dilate_K,
                          K=k))


def cross_KxK(mem_shape, K):
    """

    Parameters
    ----------
    mem_shape
    K

    Returns
    -------
    cuboid_hw
    shift_hw
    strategy
    n_temporal
    """
    T_mem, H, W, _ = mem_shape
    K = min(K, H, W)
    cuboid_hw = [(K, K)]
    shift_hw = [(0, 0)]
    strategy = [('l', 'l', 'l')]
    n_temporal = [1]
    return cuboid_hw, shift_hw, strategy, n_temporal

def cross_KxK_lg(mem_shape, K):
    """

    Parameters
    ----------
    mem_shape
    K

    Returns
    -------
    cuboid_hw
    shift_hw
    strategy
    n_temporal
    """
    T_mem, H, W, _ = mem_shape
    K = min(K, H, W)
    cuboid_hw = [(K, K), (K, K)]
    shift_hw = [(0, 0), (0, 0)]
    strategy = [('l', 'l', 'l'), ('d', 'd', 'd')]
    n_temporal = [1, 1]
    return cuboid_hw, shift_hw, strategy, n_temporal

def cross_KxK_heter(mem_shape, K):
    """

    Parameters
    ----------
    mem_shape
    K

    Returns
    -------
    cuboid_hw
    shift_hw
    strategy
    n_temporal
    """
    T_mem, H, W, _ = mem_shape
    K = min(K, H, W)
    cuboid_hw = [(K, K), (K, K), (K, K)]
    shift_hw = [(0, 0), (0, 0), (K // 2, K // 2)]
    strategy = [('l', 'l', 'l'), ('d', 'd', 'd'), ('l', 'l', 'l')]
    n_temporal = [1, 1, 1]
    return cuboid_hw, shift_hw, strategy, n_temporal

# # Our proposed CuboidCrossAttention patterns.
for k in [1, 2, 4, 8]:
    CuboidCrossAttentionPatterns.register(f'cross_{k}x{k}', functools.partial(cross_KxK, K=k))
    CuboidCrossAttentionPatterns.register(f'cross_{k}x{k}_lg', functools.partial(cross_KxK_lg, K=k))
    CuboidCrossAttentionPatterns.register(f'cross_{k}x{k}_heter', functools.partial(cross_KxK_heter, K=k))
