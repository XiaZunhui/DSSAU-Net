__author__ = 'Alex Rogozhnikov'
__version__ = '0.6.1'


class EinopsError(RuntimeError):
    """ Runtime error thrown by meinops """
    pass


__all__ = ['rearrange', 'reduce', 'repeat', 'einsum',
           'pack', 'unpack',
           'parse_shape', 'asnumpy', 'EinopsError']

from .einops import rearrange, reduce, repeat, einsum, parse_shape, asnumpy
from .packing import pack, unpack