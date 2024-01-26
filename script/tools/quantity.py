
# Defines the class `Quantity` to represent quantities with a unit of measure.


import numpy as np
import re
import functools
from typing import Dict, Optional, TypeVar, Tuple
from collections.abc import Iterable

import numpy as np


T = TypeVar('T')


def _value_parser(value_pattern, unit_pattern):
    regex_str = r'^\s*((?P<null>0)|(?P<inf>inf|∞)|((?P<val>{})\s*(?P<unit>{})))\s*$'.format(value_pattern, unit_pattern)
    return re.compile(regex_str)

default_format_dict = {
    #      (default, short)
    float: ('{:g}', '{:.4g}'),
    int:   ('{}', '{}'),
    str:   ('{}', '{}')
}

class Quantity:
    """Represents a quantity with a unit of measure."""

    def __init__(
            self,
            base_unit: str,
            name: str,
            dtype: type = float,
            units: Dict[str, float] = None) -> None:
        """Constructs a quantity with a unit of measure.

        Keyword arguments:  
        base_unit -- the unit a number representing this quantity is expressed in  
        dtype -- the datatype used to store this quantity (defaults to float)  
        name -- the name of the quantity (e.g. 'length', 'time')  
        units -- dictionary of unit names mapping to factors for conversion to the base unit (optional)  

        The base unit may be an empty string.
        """
        self.base_unit = base_unit
        self.dtype = dtype
        self.name = name
        if units is None:
            units = {}
        units[base_unit] = dtype(1)
        self.units = units

        value_pattern = '.*?'  # non-greedy match-all pattern (refine if necessary)
        unit_pattern = '|'.join([re.escape(u) for u in units.keys()])
        self.value_parser = _value_parser(value_pattern, unit_pattern)

    def in_unit(self, qvalue: T, unit: str) -> T:
        """Converts a value from the base unit of the quantity to the given unit.
        
        Keyword arguments:  
        qvalue -- value expressed in the base unit of this quantity  
        unit -- unit of the return value
        """
        return self.dtype(qvalue/self.units[unit])

    def format(self, qvalue: T, in_unit: Optional[str] = None, fmt: Dict[type, Tuple[str, str]] = default_format_dict, short: bool = False) -> str:
        """Returns a quantity with an associated unit.
        
        Keyword arguments:  
        qvalue -- the value expressed in the base unit of this quantity  
        in_unit -- the unit the quantity should be expressed in (defaults to the base unit of the quantity)  
        fmt -- dictionary mapping value types to format strings  
        short -- whether to use short formatting (no separating whitespace)
        """
        if in_unit is None:
            in_unit = self.base_unit
        fmt_str = fmt[self.dtype][1 if short else 0]
        # Certain special values are formatted without units.
        if self.dtype == float and np.isposinf(qvalue):
            return 'inf'
        if self.dtype in [int, float] and qvalue == 0:
            return '0'
        qvalue_str = fmt_str.format(self.in_unit(qvalue=qvalue, unit=in_unit))
        sep = '' if short or in_unit == '' else ' '
        return '{}{}{}'.format(qvalue_str, sep, in_unit)

    def from_unit(self, value: T, unit: str) -> T:
        """Converts a value from the given unit to the base unit of the quantity."""
        return value*self.units[unit]
        
    def parse(self, value_str: str):
        """Parses a value of this quantity from the given string."""
        match = self.value_parser.match(value_str)
        if match is None:
            raise RuntimeError('cannot parse quantity \'{}\': unrecognized syntax or unknown unit in \'{}\''.format(self.name, value_str))
        groups = match.groups()
        null_str = match.group('null')
        inf_str = match.group('inf')
        val_str = match.group('val')
        unit_str = match.group('unit')
        if inf_str is not None and self.dtype in [int, float]:
            if self.dtype == int:
                raise RuntimeError('cannot parse quantity \'{}\': integral quantities cannot represent infinity (\'{}\' given)'.format(self.name, value_str))
            return self.dtype(np.inf)
        elif null_str is not None and self.dtype in [int, float]:
            return self.dtype(0)
        elif self.dtype in [int, float] and '/' in val_str:
            if self.dtype == int:
                raise RuntimeError('cannot parse quantity \'{}\': fractions not allowed for integral arguments (\'{}\' given)'.format(self.name, value_str))
            ndstrs = val_str.split('/')
            if len(ndstrs) != 2:
                raise RuntimeError('cannot parse quantity \'{}\': \'{}\' is not a valid fraction'.format(self.name, val_str))
            num = float(ndstrs[0])
            den = float(ndstrs[1])
            value = num/den
        elif self.dtype in [int, float] and val_str == '' and unit_str != '':
            value = self.dtype(1)
        elif self.dtype == int:
            # to permit scientific notation
            fval = float(val_str)
            if fval != round(fval):
                raise RuntimeError('cannot parse quantity \'{}\': expected integral value but got \'{}\''.format(self.name, val_str))
            value = int(fval)
        else:
            value = self.dtype(val_str)
        return self.from_unit(value, unit_str)


@functools.cache
def QuantityValue(quantity, unit = None):
    """Represents a value."""

    assert quantity is not None
    
    class _QuantityValue:
        """Represents a value."""

        def _init(self, value) -> None:
            self.quantity = quantity
            self.value = quantity.dtype(value)

        def _deserialize(self, s: str):
            self.quantity = quantity
            self.value = quantity.parse(s)

        def format(self, in_unit: Optional[str] = unit, fmt: Dict[type, Tuple[str, str]] = default_format_dict, short: bool = False) -> str:
            """Returns a quantity with an associated unit.
            
            Keyword arguments:  
            in_unit -- the unit the quantity should be expressed in (defaults to the base unit of the quantity)  
            fmt -- dictionary mapping value types to format strings  
            short -- whether to use short formatting (no separating whitespace)
            """

            return self.quantity.format(self.value, in_unit=in_unit, fmt=fmt, short=short)

        def __str__(self):
            return self.format()

        def __init__(self, *args, **kwargs):
            if len(args) > 0:
                if len(args) == 1 and len(kwargs) == 0:
                    arg = args[0]
                    if isinstance(arg, str):
                        self._deserialize(arg)
                    else:
                        self._init(value=arg)
                else:
                    self._init(*args, **kwargs)
            elif len(kwargs) > 0:
                if 'str' in kwargs:
                    assert len(kwargs) == 1
                    self._deserialize(kwargs['str'])
                else:
                    self._init(**kwargs)

        def __eq__(self, other):
            if isinstance(other, _QuantityValue):
                return self.quantity == other.quantity \
                   and np.array_equal(self.value, other.value, equal_nan=True)
            else:
                return False

    return _QuantityValue


_range_parser = re.compile(r'^(.*?)\s*\.\.\s*(.*?)\s*\((\d+?)\s*(?:steps\s*|)(?:,\s*(linear|log10|log2|log)|)\s*\)\s*$')  # vmin, vmax, num[, scale]
_list_parser = re.compile(r'^\s*\[(.*)\](?:\s*\(\s*(linear|log10|log2|log)\s*\)|)\s*$')  # values[, scale]

@functools.cache
def QuantityRange(quantity, unit = None):
    """Represents a range of values."""

    assert quantity is not None
    assert quantity.dtype in [int, float]

    class _QuantityRange:
        """Represents a range of values."""

        def _init(
                self,
                vmin = None,
                vmax = None,
                num = None,
                scale = None,
                value = None,
                values = None) -> None:
            """Constructs a `QuantityRange`.

            Keyword arguments:  
            vmin, vmax -- the smallest and largest value in the range  
            num -- total number of values  
            scale -- scale for choosing intermediate values (linear, log; default: linear)  
            value -- explicit scalar value
            values -- explicit list of values
            """
            if vmin is not None and vmax is not None and scale is None:
                scale = 'linear'
            self.quantity = quantity
            self.vmin = vmin
            self.vmax = vmax
            self.num = num
            self.scale = scale
            self.values = values
            if vmin is not None or vmax is not None or scale is not None or num is not None:
                assert vmin is not None and num is not None
                assert values is None and value is None
                if num == 1:
                    values = np.array([float(vmin)])
                else:
                    assert vmax is not None
                    if scale == 'linear':
                        values = np.linspace(start=vmin, stop=vmax, num=num)
                    elif scale == 'log' or scale == 'log10':
                        values = np.logspace(start=np.log10(vmin), stop=np.log10(vmax), num=num)
                    elif scale == 'log2':
                        values = np.logspace(start=np.log2(vmin), stop=np.log2(vmax), num=num, base=2.)
                    else:
                        assert False
                if quantity.dtype == int:
                    values = values.round().astype(int)
                self.values = values
            elif values is not None:
                assert vmin is None and vmax is None and num is None
                assert value is None
                assert isinstance(values, Iterable)
                self.num = len(values)
            else:
                assert vmin is None and vmax is None and num is None
                self.values = value

        def _deserialize(self, s: str):
            match = _list_parser.match(s)
            if match is not None:
                groups = match.groups()
                value_strs = groups[0].split(',')
                scale = groups[1] if len(groups) > 1 else None
                if scale is not None and scale not in ['linear', 'log', 'log10', 'log2']:
                    raise RuntimeError('quantity \'{}\': error while parsing list \'{}\': unknown scale \'{}\' (expected \'linear\', \'log\', \'log10\', or \'log2\')'.format(quantity.name, s, scale))
                values = np.array([quantity.parse(value_str.strip()) for value_str in value_strs], dtype=quantity.dtype)
                self._init(values=values, scale=scale)
            if '..' in s:
                match = _range_parser.match(s)
                if match is None:
                    raise RuntimeError('quantity \'{}\': error while parsing range \'{}\': syntax error (missing quantity?)'.format(quantity.name, s))
                groups = match.groups()
                vmin_str, vmax_str, num_str = groups[0], groups[1], groups[2]
                if len(groups) == 4:
                    scale = groups[3]
                    if scale not in ['linear', 'log', 'log10', 'log2']:
                        raise RuntimeError('quantity \'{}\': error while parsing range \'{}\': unknown scale \'{}\' (expected \'linear\', \'log\', \'log10\', or \'log2\')'.format(quantity.name, s, scale))
                else:
                    scale = 'linear'
                vmin = quantity.parse(vmin_str.strip())
                vmax = quantity.parse(vmax_str.strip())
                num = int(num_str)
                self._init(vmin=vmin, vmax=vmax, num=num, scale=scale)
            else:
                value = quantity.parse(s)
                self._init(vmin=value, num=1)

        def format(self, in_unit: Optional[str] = unit, fmt: Dict[type, Tuple[str, str]] = default_format_dict, short: bool = False) -> str:
            """Formats a quantity range.
            
            Keyword arguments:  
            in_unit -- the unit the quantity should be expressed in (defaults to the base unit of the quantity)  
            fmt -- dictionary mapping value types to format strings  
            short -- whether to use short formatting (no separating whitespace)
            """
            if self.vmin is not None:
                if self.vmax is None:
                    return self.quantity.format(self.vmin, in_unit=in_unit, fmt=fmt, short=short)
                else:
                    fmt_str = '{}..{}({},{})' if short else '{}..{} ({} steps, {})'
                    return fmt_str.format(
                        self.quantity.format(self.vmin, in_unit=in_unit, fmt=fmt, short=short),
                        self.quantity.format(self.vmax, in_unit=in_unit, fmt=fmt, short=short),
                        self.num,
                        self.scale)
            else:
                if isinstance(self.values, Iterable):
                    sep = '' if short else ' '
                    scale_str = sep + '(' + self.scale + ')' if self.scale is not None else ''
                    return '[' + (',' + sep).join(self.quantity.format(v, in_unit=in_unit, fmt=fmt, short=short) for v in self.values) + ']' + scale_str
                else:
                    return self.quantity.format(self.values, in_unit=in_unit, fmt=fmt, short=short)

        def __str__(self):
            return self.format()

        def __init__(self, *args, **kwargs):
            if len(args) > 0:
                if len(args) == 1 and len(kwargs) == 0:
                    arg = args[0]
                    if isinstance(arg, str):
                        self._deserialize(arg)
                    elif isinstance(arg, range):
                        self._init(values=np.array(arg))
                    elif isinstance(arg, Iterable):
                        self._init(values=arg)
                    else:
                        self._init(value=arg)
                else:
                    self._init(*args, **kwargs)
            elif len(kwargs) > 0:
                if 'str' in kwargs:
                    assert len(kwargs) == 1
                    self._deserialize(kwargs['str'])
                else:
                    self._init(**kwargs)

        def __eq__(self, other):
            if isinstance(other, _QuantityRange):
                return self.vmin == other.vmin \
                   and self.vmax == other.vmax \
                   and self.scale == other.scale \
                   and self.num == other.num \
                   and np.array_equal(self.values, other.values, equal_nan=True)
            else:
                return False

    return _QuantityRange
