
# Defines the classes `Parameter` and `ParameterSet` to represent parameters.
# Implements serialization, deserialization, reporting, and command-line argument parsing with `argparse`.


import re
import sys
import types
import argparse
import functools
from typing import Any, Union, List, Dict, Optional, TypeVar, Tuple, Iterable, Callable

import tools.quantity
import tools.configuration as _cfg


NS = TypeVar('NS')  # namespace


# taken from https://stackoverflow.com/a/31174427 and extended
def rhasattr(obj, attr):
    for subattr in attr.split('.'):
        if not hasattr(obj, subattr):
            return False
        obj = getattr(obj, subattr)
    return True
def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))
def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)
def rgetattr_force(obj, attr, attr_factory=lambda _obj,_attr: types.SimpleNamespace()):
    def _getattr_force(obj, attr):
        if hasattr(obj, attr):
            return getattr(obj, attr)
        else:
            newattr = attr_factory(obj, attr)
            setattr(obj, attr, newattr)
            return newattr
    return functools.reduce(_getattr_force, [obj] + attr.split('.'))
def rsetattr_force(obj, attr, val, attr_factory=lambda _obj,_attr: types.SimpleNamespace()):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr_force(obj, pre, attr_factory) if pre else obj, post, val)


_none_obj = object()

class ParameterNamespaceBase(types.SimpleNamespace):
    pass

def _param_property(name, param):
    member = '_' + name
    def getter(self):
        x = getattr(self, member, _none_obj)
        if x is _none_obj:
            x = param.default
        return param.unwrap(x)
    def setter(self, val):
        if val is not None:
            if not isinstance(val, param._qdtype):
                val = param._qdtype(val)
        setattr(self, member, val)
    def deleter(self):
        delattr(self, member)
    return property(getter, setter, deleter, param.caption)

def _scope_property(name, ns):
    member = '_' + name
    def getter(self):
        x = getattr(self, member, _none_obj)
        if x is _none_obj:
            x = ns()
            setattr(self, member, x)
        return x
    return property(getter)

def _make_namespace(parameter_set, path):
    class ParameterNamespace(ParameterNamespaceBase):
        params = parameter_set
        name = path
    return ParameterNamespace

def _init_nsdict(parameter_set):
    return { '': _make_namespace(parameter_set, '') }

def _get_namespace(nsdict, parameter_set, path):
    ns = nsdict.get(path, _none_obj)
    if ns is _none_obj:
        ns = _make_namespace(parameter_set, path)
        pre, _, post = path.rpartition('.')
        pre_str = pre if pre else ''
        parent = _get_namespace(nsdict, parameter_set, pre_str)
        setattr(parent, post, _scope_property(post, ns))
        nsdict[path] = ns
    return ns




class Parameter:
    """Represents a parameter."""

    def __init__(self,
            name: Union[str, List[str], None] = None,
            short: Union[str, List[str], None] = None,
            quantity: Optional[tools.quantity.Quantity] = None,
            unit: Optional[str] = None,
            range: bool = False,
            default: Union[str, float, None] = None,
            dtype: Optional[type] = None,
            flags: List[str] = [],
            caption: str = '',
            description: str = '') -> None:
        """Constructs a Parameter object from the given parameter metadata.

        Keyword arguments:  
        name -- full name of the parameter, or list of parameter names (default None)  
        short -- abbreviated name of the parameter, or list of abbreviated parameter names (default None)  
        quantity -- the parameter quantity (default None)  
        unit -- the preferred unit to express the parameter in (defaults to the base unit of the quantity)  
        range -- whether the parameter specifies a range of values (default False)  
        default -- the default value of the parameter (default None)  
        dtype -- the parameter datatype (defaults to quantity.dtype if a quantity is specified, or to type(default) otherwise)  
        flags -- list of parameter flag aliases for this parameter (default [])  
        caption -- caption with brief explanation of parameter (default '')
        description -- extended parameter description (default '')

        At least one name or abbreviated name must be provided.  
        Names and short names are added to the list of parameter flags with either '--' or '-' prepended.
        If no quantity is specified, either the datatype or a default value must be given.  
        If a quantity is specified, the default value must be given as a string containing value and associated unit.
        Otherwise, the default value must match the parameter datatype.  
        """
        assert name is not None or short is not None
        name_list = []
        if short is not None:
            if type(short) is not str:
                # assume list of names
                flags = ['-' + s for s in short] + flags
                name_list += short
                short = short[0]
            else:
                flags = ['-' + short] + flags
                name_list.append(short)
        if name is not None:
            if type(name) is not str:
                # assume list of names
                flags = ['--' + n for n in name] + flags
                name_list += name
                name = name[0]
            else:
                flags = ['--' + name] + flags
                name_list.append(name)
        else:
            name = short
        self.name = name
        self.short = short if short is not None else ''
        self.member_name = name.replace('-', '_').replace('/', '_')
        argtype = dtype
        if quantity is not None:
            assert unit is None or unit in quantity.units
            assert dtype is None
            dtype = quantity.dtype
            argtype = str
            if unit is None:
                unit = quantity.base_unit
        else:
            assert unit is None and not range
        self.unit = unit
        self.range = range
        if dtype is None and default is not None:
            dtype = type(default)
        assert dtype is not None
        if argtype is None:
            argtype = dtype
        if argtype is int:
            argtype = float  # to permit scientific notation
        self.dtype = dtype
        self.argtype = argtype
        self.quantity = quantity
        if quantity is not None:
            if range:
                self._qdtype = tools.quantity.QuantityRange(quantity)
            else:
                self._qdtype = tools.quantity.QuantityValue(quantity)
        else:
            self._qdtype = dtype
        if isinstance(default, str):
            default_str = default
            default = self._qdtype(default_str)
        elif default is not None:
            default = self._qdtype(default)
            default_str = str(default)
        else:
            default_str = None
        self.default_str = default_str
        self.default = default
        self.flags = flags
        self.caption = caption
        self.description = description

    #def init(self, args: NS):
    #    """Sets the default value for this parameter in the given namespace if no value is set."""
    #    if not rhasattr(args, self.member_name):
    #        rsetattr_force(args, self.member_name, self.default)

    def get(self, args: NS):
        """Retrieves a parameter value from a namespace."""
        pre, _, post = self.member_name.rpartition('.')
        if pre and not rhasattr(args, pre):
            return None
        parent = rgetattr(args, pre) if pre else args
        if isinstance(parent, ParameterNamespaceBase):
            member = '_' + post
            return getattr(parent, member, None)
        else:
            return getattr(parent, post, None)

    def unwrap(self, arg):
        if self.quantity is not None:
            if isinstance(arg, self._qdtype):
                if self.range:
                    return arg.values
                else:
                    return arg.value
        return arg

    #def put(self, args: NS, value) -> None:
    #    """Sets a parameter value in a namespace."""
    #    rsetattr_force(args, self.member_name, value)

    def serialize_value(self, value, fmt: Dict[type, Tuple[str, str]] = tools.quantity.default_format_dict, short: bool = False) -> str:
        """Serializes a parameter value as a string (e.g. '1.1 AU').
        
        Keyword arguments:  
        value -- the parameter value  
        fmt -- dictionary mapping value types to format strings
        short -- whether to use short formatting (no separating whitespace)
        """
        if value is None:
            value = self.default
        if value is None:
            return 'None'
        elif self.quantity is not None:
            if isinstance(value, self._qdtype):
                return value.format(in_unit=self.unit, fmt=fmt, short=short)
            else:
                value = self._qdtype(value)
                return value.format(in_unit=self.unit, fmt=fmt, short=short)
        else:
            if self.dtype in fmt:
                fmt_str = fmt[self.dtype][1 if short else 0]
                return fmt_str.format(value)
            else:
                return str(value)

    def deserialize_value(self, value_str: str):
        """Deserializes a parameter value from a string (e.g. '1.1 AU').

        Keyword arguments:  
        value_str -- the parameter value string
        """
        if self.dtype != str and value_str.strip() == 'None':
            return None
        return self._qdtype(value_str)


class FilterPredicates:

    @staticmethod
    def all(param: Parameter, value) -> bool:
        return True

    @staticmethod
    def not_None(param: Parameter, value) -> bool:
        return value is not None

    @staticmethod
    def not_default(param: Parameter, value) -> bool:
        return value != param.default


class ParameterSet:
    """Represents a set of parameters."""

    def __init__(self, parameters: List[Parameter], name: str, short_name: str = '') -> None:
        """Constructs a set of parameters.

        Keyword arguments:  
        parameters -- list of parameters
        name -- name of parameter set  
        short_name -- short name of parameter set  
        """
        assert not (name == '' and short_name != '')

        self.name = name
        self.short_name = short_name
        self._params = list(parameters)
        self._param_dict = dict((param.name, param) for param in self._params)
        self._namespace_dict = _init_nsdict(self)
        for param in self._params:
            if param.short != '':
                if param.short in self._param_dict:
                    raise RuntimeError("property '{}': short name '{}' already refers to property '{}'".format(param.name, param.short, self._param_dict[param.short].name))
                self._param_dict[param.short] = param
            pre, _, post = param.member_name.rpartition('.')
            scope = pre if pre else ''
            ns = _get_namespace(self._namespace_dict, self, scope)
            setattr(ns, post, _param_property(post, param))

    def __iter__(self):
        return iter(self._params)

    def __len__(self):
        return len(self._params)

    def __contains__(self, name: str) -> bool:
        return name in self._param_dict

    def __getitem__(self, name: str) -> Parameter:
        return self._param_dict[name]

    #def init_defaults(self, args: NS) -> NS:
    #    """Initialize missing parameter values to their defaults in the given namespace."""
    #    assert args is not None
    #    for param in self._params:
    #        param.init(args)
    #    return args

    def make_namespace(self):
        ns = self._namespace_dict['']
        return ns()

    def find(self, name: str) -> Parameter:
        """Looks up a parameter by name."""
        return self._param_dict[name]

    @staticmethod
    def _pure_name(name):
        if len(name) > 0 and name[-1] == '!':
            return name[:-len('!')]
        return name

    def _enumerate(self, args, filter_pred: Callable[[Parameter, Any], bool] = FilterPredicates.all, subset: Optional[Iterable[str]] = None):
        if subset is not None:
            for decorated_name in subset:
                name = ParameterSet._pure_name(decorated_name)
                force = len(decorated_name) > 0 and decorated_name[-1] == '!'
                if name not in self._param_dict:
                    raise RuntimeError("name '{}' in given parameter subset is not a name of a parameter".format(name))
                param = self._param_dict[name]
                value = param.get(args)
                if force or filter_pred(param, value):
                    yield param, value
        else:
            for param in self._params:
                value = param.get(args)
                if filter_pred(param, value):
                    yield param, value

    def from_configuration(self, config: _cfg.Configuration, args: Optional[NS] = None) -> NS:
        if args is None:
            args = self.make_namespace()
        as_name = self.name in config
        as_short_name = self.short_name in config
        if as_name or as_short_name:
            section = config[self.name] if as_name else config[self.short_name]
            for entry in section:
                if entry.name != '':
                    if not entry.name in self._param_dict:
                        raise RuntimeError("error while reading configuration: parameter '{}' is not part of parameter set '{}'".format(entry.name, self.name))
                    param = self._param_dict[entry.name]
                    value = param.deserialize_value(entry.value)
                    rsetattr_force(args, entry.name, value)
        return args

    def to_configuration(self, args: NS, config: _cfg.Configuration = None,
            filter_pred: Callable[[Parameter, Any], bool] = FilterPredicates.all, subset: Optional[Iterable[str]] = None, with_captions: bool = False):
        section = None
        if config is not None:
            result = config
            if self.name in config:
                section = config[self.name]
            elif self.short_name in config:
                section = config[self.short_name]
        else:
            result = _cfg.Configuration()
        if section is None:
            section = _cfg.ConfigurationSection(name=self.name, short_name=self.short_name)
            result.add_section(section)
        for param, value in self._enumerate(args=args, filter_pred=filter_pred, subset=subset):
            entry = _cfg.ConfigurationEntry(
                name=param.name, short_name=param.short,
                value=param.serialize_value(value, short=False), short_value=param.serialize_value(value, short=True),
                comment=param.caption if with_captions else '')
            if config is not None:
                section.override_entry(entry)
            else:
                section.add_entry(entry)
        return result

    def copy(self, src, dest: Optional[NS] = None, filter_pred: Callable[[Parameter, Any], bool] = FilterPredicates.all, subset: Optional[List[str]] = None):
        """Copies parameter values from namespace `src` to namespace `dest`.

        Keyword arguments:  
        src -- namespace with parameter values  
        dest -- pre-existing namespace to be filled with argument values (optional)  
        filter_pred -- predicate to determine whether to copy a given value for a given parameter  
        subset -- optional subset of parameters to serialize (by default, all parameters are serialized)
        """

        if dest is None:
            dest = self.make_namespace()
        for param, value in self._enumerate(args=src, filter_pred=filter_pred, subset=subset):
            rsetattr_force(dest, param.member_name, value)
        return dest

    @staticmethod
    def _metavar(param):
        ldot = param.member_name.rfind('.')
        varname = param.member_name[ldot + 1:] if ldot >= 0 else param.member_name
        if param.range:
            return '[<{}>,...]'.format(varname)
        else:
            return '<{}>'.format(varname)

    class _StoreAction(argparse.Action):
        def __init__(self, *args, dest, quantity_param, dtype=None, **kwargs):
            super(ParameterSet._StoreAction, self).__init__(*args, dest=argparse.SUPPRESS, metavar=ParameterSet._metavar(quantity_param), type=dtype, **kwargs)
            self.rdest = dest
        def __call__(self, parser, namespace, values, option_string=None):
            rsetattr_force(namespace, self.rdest, values)

    class _StoreQuantityAction(argparse.Action):
        def __init__(self, *args, dest, quantity_param, dtype=None, **kwargs):
            super(ParameterSet._StoreQuantityAction, self).__init__(*args, dest=argparse.SUPPRESS, metavar=ParameterSet._metavar(quantity_param), type=dtype, **kwargs)
            self.rdest = dest
            self.quantity_param = quantity_param
        def __call__(self, parser, namespace, values, option_string=None):
            numericValue = self.quantity_param.deserialize_value(values)
            rsetattr_force(namespace, self.rdest, numericValue)

    def add_to_argparser(self, argparser: argparse.ArgumentParser) -> None:
        """Add arguments for all parameters to the given argument parser."""
        for param in self._params:
            actionArgs = { 'quantity_param': param }
            if param.quantity is not None:
                action = ParameterSet._StoreQuantityAction
            else:
                action = ParameterSet._StoreAction
            help = param.caption
            description_str = '{}; '.format(param.description) if param.description != '' else ''
            if param.quantity is None and param.unit is not None and param.unit != '':
                help += ' ({}{}'.format(description_str, param.unit)
                if param.default is not None:
                    help += '; default: {} {}'.format(param.default, param.unit).replace('%', '%%')
                    #help += '; default: %(default)'
                help += ')'
            elif param.default is not None and param.default_str != '':
                help += ' ({}default: {})'.format(description_str, param.default_str.replace('%', '%%'))  # we could report the default value in the given unit for quantities, but let's not bother
            elif param.description != '':
                help += ' ({})'.format(param.description)
            argparser.add_argument(
                *param.flags, default=None, dtype=param.argtype, help=help, dest=param.member_name,
                action=action, **actionArgs)
