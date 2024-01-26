    
# Defines some complex serializable types.


import re
import functools

from planets import quantities


#_float_regex = r'[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?'
_logbucketdensity_parser = re.compile(r'^\s*(?P<lo>[^<>=≥≤≠@~]+?)\s*(|\.\.(?P<hi>[^<>=≥≤≠@~]+?)\s*)(|\(\s*(|(>=|≥)\s*(?P<min>[^<>=≥≤≠@~]+?)\s*)(|@\s*(?P<mean>[^<>=≥≤≠@~]+?)\s*)\)\s*)(|~\s*(?P<width>[^<>=≥≤≠@~]+?)\s*)$')


@functools.cache
def LogBucketDensity(quantity, unit = None):
    assert quantity is not None

    xmindef = quantity.dtype(0)
    x0def = quantity.dtype(1)
    class _LogBucketDensity:
        def _init(self, bmin, bmax=None, xmin=None, x0=None, dldx=0):
            self.quantity = quantity
            self.bmin = bmin
            if bmax is None:
                bmax = bmin
            assert bmax != bmin or dldx == 0
            self.bmax = bmax
            if xmin is None:
                xmin = xmindef
            self.xmin = xmin
            if x0 is None:
                x0 = x0def
            self.x0 = x0
            self.dldx = dldx

        def _deserialize(self, s):
            self.quantity = quantity
            match = _logbucketdensity_parser.match(s)
            if match is not None:
                bminstr = match.group('lo')
                bmaxstr = match.group('hi')
                xminstr = match.group('min')
                x0str = match.group('mean')
                dldxstr = match.group('width')
                bmin = float(bminstr)
                bmax = float(bmaxstr) if bmaxstr is not None else None
                xmin = quantity.parse(xminstr) if xminstr is not None else None
                x0 = quantity.parse(x0str) if x0str is not None else None
                dldx = float(dldxstr) if dldxstr is not None else 0.
                self._init(bmin=bmin, bmax=bmax, xmin=xmin, x0=x0, dldx=dldx)
            else:
                self._init(float(s))

        def __str__(self):
            result = '{:g}'.format(self.bmin)
            if self.bmax != self.bmin:
                result = '{}..{:g}'.format(result, self.bmax)
            sub = ''
            if self.xmin > xmindef:
                sub = '≥ {}'.format(quantity.format(self.xmin, in_unit=unit))
            if self.x0 != x0def:  # TODO: maybe do unconditionally?
                if sub != '':
                    sub = sub + ' '
                sub = '{}{}@ {}'.format(sub, ' ' if sub != '' else '', quantity.format(self.x0, in_unit=unit))
            if self.xmin > xmindef or self.x0 != x0def:
                result = '{} ({})'.format(result, sub)
            if self.dldx != 0:
                result = '{} ~ {:g}'.format(result, self.dldx)
            return result

        def __init__(self, *args, **kwargs):
            if len(args) > 0:
                if isinstance(args[0], str):
                    assert len(args) == 1
                    assert len(kwargs) == 0
                    self._deserialize(args[0])
                else:
                    self._init(*args, **kwargs)
            elif len(kwargs) > 0:
                if 'str' in kwargs:
                    assert len(kwargs) == 1
                    self._deserialize(kwargs['str'])
                else:
                    self._init(**kwargs)

    return _LogBucketDensity
