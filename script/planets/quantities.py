
# Defines some common physical quantities.


import math
import const.cgs as cgs
from tools.quantity import Quantity


# integral number
number = Quantity('', name='number', dtype=int)

# real number
dimensionless = Quantity('', name='dimensionless', dtype=float)

# percentage
percentage = Quantity('', name='percentage', dtype=float, units={
    '%': 1.e-2
})

# cgs length
cm = Quantity('cm', name='length', units={
    'nm': 1.e-7,
    'µm': 1.e-4,
    'μm': 1.e-4,
    'um': 1.e-4,
    'mm': 1.e-1,
    'm': 1.e+2,
    'km': 1.e+5,
    'AU': cgs.AU
})

# cgs weight
g = Quantity('g', name='weight', units={
    'kg': 1.e+3,
    'MSun': cgs.MS,
    'M⨀': cgs.MS,
    'MEarth': cgs.Mea,
    'M♁': cgs.Mea,
    'M⨁': cgs.Mea,
    'M⊕': cgs.Mea
})

# cgs time
s = Quantity('s', name='time', units={
    'years': cgs.year,
    'yr': cgs.year,
    'year': cgs.year,
    'kyr': 1.e+3*cgs.year,
    'Myr': 1.e+6*cgs.year,
    'Gyr': 1.e+9*cgs.year
})

# cgs frequency
Hz = Quantity('s⁻¹', name='frequency', units={
    'Hz': 1.,
    'years⁻¹': 1/cgs.year,
    'yr⁻¹': 1/cgs.year,
    'year⁻¹': 1/cgs.year,
    'kyr⁻¹': 1/(1.e+3*cgs.year),
    'Myr⁻¹': 1/(1.e+6*cgs.year),
    'Gyr⁻¹': 1/(1.e+9*cgs.year),
    '/s': 1.,
    '/yr': 1/cgs.year,
    '/year': 1/cgs.year,
    '/kyr': 1/(1.e+3*cgs.year),
    '/Myr': 1/(1.e+6*cgs.year),
    '/Gyr': 1/(1.e+9*cgs.year)
})

# cgs velocity
cm_s = Quantity('cm/s', name='velocity', units={
    'm/s': 100.,
    'km/s': 1.e+5
})

# cgs volume density
g_cm3 = Quantity('g/cm³', name='volume density')

# cgs surface density
g_cm2 = Quantity('g/cm²', name='surface density')

# angle
rad = Quantity('', name='angle', units={
    'deg': math.pi/180,
    '°': math.pi/180
})

# temperature
K = Quantity('K', name='temperature')

# luminosity
erg_s = Quantity('erg/s', name='power', units={
    'μW': 1.e+1,
    'µW': 1.e+1,
    'mW': 1.e+4,
    'W': 1.e+7,
    'kW': 1.e+10,
    'MW': 1.e+13,
    'GW': 1.e+16,
    'TW': 1.e+19,
    'LSun': cgs.LS,
    'L⨀': cgs.LS,
})
