
# Entry point of simulation.


import os
import sys
import time
import math
from typing import Iterable, Dict, Any, Optional, Sequence, Union

import numpy as np
import pandas as pd

from tools import data as _data
from tools import configuration as _cfg
from tools.parameters import Parameter, ParameterSet
from tools.quantity import QuantityRange
import tools.projection as projection
from planets import quantities, implicit
import planets.types


π = np.pi


# Define simulation parameters.
params = ParameterSet(name='Simulation', short_name='sim', parameters=[
    # system properties
    Parameter(name='star.M',                            default='MSun',     quantity=quantities.g,          unit='MSun',
          caption='mass of central star'),
    Parameter(name='star.L',          short='L',        default='LSun',     quantity=quantities.erg_s,      unit='LSun',
          caption='luminosity of central star'),

    # planet properties
    Parameter(name='planet.ρ', short='ρP',              default=None,       quantity=quantities.g_cm3,                     range=True,
          caption='bulk density of planet'),
    Parameter(name='planet.m', short='mP',              default=None,       quantity=quantities.g,          unit='MEarth', range=True,
          caption='mass of planet'),
    Parameter(name='planet.R', short='RP',              default=None,       quantity=quantities.cm,         unit='km',     range=True,
          caption='bulk radius of planet'),
    Parameter(name='planet.a', short='aP',              default=None,       quantity=quantities.cm,         unit='AU',     range=True,
          caption='semimajor axis of planet'),
    Parameter(name='planet.i', short='incP',            default=None,       quantity=quantities.rad,        unit='°',      range=True,
          caption='inclination angle of planet'),
    Parameter(name='planet.sini', short='sinincP',      default=None,       quantity=quantities.dimensionless,             range=True,
          caption='inclination of planet'),
    Parameter(name='planet.e', short='eP',              default=None,       quantity=quantities.dimensionless,             range=True,
          caption='eccentricity of planet'),

    # planetesimal properties
    Parameter(name='planetesimal.ρ', short='ρPlt',      default=None,       quantity=quantities.g_cm3,                     range=True,
          caption='bulk density of planetesimals'),
    Parameter(name='planetesimal.m', short='mPlt',      default=None,       quantity=quantities.g,          unit='MEarth', range=True,
          caption='mass of planetesimal'),
    Parameter(name='planetesimal.M', short='MPlt',      default=None,       quantity=quantities.g,          unit='MEarth', range=True,
          caption='total mass of planetesimals in ring'),
    Parameter(name='planetesimal.a', short='aPlt',      default=None,       quantity=quantities.cm,         unit='AU',     range=True,
          caption='semimajor axis of planetesimals'),
    Parameter(name='planetesimal.N', short='NPlt',      default=None,       quantity=quantities.number,                    range=True,
          caption='number of planetesimals'),
    Parameter(name='planetesimal.R', short='RPlt',      default=None,       quantity=quantities.cm,         unit='km',     range=True,
          caption='bulk radius of planetesimals'),
    Parameter(name='planetesimal.i', short='incPlt',    default=None,       quantity=quantities.rad,        unit='°',      range=True,
          caption='inclination dispersion angle arcsin √<sin² i> of planetesimals'),
    Parameter(name='planetesimal.sini', short='sinincPlt', default=None,    quantity=quantities.dimensionless,             range=True,
          caption='inclination dispersion √<sin² i> of planetesimals'),
    Parameter(name='planetesimal.e', short='ePlt',      default=None,       quantity=quantities.dimensionless,             range=True,
          caption='eccentricity dispersion √<e²> of planetesimals'),
    Parameter(name='planetesimal.Δv', short='ΔvPlt',    default=None,       quantity=quantities.cm_s,                       range=True,
          caption='velocity dispersion √<Δv²> of planetesimals'),
    Parameter(name='planetesimal.Δv_vh', short='Δv_vhPlt', default=None,    quantity=quantities.dimensionless,              range=True,
          caption='velocity dispersion √<Δv²> of planetesimals in units of Hill velocity'),

    # embryo properties
    Parameter(name='embryo.ρ', short='ρE',      default=None,       quantity=quantities.g_cm3,                             range=True,
          caption='bulk density of embryos'),
    Parameter(name='embryo.m', short='mE',      default=None,       quantity=quantities.g,          unit='MEarth',         range=True,
          caption='mass of embryo'),
    Parameter(name='embryo.M', short='ME',      default=None,       quantity=quantities.g,          unit='MEarth',         range=True,
          caption='total mass of embryos in ring'),
    Parameter(name='embryo.a', short='aE',      default=None,       quantity=quantities.cm,         unit='AU',             range=True,
          caption='semimajor axis of embryos'),
    Parameter(name='embryo.N', short='NE',      default=None,       quantity=quantities.number,                            range=True,
          caption='number of embryos'),
    Parameter(name='embryo.R', short='RE',      default=None,       quantity=quantities.cm,         unit='km',             range=True,
          caption='bulk radius of embryos'),
    Parameter(name='embryo.i', short='incE',    default=None,       quantity=quantities.rad,        unit='°',              range=True,
          caption='inclination dispersion angle arcsin √<sin² i> of embryos'),
    Parameter(name='embryo.sini', short='sinincE',    default=None, quantity=quantities.dimensionless,                     range=True,
          caption='inclination dispersion √<sin² i> of embryos'),
    Parameter(name='embryo.e', short='eE',      default=None,       quantity=quantities.dimensionless,                     range=True,
          caption='eccentricity dispersion √<e²> of embryos'),
    Parameter(name='embryo.Δv', short='ΔvE',    default=None,       quantity=quantities.cm_s,                              range=True,
          caption='velocity dispersion √<Δv²> of embryos'),
    Parameter(name='embryo.Δv_vh', short='Δv_vhE', default=None,    quantity=quantities.dimensionless,                     range=True,
          caption='velocity dispersion √<Δv²> of embryos in units of Hill velocity'),

    # ring properties
    Parameter(name='ring.r',         short='r',         default=None,       quantity=quantities.cm,         unit='AU',
          caption='central radius of planetesimal ring'),
    Parameter(name='ring.Δr',        short='Δr',        default=None,       quantity=quantities.cm,         unit='AU',
          caption='width of planetesimal ring'),
    Parameter(name='ring.Δr/2',      short='Δr/2',      default=None,       quantity=quantities.cm,         unit='AU',
          caption='half-width of planetesimal ring'),

    # disk properties
    Parameter(name='disk.rMin',      short='rMin',      default=None,       quantity=quantities.cm,         unit='AU',
          caption='inner disk boundary'),
    Parameter(name='disk.rMax',      short='rMax',      default=None,       quantity=quantities.cm,         unit='AU',
          caption='outer disk boundary'),

    # gas properties
    Parameter(name='gas.Cd',         short='Cd',        default=None,       quantity=quantities.dimensionless,
          caption='gas drag coefficient'),
    Parameter(name='gas.α',          short='α',         default=None,       quantity=quantities.dimensionless,
          caption='viscosity parameter'),
    Parameter(name='gas.p',          short='pg',        default='-1.5',     quantity=quantities.dimensionless,
          caption='power-law exponent of radial gas density profile'),
    Parameter(name='gas.r0',         short='r0g',       default=None,       quantity=quantities.cm,         unit='AU',
          caption='reference radius for gas disk'),
    Parameter(name='gas.ρ0',         short='ρ0g',       default=None,       quantity=quantities.g_cm3,      unit='g/cm³',
          caption='midplane gas density at reference radius r₀'),
    Parameter(name='gas.Σ0',         short=['Σ0', 'Σ0g'], default=None,     quantity=quantities.g_cm2,      unit='g/cm²',
          caption='surface density at reference radius r₀'),
    Parameter(name='gas.cs0',         short='cs0',      default=None,       quantity=quantities.cm_s,       unit='cm/s',
          caption='speed of sound at reference radius r₀'),
    Parameter(name='gas.Tmin',        short='Tmin',     default=None,       quantity=quantities.K,          unit='K',
          caption='background temperature'),
    Parameter(name='gas.rmin',       short='rming',     default='0 AU',     quantity=quantities.cm,         unit='AU',
          caption='lower gas disk boundary'),
    Parameter(name='gas.rmax',       short='rmaxg',     default=None,       quantity=quantities.cm,         unit='AU',
          caption='upper gas disk boundary'),
    Parameter(name='gas.M',           short='Mg',       default=None,       quantity=quantities.g,          unit='MSun',
          caption='total mass of gas disk from rₗₒ to rₕᵢ'),
    Parameter(name='gas.T0',          short='T0',       default=None,       quantity=quantities.K,          unit='K',
          caption='mid-plane temperature at reference radius r₀'),
    Parameter(name='gas.Taccr0',      short='Taccr0',   default=None,       quantity=quantities.K,          unit='K',
          caption='mid-plane accretion temperature at reference radius r₀'),
    Parameter(name='gas.Hp0',         short='Hp0',       default=None,       quantity=quantities.cm,        unit='AU',
          caption='pressure scale height at reference radius r₀'),
    Parameter(name='gas.hp0',         short='hp0',       default=None,       quantity=quantities.dimensionless,
          caption='dimensionless pressure scale height Hₚ₀/r₀ at reference radius r₀'),
    Parameter(name='gas.φ',           short='φg',       default=None,        quantity=quantities.rad,       unit='',
          caption='flaring angle'),
    Parameter(name='gas.γ',           short='γ',        default=None,       quantity=quantities.dimensionless,
          caption='strength of turbulent density fluctuations'),  # TODO
    Parameter(name='gas.profile',                       default='none',
          caption='gas profile type',
          description="'none': no gas, " +
                      "'power-law': power-law gas profile, " +
                      "'power-law-with-planetary-gap': power-law gas profile with pressure gap as would be carved by planet of given mass"),
    Parameter(name='gas.rPlanet',     short='rPl',      default=None,       quantity=quantities.cm,         unit='AU',
          caption='orbital radius of gap-carving planet'),
    Parameter(name='gas.rDustTrap',   short='rTrap',    default=None,       quantity=quantities.cm,         unit='AU',
          caption='orbital radius of dust trap for St=1 particles (pebble trap, actually)'),
    Parameter(name='gas.mPlanet',     short='mPl',      default=None,       quantity=quantities.g,          unit='MEarth',
          caption='mass of gap-carving planet'),
    
    # radial zone properties
    Parameter(name='zones.ΔrMin',    short='ΔrMin',     default=None,       quantity=quantities.cm,         unit='AU',
          caption='minimal width of radial zones'),
    
    # effects to simulate
    Parameter(name='simulation.method',                                     dtype=str,
          caption="simulation method",
          description="'rpmc': RPMC simulation" +
                      "'hybrid': hybrid N-body/RPMC simulation"),
    Parameter(name='simulation.effects',                default='all',      dtype=str,
          caption="physical effects to simulate",
          description="'stirring': simulate viscous stirring by swarm particles as per Ormel et al. (2010), " +
                      "'friction': simulate dynamical friction by swarm particles as per Ormel et al. (2010), " +
                      "'collisions': simulate stochastic collisions, " +
                      "'all': all of the above"),
    Parameter(name='collisions.kernel', short='kernel',                     dtype=str,
          caption="collision handling",
          description="'constant': constant test kernel, " +
                      "'constant-threshold': constant test kernel with threshold, " +
                      "'linear': linear test kernel, " +
                      "'linear-threshold': linear test kernel with threshold, " +
                      "'product': product test kernel, " +
                      "'runaway': runaway test kernel, " +
                      "'geometric': simple estimate assuming homogeneous and isotropic motion in bounded volume as per Ormel et al. (2010)"),

    # arguments for collision test kernels
    Parameter(name='collisions.constant-collision-rate',    default=None,   quantity=quantities.dimensionless,
          caption="constant kernel collision rate"),
    Parameter(name='collisions.constant-threshold-mass',    default=None,   quantity=quantities.dimensionless,
          caption="constant kernel threshold mass",
          description="with the 'constant-threshold' kernel, particles interact only if the one has a mass above and the other below the given threshold mass"),
    Parameter(name='collisions.linear-threshold-mass',    default=None,   quantity=quantities.dimensionless,
          caption="linear kernel threshold mass",
          description="with the 'linear -threshold' kernel, particles interact only if the one has a mass above and the other below the given threshold mass"),
    Parameter(name='collisions.linear-collision-rate-coefficient', default=None, quantity=quantities.dimensionless,
          caption="linear kernel collision rate coefficient"),
    Parameter(name='collisions.product-collision-rate-coefficient', default=None, quantity=quantities.dimensionless,
          caption="product kernel collision rate coefficient"),
    Parameter(name='collisions.runaway-collision-rate-coefficient', default=None, quantity=quantities.dimensionless,
          caption="runaway kernel collision rate coefficient"),
    Parameter(name='collisions.runaway-critical-mass',      default=None, quantity=quantities.dimensionless,
          caption="runaway kernel critical mass"),

    # arguments for geometric collision kernel
    Parameter(name='collisions.outcomes',                   default='none',
          caption="'+'-delimited list of possible collision outcomes",
          description="'coagulation': inelastic collisions: coagulation, " +
                      "'fragmentation': inelastic collisions: fragmentation"),
    Parameter(name='collisions.ε',      short='ε',          default=None,   quantity=quantities.dimensionless,
          caption='coefficient of restitution in Ormel et al. (2010) collision model'),
    Parameter(name='collisions.Rfrag',  short='Rfrag',      default=None,   quantity=quantities.cm,        unit='km',
          caption='radius of fragments in Ormel et al. (2010) collision model'),

    # simulation arguments
    Parameter(name='simulation.options',                    default='none',
          caption="configuration options for simulation",
          description="'locality': enable locality optimisation for kernels which support it"),
    Parameter(name='simulation.bucket-exhaustion',          default='none',
          caption="controls particle-independent exhaustion of bucket property bounds",
          description="'none', 'tracer-mass', 'tracer-velocity', 'tracer-position', 'tracer', 'swarm', 'full'"),
    Parameter(name='simulation.N-threshold',           short='NTh',         default='1.5', quantity=quantities.dimensionless,
          caption="swarm particle number threshold for active N-body particles",
          description="only particles with a particle number below the threshold are treated as active N-body particles which " +
                      "exert direct gravitational force onto other particles"),
    Parameter(name='simulation.St-NBody-threshold', short='StNBTh',            default='∞',   quantity=quantities.dimensionless,
          caption="Stokes number threshold for N-body kinetics",
          description="mutual collision, viscous stirring, and dynamical friction is suppressed if both particles are self-representing and have Stokes numbers above the threshold"),
    Parameter(name='simulation.m-NBody-threshold', short='mNBTh',              default='∞',   quantity=quantities.g, unit='MEarth',
          caption="mass threshold for N-body kinetics",
          description="mutual collision, viscous stirring, and dynamical friction is suppressed if both particles are self-representing and have masses above the threshold"),
    Parameter(name='simulation.St-equilibrium-threshold', short='StEqTh',      default='∞',   quantity=quantities.dimensionless,
          caption="Stokes number threshold for equilibrium gas drag",
          description="equilibrium gas drag is suppressed for particles with Stokes numbers above the threshold"),
    Parameter(name='simulation.St-dust-threshold',         short='StDustTh',   default='0',   quantity=quantities.dimensionless,
          caption="Stokes number threshold for dust",
          description="collision is suppressed if both particles are below the threshold, and viscous stirring and dynamical friction " +
                      "are suppressed if at least one particle is below the threshold"),
    Parameter(name='simulation.m-dust-threshold',         short='mDustTh',     default='1 kg',  quantity=quantities.g, unit='g',
          caption="mass threshold for dust",
          description="collision is suppressed if both particles are below the threshold, and viscous stirring and dynamical friction " +
                      "are suppressed if at least one particle is below the threshold"),
    Parameter(name='simulation.relative-change-update-threshold', default='0.01', quantity=quantities.dimensionless,
          caption='maximal admitted fractional change before update',
          description="maximal admitted fractional change  maxᵢ{ Δa/a, Δe/e, Δi/i }  after which the stirring and collision rates " +
                      "should be recomputed"),
    Parameter(name='simulation.particle-regime-threshold', default='100',   quantity=quantities.number,
          caption='regime threshold for particle number',
          description="collisions involving a particle from a swarm with a particle number less than `particle-regime-threshold` " +
                      "are treated in a different regime to properly support oligarchic growth"),
    Parameter(name='simulation.particle-regime-threshold-for-interaction-rates', default='0',   quantity=quantities.dimensionless,
          caption='optional separate threshold for the particle regime when computing collision rates; experimental use only',
          description="A value of  0  indicates that `particle-regime-threshold` is used instead: the only sensible choice"),
    Parameter(name='simulation.bin-widening', short='bin-widening',           default='0',   quantity=quantities.dimensionless,
          caption='percentual widening of buckets to decrease updating rate'),
    Parameter(name='simulation.removal-bucket-update-delay', short='removal-bucket-update-delay',     default='1',   quantity=quantities.dimensionless,
          caption='percentual update delay of buckets to decrease updating rate on particle removal'),
    Parameter(name='simulation.rejection-bucket-update-delay', short='rejection-bucket-update-delay', default='1',   quantity=quantities.dimensionless,
          caption='percentual update delay of buckets to decrease updating rate on event rejection'),
    Parameter(name='simulation.subclass-resolution-factor',                 default='8',          quantity=quantities.number,
          caption='number of subdivisions of the smallest resolved length (a granularity of sorts)'),
    Parameter(name='simulation.subclass-widening-fraction', short='spread',   default='0.05',       quantity=quantities.dimensionless,
          caption='widening fraction of sub-bucket widths'),
    Parameter(name='simulation.r-bins', short='r-bins', default='10',       quantity=quantities.dimensionless,
          caption='number of radial bins in the ring'),
    Parameter(name='simulation.M-bins-per-decade', short='M-bins-per-decade', default='2 (@ 1.e+24 g)',        dtype=planets.types.LogBucketDensity(quantity=quantities.g, unit='MEarth'),
          caption='number of bins per decadic order of magnitude of swarm mass'),
    Parameter(name='simulation.m-bins-per-decade', short='m-bins-per-decade', default='2 (@ 1.e+24 g)',        dtype=planets.types.LogBucketDensity(quantity=quantities.g, unit='MEarth'),
          caption='number of bins per decadic order of magnitude of particle mass'),
    Parameter(name='simulation.e-bins-per-decade', short='e-bins-per-decade', default='1 (≥ 1.e-6)',        dtype=planets.types.LogBucketDensity(quantity=quantities.dimensionless),
          caption='number of bins per decadic order of magnitude of eccentricity'),
    Parameter(name='simulation.sininc-bins-per-decade', short='sininc-bins-per-decade', default='1 (≥ 1.e-6)', dtype=planets.types.LogBucketDensity(quantity=quantities.dimensionless),
          caption='number of bins per decadic order of magnitude of inclination'),
    Parameter(name='simulation.mass-growth-factor',     default='0',        quantity=quantities.percentage, unit='%',
          caption='minimal percentaged mass increase in coagulation'),
    Parameter(name='simulation.velocity-growth-factor', default='0',        quantity=quantities.percentage, unit='%',
          caption='minimal percentaged velocity change in stirring/friction'),
    Parameter(name='simulation.velocity-growth-rate',   default='0/yr',     quantity=quantities.Hz,        unit='yr⁻¹',
          caption='maximal interaction rate for stochastic velocity changes'),
    Parameter(name='simulation.timesteps-per-orbit',    default='20',       quantity=quantities.dimensionless,
          caption='how many timesteps to take for the smallest orbit represented in the system'),
    Parameter(name='simulation.min-sync-time',          default='0 yr',     quantity=quantities.s,         unit='yr',
          caption='minimal time interval after which to synchronize N-body state and stochastic simulation state'),
    Parameter(name='simulation.nPlt',     short='nPlt', default='1',        quantity=quantities.number,
          caption='number of tracers for planetesimals'),
    Parameter(name='simulation.nPltR',    short='nPltR', default='0',       quantity=quantities.number,
          caption='number of unallocated tracers to reserve for planetesimals'),
    Parameter(name='simulation.nE',       short='nE',   default='0',        quantity=quantities.number,
          caption='number of tracers for embryos'),
    Parameter(name='simulation.hierarchical-ordering-base', default='10',   quantity=quantities.dimensionless,
          caption='base of logarithmic buckets in hierarchical ordering'),
    Parameter(name='simulation.random-seed',            default='42',       quantity=quantities.number,
          caption='random seed'),

    # simulation metadata
    Parameter(name='simulation.name',     short='name', default='',
          caption='name of simulation scenario'),
    Parameter(name='simulation.id',       short='id',   default='',
          caption='id of simulation scenario'),

    # snapshot control
    Parameter(name='simulation.NSteps', short='NSteps', default='256',      quantity=quantities.number,
          caption='number of time steps'),
    Parameter(name='simulation.tMinLinear',             default='0 years',  quantity=quantities.s,         unit='Myr',
          caption='minimal time on linear timescale'),
    Parameter(name='simulation.tMinLog',                default='100 years', quantity=quantities.s,         unit='Myr',
          caption='minimal time on logarithmic timescale'),
    Parameter(name='simulation.T', short='T',           default='100 years', quantity=quantities.s,         unit='Myr',
          caption='total duration of simulation'),
])


# Define analysis parameters.
analysis_params = ParameterSet(name='Analysis of parameter study', short_name='analysis', parameters=[
    Parameter(name='analysis.mTh', short='mTh',         default='MEarth',   quantity=quantities.g,          unit='MEarth',
          caption='threshold mass for planetesimals'),
    Parameter(name='analysis.tTh', short='tTh',         default='1 Myr',    quantity=quantities.s,          unit='Myr',
          caption='threshold time for planetesimals')
])

def analysis_params_equal(lhs, rhs):
    return lhs.analysis.mTh == rhs.analysis.mTh \
       and lhs.analysis.tTh == rhs.analysis.tTh


# Define simulation analysis results.
analysis_results = ParameterSet(name='Simulation analysis results', short_name='analysis-results', parameters=[
    Parameter(name='results.planetesimal.RMax',         default=None,       quantity=quantities.cm,         unit='km',
          caption='maximum planetesimal radius'),
    Parameter(name='results.planetesimal.mMax',         default=None,       quantity=quantities.g,          unit='MEarth',
          caption='maximum planetesimal mass'),
    Parameter(name='results.planetesimal.tMin_mTh',     default=None,       quantity=quantities.s,          unit='Myr',
          caption='time when threshold planetesimal mass is reached'),
    Parameter(name='results.planetesimal.R_mMax_tTh',   default=None,       quantity=quantities.cm,         unit='km',
          caption='radius of planetesimal with maximum mass at threshold time'),
    Parameter(name='results.planetesimal.mMax_tTh',     default=None,       quantity=quantities.g,          unit='MEarth',
          caption='maximum planetesimal mass at threshold time'),
    Parameter(name='results.planetesimal.min_Δv/vh',    default=None,       quantity=quantities.dimensionless,
          caption='minimum value of Δv/vh (relative velocity / Hill velocity)'),
    Parameter(name='results.planetesimal.max_vEsc/v',   default=None,       quantity=quantities.dimensionless,
          caption='maximum value of vEsc/v (escape velocity / kinetic velocity)'),
    Parameter(name='results.planetesimal.nExcess',      default='0',        quantity=quantities.number,
          caption='excess count of unresolved planetesimals'),
])


# Given a defining subset of particle traits, fill in the omitted traits.
def fill_implicit_args(args):
    pltsm = args.planetesimal
    emb = args.embryo

    context = 'ring of planetesimals'
    if args.ring.r is None and (any(x is not None for x in [pltsm.e, pltsm.i, pltsm.Δv, pltsm.Δv_vh]) or any(x is not None for x in [emb.m, emb.M, emb.ρ, emb.N, emb.R, emb.e, emb.i, emb.Δv, emb.Δv_vh])):
        raise RuntimeError(context + ': must specify ring radius if any orbital properties of planetesimals, embryos, or planets are specified')

    if args.ring.r is not None:
        if args.ring.Δr is not None or args.ring.Δr_2 is not None:
            args.ring.Δr, args.ring.Δr_2 = implicit.compute_dist_traits(args.ring.Δr, args.ring.Δr_2, context=context)

        # planetesimals are required
        pltsm.m, pltsm.M, pltsm.ρ, pltsm.N, pltsm.R = implicit.compute_intrinsic_traits(
            pltsm.m, pltsm.M, pltsm.ρ, pltsm.N, pltsm.R,
            context=context)
        if any(x is not None for x in [pltsm.e, pltsm.i, pltsm.Δv, pltsm.Δv_vh]):
            pltsm.e, pltsm.i, pltsm.sini, pltsm.Δv, pltsm.Δv_vh = implicit.compute_kinetic_traits(
                pltsm.e, pltsm.i, pltsm.sini, pltsm.Δv, pltsm.Δv_vh, pltsm.m, args.ring.r, args.star.M, context=context)
        else:
            pltsm.e, pltsm.i, pltsm.sini, pltsm.Δv, pltsm.Δv_vh = 0., 0., 0., 0., 0.  # no kinetic traits are given; default to 0

        # embryos are optional
        if any(x is not None for x in [emb.m, emb.M, emb.ρ, emb.N, emb.R]):
            emb.m, emb.M, emb.ρ, emb.N, emb.R = implicit.compute_intrinsic_traits(
            emb.m, emb.M, emb.ρ, emb.N, emb.R,
            context=context)
        else:
            emb.m, emb.M, emb.ρ, emb.N, emb.R = 0., 0., 0., 0, 0.
        if any(x is not None for x in [emb.e, emb.i, emb.Δv, emb.Δv_vh]):
            emb.e, emb.i, emb.sini, emb.Δv, emb.Δv_vh = implicit.compute_kinetic_traits(
                emb.e, emb.i, emb.sini, emb.Δv, emb.Δv_vh, emb.m, args.ring.r, args.star.M, context=context)
        else:
            emb.e, emb.i, emb.sini, emb.Δv, emb.Δv_vh = 0., 0., 0., 0., 0.  # no kinetic traits are given; default to 0

        # radial zone settings
        if args.zones.ΔrMin is None and args.collisions.kernel == 'geometric-radial':
            if args.ring.Δr is None:
                raise RuntimeError(context + ': must specify ring width')
            args.zones.ΔrMin = 2*args.ring.Δr/args.simulation.nPlt  # TODO: refine constant

        # gas profile settings
        if args.gas.M is not None or args.gas.ρ0 is not None or args.gas.Σ0 is not None:
            args.gas.Σ0, args.gas.ρ0, args.gas.T0, args.gas.Tmin, args.gas.cs0, args.gas.Hp0, args.gas.hp0 = implicit.compute_gas_traits(
                args.gas.Σ0, args.gas.ρ0, args.gas.T0, args.gas.Tmin, args.gas.cs0, args.gas.Hp0, args.gas.hp0,
                M=args.gas.M, rlo=args.gas.rmin, rhi=args.gas.rmax,
                MStar=args.star.M, LStar=args.star.L,
                φ=args.gas.φ,
                r0=args.gas.r0, p=args.gas.p,
                context=context)

        # planet settings
        if args.gas.mPlanet is not None:
            import rpmc
            gas_profile_base_args = rpmc.GasProfileBaseArgs(args)
            args.gas.rPlanet, args.gas.rDustTrap = implicit.compute_dust_trapping_planet_traits(
                args.gas.rPlanet, args.gas.rDustTrap,
                gas_profile_base_args=gas_profile_base_args,
                mPlanet=args.gas.mPlanet,
                context=context)
        if args.planet.m is not None:
            args.planet.ρ, args.planet.R, args.planet.e, args.planet.i, args.planet.sini = implicit.compute_planet_traits(
                args.planet.ρ, args.planet.R, args.planet.e, args.planet.i, args.planet.sini,
                m=args.planet.m,
                context=context)
            if args.planet.a is None and args.gas.mPlanet is not None:
                args.planet.a = args.gas.rPlanet


def get_times_lin_log(args):
    from const.cgs import year

    tLinear = np.linspace(start=args.simulation.tMinLinear, stop=args.simulation.T, num=args.simulation.NSteps//2, endpoint=True)
    if args.simulation.T < args.simulation.tMinLog:
        tLog = []
    else:
        tLog = np.logspace(start=np.log10(args.simulation.tMinLog), stop=np.log10(args.simulation.T), num=args.simulation.NSteps - len(tLinear), endpoint=True)
    return tLinear, tLog

def get_times(args):
    tLinear, tLog = get_times_lin_log(args)
    return np.unique(np.concatenate([tLinear, tLog]))


class SimulationData:
    def __init__(self, timesteps: Union[Dict[str, Any], _data.Memoize], snapshots = None, histograms = None,
                 state = None, sim = None):
        assert timesteps is not None
        self.timesteps = timesteps if isinstance(timesteps, _data.Memoize) else _data.Memoize(value=timesteps)
        self.snapshots = snapshots if isinstance(snapshots, _data.Memoize) else _data.Memoize(value=snapshots)
        self.histograms = histograms if isinstance(histograms, _data.Memoize) else _data.Memoize(value=histograms)
        def _snapshots_by_time():
            data = self.snapshots()
            grouping = data.groupby(data.t, as_index=False)
            ts = np.fromiter(grouping.indices.keys(), dtype=float)
            def fs(t):
                return grouping.get_group(t)
            return ts, fs
        self.snapshots_by_time = _data.Memoize(func=_snapshots_by_time)
        self.state = state
        self.sim = sim
        #self.archive = None

    @staticmethod
    def load(archive):  # Note that `archive` must be kept alive for the lifetime of the `SimulationData` object because of lazy loading
        raw_timesteps = archive.entries['timesteps'].load_data()
        snapshots = _data.Memoize(func=lambda: archive.entries['snapshots'].load_data())
        histograms = _data.MemoizeResource(func=lambda: archive.entries['stir-histograms'].load_data())

        # Due to possible loss of precision in serialization of floating-point data, we have to look for the nearest times rather
        # than doing a straight lookup with `get_group()`.
        def _timesteps():
            data = snapshots()
            times = np.unique(data.t)
            return {
                key: projection.find_nearest_indices(times, value)
                for key, value in raw_timesteps.items()
            }
        timesteps = _data.Memoize(func=_timesteps)

        data = SimulationData(timesteps=timesteps, snapshots=snapshots, histograms=histograms)
        #data._archive = archive
        return data

    @staticmethod
    def save(archive_writer, sim_data):
        archive_writer.write_file(name='timesteps', data=sim_data.timesteps())
        if not sim_data.snapshots.is_none():
            archive_writer.write_file(name='snapshots', data=sim_data.snapshots())
        if not sim_data.histograms.is_none():
            archive_writer.write_file(name='stir-histograms', data=sim_data.histograms())

    def close(self):
        if hasattr(self.timesteps, 'close'):
            self.timesteps.close()
        if hasattr(self.snapshots, 'close'):
            self.snapshots.close()
        if hasattr(self.histograms, 'close'):
            self.histograms.close()
        #if self._archive is not None:
        #    self._archive.close()
        #    self._archive = None
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()


def run(args, rng, log: bool = False, display: bool = False, timesteps: Optional[Dict[str, Sequence]] = { }, inspect_callback=None):
    """Run simulation and return snapshot data."""

    import rpmc

    def fill(*args):
        return np.concatenate([np.zeros(n) + x for n, x in args])
    def fill_random(*args, lo, hi):
        return np.concatenate([rng.random(size=n)*(hi - lo) + lo for n in args])
    #def fill_random_poisson(*args):
    #    return np.concatenate([rng.poisson(lam=float(x), size=n) for n, x in args])

    # Initialize simulation state.
    if args.planet.m is not None:
        have_planet = True
        mP, rP, ρP = args.planet.m, args.planet.a, args.planet.ρ
        eP, siniP = args.planet.e, args.planet.sini
        nP = 1
    else:
        have_planet = False
        mP, rP, ρP = 0., 0., 1.
        eP, siniP = 0., 0.
        nP = 0
    nPlt, nPltR, nE = args.simulation.nPlt, args.simulation.nPltR, args.simulation.nE
    n = nP + nPlt + nPltR + nE
    pltsm = args.planetesimal
    emb = args.embryo
    ME = (emb.M/args.simulation.nE if np.ndim(emb.M) == 0 else emb.M) if args.simulation.nE != 0 else 0.  # TODO: check if this adds up
    r = args.ring.r if args.ring.r is not None else 0.
    Δr = args.ring.Δr if args.ring.Δr is not None else 0.
    rE = emb.a if emb.a is not None else np.sort(r + (rng.random(size=nE) - 0.5)*Δr)
    if nPlt > 0:
        MPlt = pltsm.M/nPlt if np.ndim(pltsm.M) == 0 else pltsm.M
        rPlt = pltsm.a if pltsm.a is not None else np.sort(r + (rng.random(size=nPlt) - 0.5)*Δr)
        ePlt = pltsm.e if pltsm.e is not None else 0.
        siniPlt = pltsm.sini if pltsm.sini is not None else 0.
        ρPlt = pltsm.ρ if pltsm.ρ is not None else 1.
        mPlt = pltsm.m if pltsm.m is not None else 0.
    else:
        MPlt = 0.
        rPlt = 0.
        ePlt = 0.
        siniPlt = 0.
        ρPlt = 0.
        mPlt = 0.
    eE = emb.e if emb.e is not None else 0.
    siniE = emb.sini if emb.sini is not None else 0.
    ρE = emb.ρ if emb.ρ is not None else 1.
    mE = emb.m if emb.m is not None else 0.
    Ms = fill((nP, mP), (nE, ME), (nPlt, MPlt), (nPltR, 0.))
    ms = fill((nP, mP), (nE, mE), (nPlt, mPlt), (nPltR, 0.))
    Ns = np.divide(Ms, ms, out=np.zeros_like(Ms), where=ms != 0.)
    state = pd.DataFrame.from_dict({
        'id':             np.array(range(n), dtype=np.int32),                     # tracer id
        'a':              fill((nP, rP), (nE, rE), (nPlt, rPlt), (nPltR, 0.)),    # radial position of tracer  (cm)
        'e':              fill((nP, eP), (nE, eE), (nPlt, ePlt), (nPltR, 0.)),    # tracer eccentricity
        'sininc':         fill((nP, siniP), (nE, siniE), (nPlt, siniPlt), (nPltR, 0.)),    # tracer inclination
        'vr':             fill((n, 0.)),                                          # radial velocity  vᵣ = da/dt  (cm/s)
        'vφ':             fill((n, 0.)),                                          # systematic azimuthal velocity  vᵩ  (cm/s)
        'St':             fill((n, 0.)),                                          # Stokes number  St
        'M':              Ms,                                                     # total amount of mass represented by tracer  (g)
        'm':              ms,                                                     # tracer mass  (g)
        'N':              Ns,                                                     # number of particles in swarm
        'ρ':              fill((nP, ρP), (nE, ρE), (nPlt, ρPlt), (nPltR, 0.)),    # tracer bulk density  (g/cm³)
        'hd':             fill((n, 0.))                                           # dust scale height  (cm)
    })
    state.sort_values(by=['a', 'm'], inplace=True)

    # Create simulation.
    if args.simulation.method == 'rpmc':
        sim = rpmc.RPMCSimulation(args=args, state=state, log=log, display=display)
    elif args.simulation.method == 'rpmc-traditional':
        sim = rpmc.RPMCTraditionalSimulation(args=args, state=state, log=log, display=display)
    else:
        raise RuntimeError('simulation.method: unknown argument "{}"; supported arguments: {}'.format(args.simulation.method, ['rpmc', 'rpmc-traditional']))

    # Merge timesteps.
    ts_linear, ts_log = get_times_lin_log(args)
    ts = np.unique(np.concatenate([ts_linear, ts_log, *timesteps.values()]))
    timesteps = timesteps.copy()
    timesteps['linear'] = ts_linear
    timesteps['log'] = ts_log
    timesteps['unified'] = ts

    if inspect_callback is not None:
        ts_inspect = timesteps['inspect']
        i_inspect = 0

    # Allocate dataframe for snapshot data.
    #ts = get_times(args)
    snapshot_rows = n*len(ts)
    snapshot_data = pd.DataFrame.from_dict({
        'id':             np.zeros(shape=[snapshot_rows], dtype=np.int32),  # tracer id
        't':              np.zeros(shape=[snapshot_rows], dtype=float),  # time  (s)
        'a':              np.zeros(shape=[snapshot_rows], dtype=float),  # semimajor axis of tracer  (cm)
        'e':              np.zeros(shape=[snapshot_rows], dtype=float),  # tracer eccentricity
        'sininc':         np.zeros(shape=[snapshot_rows], dtype=float),  # tracer inclination
        'vr':             np.zeros(shape=[snapshot_rows], dtype=float),  # radial velocity  uᵣ = da/dt  (cm/s)
        'vφ':             np.zeros(shape=[snapshot_rows], dtype=float),  # systematic azimuthal velocity  vᵩ  (cm/s)
        'St':             np.zeros(shape=[snapshot_rows], dtype=float),  # Stokes number  St
        'M':              np.zeros(shape=[snapshot_rows], dtype=float),  # total amount of mass represented by tracer  (g)
        'm':              np.zeros(shape=[snapshot_rows], dtype=float),  # tracer mass  (g)
        'N':              np.zeros(shape=[snapshot_rows], dtype=float),  # number of particles represented
        'R':              np.zeros(shape=[snapshot_rows], dtype=float)   # tracer bulk radius  (cm)
    })

    # Run simulation.
    print('Running simulation... ', end=None if log else '')
    t0 = time.perf_counter()
    def snapshot_callback(ti, t):
        nonlocal i_inspect

        ## Run until given time.
        if log:
            print('Simulation run until t = {}.'.format(quantities.s.format(t, 'Myr')))

        if inspect_callback is not None:
            if i_inspect < len(ts_inspect) and t == ts_inspect[i_inspect]:
                inspect_callback(sim, state, i_inspect, t)
                i_inspect += 1

        # Take snapshot of simulation state.
        rows = slice(ti*n, (ti + 1)*n)
        snapshot_data.id.to_numpy()[rows] = state.id
        snapshot_data.t.to_numpy()[rows] = t
        snapshot_data.m.to_numpy()[rows] = state.m
        snapshot_data.a.to_numpy()[rows] = state.a
        snapshot_data.e.to_numpy()[rows] = state.e
        snapshot_data.sininc.to_numpy()[rows] = state.sininc
        snapshot_data.vr.to_numpy()[rows] = state.vr
        snapshot_data.vφ.to_numpy()[rows] = state.vφ
        snapshot_data.St.to_numpy()[rows] = state.St
        snapshot_data.M.to_numpy()[rows] = state.M
        #snapshot_data.N.to_numpy()[rows] = np.where(state.M != 0, state.M/state.m, 0.)
        snapshot_data.N.to_numpy()[rows] = state.N
        snapshot_data.R.to_numpy()[rows] = (state.m/(4/3*np.pi*state.ρ))**(1/3)

    sim.run_to(t_end=ts[-1], snapshot_times=ts, snapshot_callback=snapshot_callback)
    t1 = time.perf_counter()

    print('done.\nElapsed time: {} s'.format(t1 - t0))

    return SimulationData(timesteps=timesteps, snapshots=snapshot_data, state=state, sim=sim)


def square_subplots(fig, rows, cols):
    # Taken from https://stackoverflow.com/a/51483579.
    #rows, cols = ax.get_subplotspec().get_gridspec().get_geometry()
    l = fig.subplotpars.left
    r = fig.subplotpars.right
    t = fig.subplotpars.top
    b = fig.subplotpars.bottom
    wspace = fig.subplotpars.wspace
    hspace = fig.subplotpars.hspace
    figw,figh = fig.get_size_inches()

    axw = figw*(r-l)/(cols+(cols-1)*wspace)
    axh = figh*(t-b)/(rows+(rows-1)*hspace)
    axs = min(axw,axh)
    w = (1-axs/figw*(cols+(cols-1)*wspace))/2.
    h = (1-axs/figh*(rows+(rows-1)*hspace))/2.
    fig.subplots_adjust(bottom=h, top=1-h, left=w, right=1-w)


def sig(num, digits=3):
    "Return number formatted for significant digits"
    # Taken from https://stackoverflow.com/a/67587629
    if num == 0:
        return 0
    negative = '-' if num < 0 else ''
    num = abs(float(num))
    power = math.log(num, 10)
    if num < 1:
        step = int(10**(-int(power) + digits) * num)
        return negative + '0.' + '0' * -int(power) + str(int(step)).rstrip('0')
    elif power < digits - 1:
        return negative + ('{0:.' + str(digits) + 'g}').format(num)
    else:
        return negative + str(int(num))


def make_snapshot_plots(sim_args, sim_data: SimulationData):
    """Generate requested plots from simulation snapshot data."""
    import matplotlib
    import matplotlib.animation
    import matplotlib.pyplot as plt
    import matplotlib.offsetbox
    import matplotlib.ticker as ticker
    from const.cgs import year, km, AU, Mea, MS, GG

    class AnchoredHScaleBar(matplotlib.offsetbox.AnchoredOffsetbox):
        """ size: length of bar in data units
            extent : height of bar ends in axes units """
        # taken from https://stackoverflow.com/a/43343934
        def __init__(self, size=1, extent=0.03, label="", loc=2, ax=None,
                    pad=0.4, borderpad=0.5, ppad=0, sep=2, prop=None, 
                    frameon=True, align='center', linekw={}, textkw={}, **kwargs):
            from matplotlib.lines import Line2D
            if not ax:
                ax = plt.gca()
            trans = ax.get_xaxis_transform()
            size_bar = matplotlib.offsetbox.AuxTransformBox(trans)
            line = Line2D([0,size],[0,0], **linekw)
            vline1 = Line2D([0,0],[-extent/2,extent/2], **linekw)
            vline2 = Line2D([size,size],[-extent/2,extent/2], **linekw)
            size_bar.add_artist(line)
            size_bar.add_artist(vline1)
            size_bar.add_artist(vline2)
            txt = matplotlib.offsetbox.TextArea(label, minimumdescent=False, textprops=dict(horizontalalignment=align, **textkw))
            self.vpac = matplotlib.offsetbox.VPacker(children=[size_bar,txt],
                                    align=align, pad=ppad, sep=sep)
            matplotlib.offsetbox.AnchoredOffsetbox.__init__(self, loc, pad=pad,
                    borderpad=borderpad, child=self.vpac, prop=prop, frameon=frameon,
                    **kwargs)

    class AnchoredVScaleBar(matplotlib.offsetbox.AnchoredOffsetbox):
        """ size: length of bar in data units
            extent : height of bar ends in axes units """
        # taken from https://stackoverflow.com/a/43343934
        def __init__(self, size=1, extent=0.03, label="", loc=2, ax=None,
                    pad=0.4, borderpad=0.5, ppad=0, sep=2, prop=None, 
                    frameon=True, align='center', linekw={}, textkw={}, **kwargs):
            from matplotlib.lines import Line2D
            if not ax:
                ax = plt.gca()
            trans = ax.get_yaxis_transform()
            size_bar = matplotlib.offsetbox.AuxTransformBox(trans)
            line = Line2D([0,0],[0,size], **linekw)
            vline1 = Line2D([-extent/2,extent/2],[0,0], **linekw)
            vline2 = Line2D([-extent/2,extent/2],[size,size], **linekw)
            size_bar.add_artist(line)
            size_bar.add_artist(vline1)
            size_bar.add_artist(vline2)
            txt = matplotlib.offsetbox.TextArea(label, minimumdescent=False, textprops=dict(rotation='vertical', horizontalalignment=align, **textkw))
            self.hpac = matplotlib.offsetbox.HPacker(children=[size_bar,txt],
                                    align=align, pad=ppad, sep=sep)
            matplotlib.offsetbox.AnchoredOffsetbox.__init__(self, loc, pad=pad,
                    borderpad=borderpad, child=self.hpac, prop=prop, frameon=frameon,
                    **kwargs)


    @plt.FuncFormatter
    def fake_log(value, tick_position):
        return '$10^{{{}}}$'.format(int(value))

    Ny = 128

    # compute reference quantities
    ΩK = np.sqrt(GG*MS/sim_args.ring.r**3)  # Kepler orbital velocity (1/s)
    vK = sim_args.ring.r*ΩK  # Kepler velocity of planetesimal (cm/s)

    # load snapshot data
    snapshot_data = sim_data.snapshots()
    #tLinear, tLog = get_times_lin_log(sim_args)
    tLinear, tLog = sim_data.timesteps()['linear'], sim_data.timesteps()['log']

    # preprocess
    time = snapshot_data.t.unique()
    rms = lambda x: np.sqrt(np.sum(x**2)/len(x))
    sqrtSum = lambda x: np.sqrt(np.sum(x))
    theta = lambda x: np.sum(np.minimum(x, 1))
    #MRef = np.repeat(np.concatenate([np.repeat(sim_args.embryo.M, sim_args.simulation.nE), np.repeat(sim_args.planetesimal.M, sim_args.simulation.nPlt)]), len(time))
    snapshot_data['v'] = vK*np.sqrt(snapshot_data.e**2 + snapshot_data.sininc**2)  # rms velocities (cm/s)
    h = (snapshot_data.m/(3*MS))**(1/3)  # dimensionless Hill radius
    Rh = sim_args.ring.r*h  # Hill radius (cm)
    vh = Rh*ΩK  # Hill velocity (cm/s)
    snapshot_data['vEsc'] = np.sqrt(2*GG*snapshot_data.m/snapshot_data.R)  # escape velocity from planetesimal (cm/s)
    snapshot_data['Δve_vh'] = np.sqrt(snapshot_data.e**2)*(vK/vh)
    snapshot_data['vEsc_v'] = snapshot_data.vEsc/snapshot_data.v
    MRef = np.sum(sim_args.embryo.M) + np.sum(sim_args.planetesimal.M)
    weights = snapshot_data.M/MRef
    wdata = pd.DataFrame.from_dict({
        'id': snapshot_data.id,
        't': snapshot_data.t,
        'm': snapshot_data.m*weights,
        'N': snapshot_data.N,
        'R': snapshot_data.R*weights,
        'vSq': snapshot_data.v**2*weights,
        'eSq': snapshot_data.e**2*weights,
        'vEscSq_vSq': snapshot_data.vEsc_v**2*weights,
        'Δve_vhSq': snapshot_data.Δve_vh**2*weights
    })
    data_by_time = snapshot_data \
        .groupby(snapshot_data.t, as_index=False)  # TODO: why as_index?
    data_sum = data_by_time \
        .agg({
            'collision_rate': 'sum',
            'N': 'sum'
        })
    have_planet = sim_args.dust_trap.mode == 'synchronized-pressure-bump'
    nP = 1 if have_planet else 0
    embdata = snapshot_data[(snapshot_data.id >= nP) & (snapshot_data.id < nP + sim_args.simulation.nE)]
    #print(embdata)
    embdata_by_time = embdata \
        .groupby(embdata.t, as_index=False)  # TODO: why as_index?
    embdata_sum = embdata_by_time \
        .agg({
            'M': 'sum',
            'collision_rate': 'sum',
            #'N': 'sum'
        })
    #print(embdata)
    wdata_by_time = wdata \
        .groupby(wdata.t, as_index=False)  # TODO: why as_index?
    data_mean = wdata_by_time \
        .agg({
            'm': 'sum',
            'N': theta,
            'R': 'sum',
            #'vSq': sqrtSum,
            'eSq': sqrtSum,
            'vEscSq_vSq': sqrtSum,
            'Δve_vhSq': sqrtSum
        })
    mean_m = data_mean.m.to_numpy()
    mean_R = data_mean.R.to_numpy()
    total_N = data_sum.N.to_numpy()
    total_n = data_mean.N.to_numpy()
    total_collision_rate = data_sum.collision_rate.to_numpy()
    #rms_v = data_mean.vSq.to_numpy()
    rms_e = data_mean.eSq.to_numpy()
    rms_vEsc_v = data_mean.vEscSq_vSq.to_numpy()
    #rms_inc = data_mean.incSq.to_numpy()
    rms_Δve_vh = data_mean.Δve_vhSq.to_numpy()
    #rms_Δvinc_vh = data_mean.Δvinc_vhSq.to_numpy()
    MRefE = sim_args.embryo.M
    #MRefE = embdata_sum.M.to_numpy()  # TODO: ??
    #print(MRefE)
    weightsE = embdata.M/MRefE
    embwdata = pd.DataFrame.from_dict({
        'id': embdata.id,
        't': embdata.t,
        'm': embdata.m*weightsE,
        'N': embdata.N,
        'R': embdata.R*weightsE,
        'vSq': embdata.v**2*weightsE,
        'eSq': embdata.e**2*weightsE,
        'vEscSq_vSq': embdata.vEsc_v**2*weightsE,
        #'incSq': embdata.incSq*weightsE,
        'Δve_vhSq': embdata.Δve_vh**2*weightsE,
        #'Δvinc_vhSq': embdata.Δvinc_vh**2*weightsE
    })
    embwdata_by_time = embwdata \
        .groupby(embwdata.t, as_index=False)  # TODO: why as_index?
    embdata_mean = embwdata_by_time \
        .agg({
            'm': 'sum',
            'R': 'sum',
            #'vSq': sqrtSum,
            'eSq': sqrtSum,
            'vEscSq_vSq': sqrtSum,
            #'incSq': sqrtSum,
            'Δve_vhSq': sqrtSum,
            #'Δvinc_vhSq': sqrtSum
        })
    mean_mE = embdata_mean.m.to_numpy()
    mean_RE = embdata_mean.R.to_numpy()
    total_collision_rateE = embdata_sum.collision_rate.to_numpy()
    #rms_vE = embdata_mean.vSq.to_numpy()
    rms_eE = embdata_mean.eSq.to_numpy()
    rms_vEscE_vE = embdata_mean.vEscSq_vSq.to_numpy()
    #rms_incE = embdata_mean.incSq.to_numpy()
    rms_Δve_vhE = embdata_mean.Δve_vhSq.to_numpy()
    #rms_Δvinc_vhE = embdata_mean.Δvinc_vhSq.to_numpy()

    minor_ticks = np.array([i + np.log10(k) for k in range(2, 10) for i in range(-20, 15)])

    # Due to loss of precision in serialization of floating-point data, we have to look for the nearest times rather than doing
    # a straight lookup with `get_group()`.
    group_times = np.fromiter(data_by_time.indices.keys(), dtype=float)
    iLinear = projection.find_nearest_indices(group_times, tLinear)
    iLog = projection.find_nearest_indices(group_times, tLog)
    tLinearP = group_times[iLinear]
    tLogP = group_times[iLog]
    #data_by_tLinear = [data_by_time.get_group(t) for t in tLinearP]
    #data_by_tLog = [data_by_time.get_group(t) for t in tLogP]

    class ColorScheme:
        def __init__(self, hist, mean, embryo):
            self.hist_cmap = plt.get_cmap(hist)
            self.mean = mean
            self.embryo = embryo

    csMass          = ColorScheme(hist='Blues',   mean='tab:orange', embryo='tab:brown')
    csRadius        = ColorScheme(hist='Blues',   mean='tab:orange', embryo='tab:brown')
    csCollRate      = ColorScheme(hist='Purples', mean='tab:orange', embryo='tab:brown')
    csNumberDensity = ColorScheme(hist='Greys',   mean='tab:orange', embryo='tab:brown')
    csVelEcc        = ColorScheme(hist='bone_r',  mean='tab:orange', embryo='tab:brown')
    wMean = 2.5
    wEmb = 2.5

    class Plots:
        @staticmethod
        def positive_finite(Y, w):
            mask = (Y > 0.) & (w > 0.) & ~np.isinf(Y)
            return Y[mask], w[mask]

        # self.plot_loglog_histogram(fig=fig, ax=ax, data_by_time=data_by_time, Y_w=lambda data: (data.R, data.M), Ymean=mean_R, YmeanE=mean_RE, Yunit=km)
        def plot_loglog_histogram(self, fig, ax, data_by_time, Y_w, color_scheme, Ymean=None, YmeanE=None, Yunit=1., mean_label='average', embryo_label='ova', nbin_factor=1., colorbar_pad=0.1, minor=True):
            if sim_args.simulation.nPlt > 1:
                data_by_tLog = [Plots.positive_finite(*Y_w(data_by_time.get_group(t))) for t in tLogP]
                hist, bins = projection.project_log_histograms(data_by_tLog, Ny, nbin_factor=nbin_factor)
                Y0, Y1 = bins[0], bins[-1]
                print('min(hist): {}, max(hist): {}'.format(np.min(hist), np.max(hist)))
                im = ax.imshow(hist, interpolation='nearest', origin='lower',
                               extent=[np.log10(tLogP[0]/year/1.e+6), np.log10(tLogP[-1]/year/1.e+6), np.log10(Y0/Yunit), np.log10(Y1/Yunit)],
                               cmap=color_scheme.hist_cmap, norm=matplotlib.colors.LogNorm())
                cb = fig.colorbar(im, ax=ax, pad=colorbar_pad)
                ax.set_aspect(np.log10(tLogP[-1]/tLogP[0]) / np.log10(Y1/Y0))
            ax.xaxis.set_major_formatter(fake_log)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(1.))
            ax.xaxis.set_minor_formatter(ticker.NullFormatter())
            ax.xaxis.set_minor_locator(ticker.FixedLocator(minor_ticks))
            ax.yaxis.set_major_formatter(fake_log)
            ax.yaxis.set_major_locator(ticker.MultipleLocator(1.))
            if minor:
                ax.yaxis.set_minor_formatter(ticker.NullFormatter())
                ax.yaxis.set_minor_locator(ticker.FixedLocator(minor_ticks))
            ax.grid(which='major', axis='both')
            if minor:
                ax.grid(which='minor', axis='both', linestyle='--', color='lightgray')
            else:
                ax.grid(which='minor', axis='x', linestyle='--', color='lightgray')
            Xs = time[iLog]
            if Ymean is not None:
                Ys = Ymean[iLog]
                ax.plot(np.log10(Xs[Ys > 0.]/year/1.e+6), np.log10(Ys[Ys > 0.]/Yunit), label=mean_label, color=color_scheme.mean, linewidth=wMean)
            if sim_args.simulation.nE >= 1 and YmeanE is not None:
                Ys = YmeanE[iLog]
                ax.plot(np.log10(Xs[Ys > 0.]/year/1.e+6), np.log10(Ys[Ys > 0.]/Yunit), label=embryo_label, color=color_scheme.embryo, linewidth=wEmb)
            #ax.set_ylim(bottom=np.log10(Y0/Yunit), top=np.log10(Y1/Yunit))
            if Ymean is not None or (sim_args.simulation.nE >= 1 and YmeanE is not None):
                ax.legend()

        def plot_linlog_histogram(self, fig, ax, data_by_time, Y_w, color_scheme, Ymean=None, YmeanE=None, Yunit=1., mean_label='average', embryo_label='ova', nbin_factor=1., colorbar_pad=0.1, minor=True):
            if sim_args.simulation.nPlt > 1:
                data_by_tLinear = [Plots.positive_finite(*Y_w(data_by_time.get_group(t))) for t in tLinearP]
                hist, bins = projection.project_log_histograms(data_by_tLinear, Ny, nbin_factor=nbin_factor)
                Y0, Y1 = bins[0], bins[-1]
                print('Y0: {}, Y1: {}'.format(Y0, Y1))
                print('min(hist): {}, max(hist): {}'.format(np.min(hist), np.max(hist)))
                im = ax.imshow(hist, interpolation='nearest', origin='lower',
                               extent=[tLinearP[0]/year/1.e+6, tLinearP[-1]/year/1.e+6, np.log10(Y0/Yunit), np.log10(Y1/Yunit)],
                               cmap=color_scheme.hist_cmap, norm=matplotlib.colors.LogNorm())
                cb = fig.colorbar(im, ax=ax, pad=colorbar_pad)
                ax.set_aspect((sim_args.simulation.T - sim_args.simulation.tMinLinear)/year/1.e+6 / np.log10(Y1/Y0))
            ax.yaxis.set_major_formatter(fake_log)
            ax.yaxis.set_major_locator(ticker.MultipleLocator(1.))
            if minor:
                ax.yaxis.set_minor_formatter(ticker.NullFormatter())
                ax.yaxis.set_minor_locator(ticker.FixedLocator(minor_ticks))
            ax.grid(which='major', axis='both')
            if minor:
                ax.grid(which='minor', axis='y', linestyle='--', color='lightgray')
            Xs = time[iLinear]
            if Ymean is not None:
                Ys = Ymean[iLinear]
                ax.plot(Xs[Ys > 0.]/year/1.e+6, np.log10(Ys[Ys > 0.]/Yunit), label=mean_label, color=color_scheme.mean, linewidth=wMean)
            if sim_args.simulation.nE >= 1 and YmeanE is not None:
                Ys = YmeanE[iLinear]
                ax.plot(Xs[Ys > 0.]/year/1.e+6, np.log10(Ys[Ys > 0.]/Yunit), label=embryo_label, color=color_scheme.embryo, linewidth=wEmb)
            ax.set_xlim(left=sim_args.simulation.tMinLinear/year/1.e+6, right=sim_args.simulation.T/year/1.e+6)
            #ax.set_ylim(bottom=np.log10(Y0/Yunit), top=np.log10(Y1/Yunit))
            if Ymean is not None or (sim_args.simulation.nE >= 1 and YmeanE is not None):
                ax.legend()

        def plot_bulk_radius_to(self, fig, ax, *args, **kwargs):
            ax.set_title('bulk radii')
            ax.set_xlabel('time (Myr)')
            ax.set_ylabel('bulk radius (km)')
            self.plot_loglog_histogram(fig, ax, data_by_time=data_by_time, color_scheme=csRadius,
                Y_w=lambda data: (data.R, data.M), Ymean=mean_R, YmeanE=mean_RE, Yunit=km, *args, **kwargs)

        def plot_vEsc_v_to(self, fig, ax, *args, **kwargs):
            ax.set_title('escape velocity / kinetic velocity')
            ax.set_xlabel('time (Myr)')
            ax.set_ylabel('$v_\\mathrm{Esc}/\\left|\\Delta v\\right|$')
            self.plot_loglog_histogram(fig, ax, data_by_time=data_by_time, color_scheme=csVelEcc,
                Y_w=lambda data: (data.vEsc_v, data.M), Ymean=rms_vEsc_v, YmeanE=rms_vEscE_vE, *args, **kwargs, minor=False)

        def plot_mass_to(self, fig, ax, *args, **kwargs):
            ax.set_title('masses')
            ax.set_xlabel('time (Myr)')
            ax.set_ylabel('planetesimal mass ($M_{\\oplus}$)')
            self.plot_loglog_histogram(fig, ax, data_by_time=data_by_time, color_scheme=csMass,
                Y_w=lambda data: (data.m, data.M), Ymean=mean_m, YmeanE=mean_mE, Yunit=Mea, *args, **kwargs, minor=False)

        def plot_collision_rate_to(self, fig, ax, *args, **kwargs):
            ax.set_title('collision rates')
            ax.set_xlabel('time (Myr)')
            ax.set_ylabel('collision rate (1/Myr)')
            self.plot_linlog_histogram(fig, ax, data_by_time=data_by_time, color_scheme=csCollRate,
                Y_w=lambda data: (data.collision_rate, data.M), Ymean=total_collision_rate, YmeanE=total_collision_rateE, Yunit=1/(1.e+6*year),
                mean_label='total collision rate', embryo_label='total collision rate of ova', *args, **kwargs)

        def plot_logtime_collision_rate_to(self, fig, ax, *args, **kwargs):
            ax.set_title('collision rates')
            ax.set_xlabel('time (Myr)')
            ax.set_ylabel('collision rate (1/Myr)')
            self.plot_loglog_histogram(fig, ax, data_by_time=data_by_time, color_scheme=csCollRate,
                Y_w=lambda data: (data.collision_rate, data.M), Ymean=total_collision_rate, YmeanE=total_collision_rateE, Yunit=1/(1.e+6*year),
                mean_label='total collision rate', embryo_label='total collision rate of ova', *args, **kwargs)

        def plot_velocity_to(self, fig, ax, *args, **kwargs):
            ax.set_title('rms velocities')
            ax.set_xlabel('time (Myr)')
            ax.set_ylabel('rms velocity $\\sqrt{\\left<v^2\\right>}$ (cm/s)')
            self.plot_linlog_histogram(fig, ax, data_by_time=data_by_time, color_scheme=csVelEcc,
                Y_w=lambda data: (data.v, data.M), Ymean=rms_v, YmeanE=rms_vE, Yunit=1/(1.e+6*year),
                mean_label='rms average', embryo_label='rms ova', *args, **kwargs)

        def plot_eccentricity_to(self, fig, ax, *args, **kwargs):
            ax.set_title('rms eccentricities')
            ax.set_xlabel('time (Myr)')
            ax.set_ylabel('$\\sqrt{\\left<e^2\\right>}$')
            self.plot_loglog_histogram(fig, ax, data_by_time=data_by_time, color_scheme=csVelEcc,
                Y_w=lambda data: (data.e[data.e > 1.e-9], data.M[data.e > 1.e-9]), Ymean=rms_e, YmeanE=rms_eE,
                mean_label='rms average', embryo_label='rms ova', *args, **kwargs)

        def plot_veccentricity_to(self, fig, ax, *args, **kwargs):
            ax.set_title('rms eccentricities in Hill velocity units')
            ax.set_xlabel('time (Myr)')
            ax.set_ylabel('$v_\\mathrm{K} \\sqrt{\\left<e^2\\right>}/v_\\mathrm{h}$')
            self.plot_loglog_histogram(fig, ax, data_by_time=data_by_time, color_scheme=csVelEcc,
                Y_w=lambda data: (data.Δve_vh[data.Δve_vh > 1.e-9], data.M[data.Δve_vh > 1.e-9]), Ymean=rms_Δve_vh, YmeanE=rms_Δve_vhE,
                mean_label='rms average', embryo_label='rms ova', *args, **kwargs)

        def plot_tracer_distribution(self, fig, ax, *args, **kwargs):
            def weight(N):
                return np.floor(np.minimum(N, 1.))
            ax.set_title('tracer distribution')
            ax.set_xlabel('time (Myr)')
            ax.set_ylabel('mass ($M_{\\oplus}$)')
            self.plot_loglog_histogram(fig, ax, data_by_time=data_by_time, color_scheme=csNumberDensity,
                Y_w=lambda data: (data.m, weight(data.N)), Yunit=Mea, colorbar_pad=0.15, *args, **kwargs, minor=False)
            ax2 = ax.twinx()
            ax2.set_ylabel('tracer count')
            Xs = time[iLog]
            Yn = total_n[iLog]
            ax2.plot(np.log10(Xs/year/1.e+6), Yn, label='tracer count', color=csNumberDensity.embryo, linewidth=wEmb)
            #ax2.set_yscale('log')
            ax2.legend()

        def plot_normalized_tracer_distribution(self, fig, ax, *args, **kwargs):
            def percentual_mass(m, M):
                mask = M != 0.
                min_m = np.min(m[mask])
                max_m = np.max(m[mask])
                if max_m == min_m:
                    max_m *= 2
                #return (m - min_m)/(max_m - min_m)
                return 10**(np.log10(np.where(mask, m, min_m)/min_m)/np.log10(max_m/min_m))
            def weight(N):
                return np.floor(np.minimum(N, 1.))
            ax.set_title('tracer distribution, normalized range')
            ax.set_xlabel('time (Myr)')
            ax.set_ylabel('relative logarithmic mass $(\log_{10} M/M_{\mathrm{min}})/(\log_{10} M_{\mathrm{max}}/M_{\mathrm{min}})$')
            self.plot_loglog_histogram(fig, ax, data_by_time=data_by_time, color_scheme=csNumberDensity,
                Y_w=lambda data: (percentual_mass(data.m, data.M), weight(data.N)), colorbar_pad=0.15, *args, **kwargs)
            ax2 = ax.twinx()
            ax2.set_ylabel('tracer count')
            Xs = time[iLog]
            Yn = total_n[iLog]
            ax2.plot(np.log10(Xs/year/1.e+6), Yn, label='tracer count', color=csNumberDensity.embryo, linewidth=wEmb)
            #ax2.set_yscale('log')
            ax2.legend()

        def plot_swarm_mass_and_particle_count(self, fig, ax, *args, **kwargs):
            ax.set_title('swarm mass and particle count')
            ax.set_xlabel('time (Myr)')
            ax.set_ylabel('mass ($M_{\\oplus}$)')
            self.plot_loglog_histogram(fig, ax, data_by_time=data_by_time, color_scheme=csMass,
                Y_w=lambda data: (data.M, np.ones(len(data.M))), Yunit=Mea, colorbar_pad=0.15, *args, **kwargs)
            ax2 = ax.twinx()
            ax2.set_ylabel('particle count')
            Xs = time[iLog]
            YN = total_N[iLog]
            Yn = total_n[iLog]
            #log10Ymin = np.floor(np.log10(np.min([YN, Yn])))
            #log10Ymax = np.ceil(np.log10(np.max([YN, Yn])))
            ax2.plot(np.log10(Xs/year/1.e+6), YN, label='particle count', color=csNumberDensity.mean, linewidth=wMean)
            #ax2.plot(np.log10(Xs/year/1.e+6), Yn, label='tracer count', color=csNumberDensity.embryo, linewidth=wEmb)
            #ax2.set_aspect(np.log10(tLogP[-1]/tLogP[0]) / (log10Ymax - log10Ymin))
            ax2.set_yscale('log')
            ax2.legend()

        def plot_normalized_tracer_distribution(self, fig, ax, *args, **kwargs):
            def percentual_mass(m, M):
                mask = M != 0.
                min_m = np.min(m[mask])
                max_m = np.max(m[mask])
                if max_m == min_m:
                    max_m *= 2
                #return (m - min_m)/(max_m - min_m)
                return 10**(np.log10(np.where(mask, m, min_m)/min_m)/np.log10(max_m/min_m))
            def weight(N):
                return np.floor(np.minimum(N, 1.))
            ax.set_title('tracer distribution, normalized range')
            ax.set_xlabel('time (Myr)')
            ax.set_ylabel('relative logarithmic mass $(\log_{10} M/M_{\mathrm{min}})/(\log_{10} M_{\mathrm{max}}/M_{\mathrm{min}})$')
            self.plot_loglog_histogram(fig, ax, data_by_time=data_by_time, color_scheme=csNumberDensity,
                Y_w=lambda data: (percentual_mass(data.m, data.M), weight(data.N)), colorbar_pad=0.15, *args, **kwargs)
            ax2 = ax.twinx()
            ax2.set_ylabel('tracer count')
            Xs = time[iLog]
            Yn = total_n[iLog]
            ax2.plot(np.log10(Xs/year/1.e+6), Yn, label='tracer count', color=csNumberDensity.embryo, linewidth=wEmb)
            #ax2.set_yscale('log')
            ax2.legend()

        def plot_x(self, fig, ax, *args, **kwargs):
            # group_times
            #max_mass_indices1 = [data_at_time.m.values.argmax() for t, data_at_time in data_by_time]
            max_mass_indices = [data_at_time.index[data_at_time.m.values.argmax()] for t, data_at_time in data_by_time]
            ms = snapshot_data.m[max_mass_indices]
            ts = snapshot_data.t[max_mass_indices]
            Rs = snapshot_data.R[max_mass_indices]
            es = snapshot_data.e[max_mass_indices]
            def mChar(data):
                max_mass_idx = data.m.values.argmax()
                MMax = snapshot_data.M[max_mass_idx]
                mMax = snapshot_data.m[max_mass_idx]
                return (np.sum(data.M*data.m) - MMax*mMax)/np.sum(data.M)
            def m1_mChar(data):
                max_mass_idx = data.m.values.argmax()
                MMax = snapshot_data.M[max_mass_idx]
                mMax = snapshot_data.m[max_mass_idx]
                mChar = (np.sum(data.M*data.m) - MMax*mMax)/np.sum(data.M)
                return mMax/mChar
            hs = (ms/(3*MS))**(1/3)  # dimensionless Hill radius
            Rhs = sim_args.ring.r*hs  # Hill radius (cm)
            vhs = Rhs*ΩK  # Hill velocity (cm/s)
            #mChars = [mChar(data_at_time) for _, data_at_time in data_by_time]
            m1_mChars = [m1_mChar(data_at_time) for _, data_at_time in data_by_time]
            eMaxs = np.array([np.max(data_at_time.e) for _, data_at_time in data_by_time])
            veMax_vhs = eMaxs*vK/vhs
            min_v_index = veMax_vhs.argmin()
            ax.set_xscale('log')
            ax.set_yscale('log')
            #ax.plot(Rs/km, mChars, label='$m_*$', color='red')
            #ax.plot(Rs/km, es, label='$e_1$', color='yellow')
            #ax.plot(Rs/km, eMaxs, label='$e_{\\mathrm{max}}$', color='orange')
            ax.plot(Rs/km, m1_mChars, label='$m_1/m_*$', color='black')
            #ax.plot(Rs/km, eMaxs*vK, label='$v_e$', color='red')
            #ax.plot(Rs/km, vhs, label='$v_h$', color='green')
            ax.plot(Rs/km, veMax_vhs, label='$v_e/v_h$', color='black', linestyle='--')
            ax.axvline(x=Rs.values[min_v_index]/km, linestyle='dotted', label='$R_{\\mathrm{tr}}$')
            ax2 = ax.twinx()
            ax2.plot(Rs/km, ts/1.e+6/year, label='time (Myr)', color='gray')
            ax.set_xlabel('R (km)')
            ax2.set_ylabel('Myr')
            ax.legend(loc='lower left')
            ax2.legend(loc='upper right')
            #ax.grid(which='minor', axis='both', linestyle='--', color='lightgray')
            #maxMassIndices = 
            #idxMaxMass = data_at_time.m.values.argmax()
            #idx = data_at_time.index[idxMaxMass]
            #return data_at_time.id[idx], t, data_at_time.R[idx], data_at_time.m[idx]

        def plot_rR(self, fig, ax, t, vmin, vmax, *args, **kwargs):
            it = projection.find_nearest_indices(group_times, t)
            tR = group_times[it]
            ax.set_title('t = {:.2f} Myr'.format(tR/1.e+6/year))
            ax.set_xlabel('particle radius R (km)')
            ax.set_xscale('log')
            data = data_by_time.get_group(tR)
            mask = (data.M != 0) & (data.e != 0) & (data.sininc != 0)
            #data = data_by_time[it]
            Xs = data.R[mask]/km
            Ys = data.a[mask]/AU
            S0s = data.M[mask]
            Ss = S0s/np.sum(S0s)*1.e+4
            Cs = data.e[mask]
            sc = ax.scatter(Xs, Ys, c=Cs, s=Ss, cmap='coolwarm', *args, **kwargs, norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax))
            ax.legend(*sc.legend_elements(num=5), loc="upper left", title="eccentricity")
            #ax.legend(*sc.legend_elements(num=5), loc="upper left", title="eccentricity")

        #def plot_rR_video(self, fig, ax, t, vmin, vmax, *args, **kwargs)

        def _annotate(self, fig):
            if sim_args.simulation.nE > 0:
                embryo_str = '$M_{{\\mathrm{{E}}}}$ = {}, $R_{{\\mathrm{{E}},0}}$ = {}, $\\sqrt{{\\left<e_0^2\\right>}}$ = {:.4f}\n'.format(
                    params.embryo.M.serialize_value(sim_args.embryo.M), params.embryo.R.serialize_value(sim_args.embryo.R), sim_args.embryo.e)
            else:
                embryo_str = ''
            fig.suptitle(
                '{}\n'.format(sim_args.simulation.name) +
                'a = {}, Δa = {}, $M_*$ = {}, $M_{{\\mathrm{{Plt}}}}$ = {}, $R_{{\\mathrm{{Plt}},0}}$ = {}, $\\sqrt{{\\left<e_0^2\\right>}}$ = {:.4f}\n'.format(
                    params.ring.r.serialize_value(sim_args.ring.r), params.ring.Δr.serialize_value(sim_args.ring.Δr),
                    params.star.M.serialize_value(sim_args.star.M), params.planetesimal.M.serialize_value(sim_args.planetesimal.M),
                    params.planetesimal.R.serialize_value(sim_args.planetesimal.R), sim_args.planetesimal.e) +
                embryo_str +
                '$n_{{\\mathrm{{Plt}}}}$ = {} + {}, $n_{{\\mathrm{{E}}}}$ = {}, method: {}'.format(
                    sim_args.simulation.nPlt, sim_args.simulation.nPltR, sim_args.simulation.nE, sim_args.simulation.method) +
                '\nkernel: {}'.format(
                    sim_args.collisions.kernel) +
                (', discrete operators: {}'.format(
                    sim_args.collisions.Orm10_discrete_operators) if sim_args.collisions.kernel == 'Orm10' else '') +
                '\ncontinuous evolution: {}'.format(
                    sim_args.motion.stirring) +
                (', continuous operators: {}'.format(
                    sim_args.motion.Orm10_continuous_operators) if sim_args.motion.stirring == 'Orm10' else ''),
                fontsize=20)

        def plot_overview(self):
            # make space for 9 plots
            fig = plt.Figure(figsize=[24, 24])
            fig.subplots_adjust(wspace=0.1, hspace=0.15)
            self._annotate(fig)

            nplots = (3, 3)

#            ##print("- plotting bulk radii")
#            ##ax = fig.add_subplot(*nplots, 1)
#            ##self.plot_bulk_radius_to(fig, ax, nbin_factor=2.)
#
            print("- plotting masses")
            ax = fig.add_subplot(*nplots, 1)
            self.plot_mass_to(fig, ax, nbin_factor=2.)

#            print("- plotting collision rates")
#            ax = fig.add_subplot(*nplots, 2)
#            self.plot_collision_rate_to(fig, ax, nbin_factor=2.)
#
#            print("- plotting log collision rates")
#            ax = fig.add_subplot(*nplots, 3)
#            self.plot_logtime_collision_rate_to(fig, ax, nbin_factor=2.)
#
            print("- plotting eccentricities")
            ax = fig.add_subplot(*nplots, 4)
            self.plot_eccentricity_to(fig, ax, nbin_factor=2.)

            print("- plotting Hill velocities")
            ax = fig.add_subplot(*nplots, 5)
            self.plot_veccentricity_to(fig, ax, nbin_factor=2.)

            print("- plotting escape velocities")
            ax = fig.add_subplot(*nplots, 6)
            self.plot_vEsc_v_to(fig, ax, nbin_factor=2.)

#            ##print("- plotting rms velocities")
#            ##ax = fig.add_subplot(*nplots, 4)
#            ##self.plot_velocity_to(fig, ax, nbin_factor=2.)
#
            print("- plotting tracer distribution")
            ax = fig.add_subplot(*nplots, 7)
            self.plot_tracer_distribution(fig, ax, nbin_factor=2.)

#            #print("- plotting normalized tracer distribution")
#            #ax = fig.add_subplot(*nplots, 7)
#            #self.plot_normalized_tracer_distribution(fig, ax, nbin_factor=1.)
#
#            print("- plotting normalized tracer distribution")
#            ax = fig.add_subplot(*nplots, 8)
#            self.plot_normalized_tracer_distribution(fig, ax, nbin_factor=2.)
#
            print("- plotting swarm mass and particle count")
            ax = fig.add_subplot(*nplots, 9)
            self.plot_swarm_mass_and_particle_count(fig, ax, nbin_factor=2.)

#            #square_subplots(fig, *nplots)
#            ##print("- plotting normalized tracer distribution")
#            ##ax = fig.add_subplot(*nplots, 9)
#            ##self.plot_normalized_tracer_distribution(fig, ax, nbin_factor=4.)

            return fig

        def plot_xs(self):
            # make space for 9 plots
            fig = plt.Figure(figsize=[24, 24])
            fig.subplots_adjust(wspace=0.1, hspace=0.15)
            self._annotate(fig)

            nplots = (3, 3)

            print("- plotting x")
            ax = fig.add_subplot(*nplots, 1)
            self.plot_x(fig, ax, nbin_factor=2.)

            #print("- plotting radial mass distribution")
            #ax = fig.add_subplot(*nplots, 2)
            #self.plot_radial_mass_distribution(fig, ax, nbin_factor=2.)

            return fig

        def plot_rs(self):
            # make space for 3 plots
            fig = plt.Figure(figsize=[22, 10])
            fig.subplots_adjust(wspace=0., hspace=0., top=0.7)
            gs = fig.add_gridspec(nrows=1, ncols=3, hspace=0.15, wspace=0)
            ax1, ax2, ax3 = gs.subplots(sharex='col', sharey='row')
            ax1.set_ylabel('orbital radius r (AU)')
            ΔrScale = AnchoredVScaleBar(ax=ax1, size=sim_args.zones.ΔrMin/AU, label="radial zone width\n$\Delta r_{{\\mathrm{{min}}}}={:.3f}$ AU".format(sim_args.zones.ΔrMin/AU), loc='lower left', frameon=False,
                pad=0.6, sep=12, linekw=dict(color="k", linewidth=0.8))
            ax1.add_artist(ΔrScale)
            self._annotate(fig)

            mask = (snapshot_data.M != 0) & (snapshot_data.e != 0) & (snapshot_data.sininc != 0)
            emin = np.min(snapshot_data.e[mask])
            emax = np.max(snapshot_data.e[mask])
            vminScale = AnchoredVScaleBar(ax=ax1, size=sim_args.ring.r*emin/AU, label="scale length for\n$e_{{\\mathrm{{min}}}}={:.4f}$".format(emin), loc='lower left', frameon=False,
                pad=0.6, sep=12, linekw=dict(color="k", linewidth=0.8))
            vmaxScale = AnchoredVScaleBar(ax=ax1, size=sim_args.ring.r*emax/AU, label="scale length for\n$e_{{\\mathrm{{max}}}}={:.4f}$".format(emax), loc='lower left', frameon=False,
                pad=0.6, sep=12, linekw=dict(color="k", linewidth=0.8))
            ax2.add_artist(vminScale)
            ax3.add_artist(vmaxScale)
            
            print("- plotting rR")
            self.plot_rR(fig, ax1, t=1.3e4*year, vmin=emin, vmax=emax)
            #self.plot_rR(fig, ax1, t=1.e4*year, vmin=emin, vmax=emax)
            self.plot_rR(fig, ax2, t=3.8e4*year, vmin=emin, vmax=emax)
            #self.plot_rR(fig, ax2, t=1.e5*year, vmin=emin, vmax=emax)
            self.plot_rR(fig, ax3, t=1.8e5*year, vmin=emin, vmax=emax)
            #self.plot_rR(fig, ax3, t=1.e+6*year, vmin=emin, vmax=emax)

            return fig

        def render_movie_rR(self, scale='linear'):
            from matplotlib.lines import Line2D

            if scale == 'log':
                times = tLogP
            elif scale == 'linear':
                times = tLinearP
            else:
                raise RuntimeError("Unknown timescale '{}'".format(scale))

            mask1 = snapshot_data.M != 0
            mask2 = (snapshot_data.M != 0) & (snapshot_data.e != 0) & (snapshot_data.sininc != 0)
            rmin0 = np.min(snapshot_data.a[mask1])
            rmax0 = np.max(snapshot_data.a[mask1])
            rmin = rmin0 - 0.05*(rmax0 - rmin0)
            rmax = rmax0 + 0.05*(rmax0 - rmin0)
            Rmin0 = np.min(snapshot_data.R[mask1])
            Rmax0 = np.max(snapshot_data.R[mask1])
            Rmin = Rmin0/(Rmax0/Rmin0)**0.05
            Rmax = Rmax0*(Rmax0/Rmin0)**0.05
            cmap = plt.cm.coolwarm
            emin = np.min(snapshot_data.e[mask2])
            emax = np.max(snapshot_data.e[mask2])
            legend_es = np.logspace(np.log10(emin), np.log10(emax), num=5, endpoint=True, base=10)
            legend_elements = [
                Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=cmap(np.log(e/emin)/np.log(emax/emin)), markersize=12)
                for e in legend_es
            ]
            legend_labels = [
                'e = {}'.format(sig(e, digits=2))
                for e in legend_es
            ]

            print("r: {}..{} AU\nR: {}..{} km\ne: {}..{}".format(rmin/AU, rmax/AU, Rmin/km, Rmax/km, emin, emax))
            fig = plt.Figure(figsize=[11, 9])
            #fig.subplots_adjust(wspace=0., hspace=0., top=1.)
            #gs = fig.add_gridspec(nrows=1, ncols=1, hspace=0.15, wspace=0)
            gs = fig.add_gridspec(nrows=1, ncols=1)
            ax1 = gs.subplots(sharex='col', sharey='row')
            ax1.set_xlabel('particle radius R (km)')
            ax1.set_ylabel('orbital radius r (AU)')
            ax1.legend(legend_elements, legend_labels, loc='upper left')
            sc = None

            def data(tR):
                data = data_by_time.get_group(tR)
                mask = data.M != 0
                Xs = data.R[mask]/km
                Ys = data.a[mask]/AU
                S0s = data.M[mask]
                Ss = S0s/np.sum(S0s)*1.e+4
                Cs = np.maximum(1.e-6, data.e[mask])
                return Xs, Ys, Ss, Cs

            def init():
                ax1.set_xscale('log')
                ax1.set_xlim(Rmin/km, Rmax/km)
                ax1.set_ylim(rmin/AU, rmax/AU)
                nonlocal sc
                #it = projection.find_nearest_indices(group_times, tLogP[0])
                #tR = group_times[it]
                Xs, Ys, Ss, Cs = data(times[0])
                sc = ax1.scatter(Xs, Ys, c=Cs, s=Ss, cmap=cmap, norm=matplotlib.colors.LogNorm(vmin=emin, vmax=emax))
                return sc,

            def update(t):
                #it = projection.find_nearest_indices(group_times, t)
                #tR = group_times[it]
                if scale == 'log':
                    print("  - update(t = {:g} Myr)".format(t/1.e+6/year))
                    ax1.set_title('t = {:g} Myr'.format(t/1.e+6/year))
                else:
                    print("  - update(t = {:.2f} Myr)".format(t/1.e+6/year))
                    ax1.set_title('t = {:.2f} Myr'.format(t/1.e+6/year))
                Xs, Ys, Ss, Cs = data(t)
                sc.set_offsets(np.transpose(np.array([Xs, Ys])))
                sc.set_sizes(Ss)
                sc.set_array(Cs)
                return sc,
                #sc = ax.scatter(Xs, Ys, c=Cs, s=Ss, cmap=cmap, *args, **kwargs, norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax))
                #ax.legend(*sc.legend_elements(num=5), loc="upper left", title="eccentricity")

            ani = matplotlib.animation.FuncAnimation(fig, update, frames=times[1:], init_func=init, blit=True)
            return ani

    return Plots()

def make_histogram_plots(sim_args, sim_data: SimulationData):

    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib import animation, ticker
    from const.cgs import year

    @plt.FuncFormatter
    def fake_log(value, tick_position):
        return '$10^{{{}}}$'.format(int(value))

    # load histogram data
    hists = sim_data.histograms()
    eSq_stir_hists = hists['de²_dt']
    interaction_mode_hists = hists['stirring interaction mode']
    log_R_min, log_R_max = hists['de²_dt params']
    ts, ts_linear, ts_log = sim_data.timesteps()['unified'], sim_data.timesteps()['linear'], sim_data.timesteps()['log']
    linear_mask = projection.find_nearest_indices(ts, ts_linear)
    log_mask = projection.find_nearest_indices(ts, ts_log)

    class Plots:
        def render_movie_stir_map(self, scale):
            if scale == 'log':
                mask = log_mask
            elif scale == 'linear':
                mask = linear_mask
            else:
                raise RuntimeError("Unknown timescale '{}'".format(scale))

            #masked_hists = [eSq_stir_hists[j] for j in mask]

            cmap = plt.cm.coolwarm
            cmap2 = plt.get_cmap('Set1')
            #vmin = np.min([np.min(masked_hist[masked_hist > 0.], initial=np.inf) for masked_hist in masked_hists])
            #vmin = np.min([np.min(masked_hist) for masked_hist in masked_hists])
            #vmax = np.max([np.max(masked_hist) for masked_hist in masked_hists])
            vmin = np.min(eSq_stir_hists[~np.isnan(eSq_stir_hists)])*1.e+6*year
            vmax = np.max(eSq_stir_hists[~np.isnan(eSq_stir_hists)])*1.e+6*year
            vmin2 = 0.
            vmax2 = 2.

            norm = matplotlib.colors.SymLogNorm(
                linthresh=1.e-10,  # Myr⁻¹
                linscale=1,
                base=10, vmin=vmin, vmax=vmax)

            #legend_vs = np.logspace(np.log10(vmin), np.log10(vmax), num=5, endpoint=True, base=10)
            legend_cvs = np.linspace(0, 1, num=5, endpoint=True)
            legend_elements = [
                Line2D([0], [0], marker='s', color='w', markerfacecolor=cmap(cv), markersize=12)
                for cv in legend_cvs
            ]
            legend_labels = [
                #'$\\mathrm{{d}} \\left\\langle e^2\\right\\rangle/\\mathrm{{d}} t =$ {:g} $\\mathrm{{Myr}}^{{-1}}$'.format(v*1.e+6*year)
                '$\\mathrm{{d}} \\left\\langle e^2\\right\\rangle/\\mathrm{{d}} t =$ {:g} $\\mathrm{{Myr}}^{{-1}}$'.format(norm.inverse(np.array([cv]))[0])
                for cv in legend_cvs
            ]

            legend2_elements = [
                Line2D([0], [0], marker='s', color='w', markerfacecolor=cmap2(v/2), markersize=12)
                for v in range(0, 3)
            ]
            legend2_labels = [
                'dispersion-dominated regime',
                'shear-dominated regime',
                'superescape regime'
            ]

            fig = plt.Figure(figsize=[22, 9])
            #fig.subplots_adjust(wspace=0., hspace=0., top=1.)
            fig.subplots_adjust(wspace=0., hspace=0.)
            #gs = fig.add_gridspec(nrows=1, ncols=1, hspace=0.15, wspace=0)
            gs = fig.add_gridspec(nrows=1, ncols=2, wspace=0)
            ax1,ax2 = gs.subplots(sharex='col', sharey='row')
            ax1.set_xlabel('radius $R$ of stirred tracer (m)')
            ax2.set_xlabel('radius $R$ of stirred tracer (m)')
            ax1.set_ylabel('radius $R$ of stirring swarm (m)')
            ax1.xaxis.set_major_formatter(fake_log)
            ax1.xaxis.set_major_locator(ticker.MultipleLocator(1.))
            ax2.xaxis.set_major_formatter(fake_log)
            ax2.xaxis.set_major_locator(ticker.MultipleLocator(1.))
            ax1.yaxis.set_major_formatter(fake_log)
            ax1.yaxis.set_major_locator(ticker.MultipleLocator(1.))
            ax1.legend(legend_elements, legend_labels, loc='lower left')
            ax2.legend(legend2_elements, legend2_labels, loc='lower left')
            sc = None

            def image(j):
                #data = eSq_stir_hists[j].copy()
                #data[data <= 0.] = np.nan
                #return data
                return eSq_stir_hists[j]
            def image2(j):
                return interaction_mode_hists[j]

            im = ax1.imshow(image(mask[0])*1.e+6*year, interpolation='nearest', origin='lower',
                            extent=[log_R_min - 2, log_R_max - 2, log_R_min - 2, log_R_max - 2],
                            cmap=cmap, norm=norm)
            im2 = ax2.imshow(image2(mask[0]), interpolation='nearest', origin='lower',
                            extent=[log_R_min - 2, log_R_max - 2, log_R_min - 2, log_R_max - 2],
                            cmap=cmap2, norm=matplotlib.colors.Normalize(vmin=vmin2, vmax=vmax2))

            def init():
                #im.set_data(eSq_stir_hists[0])
                return im,im2,

            def update(i):
                j = mask[i]
                t = ts[j]
                if scale == 'log':
                    print("  - update(t = {:g} Myr)".format(t/1.e+6/year))
                    ax1.set_title('t = {:g} Myr'.format(t/1.e+6/year))
                else:
                    print("  - update(t = {:.2f} Myr)".format(t/1.e+6/year))
                    ax1.set_title('t = {:.2f} Myr'.format(t/1.e+6/year))
                #a = im.get_array()
                #a[:] = eSq_stir_hists[i]
                im.set_array(image(j))
                im2.set_array(image2(j))
                return im,im2,

            ani = animation.FuncAnimation(fig, update, frames=len(mask), init_func=init, blit=True)
            return ani

            #if sim_args.simulation.nPlt > 1:
            #    data_by_tLog = [Plots.positive_finite(*Y_w(data_by_time.get_group(t))) for t in tLogP]
            #    hist, bins = projection.project_log_histograms(data_by_tLog, Ny, nbin_factor=nbin_factor)
            #    Y0, Y1 = bins[0], bins[-1]
            #    print('min(hist): {}, max(hist): {}'.format(np.min(hist), np.max(hist)))
            #    im = ax.imshow(hist, interpolation='nearest', origin='lower',
            #                   extent=[np.log10(tLogP[0]/year/1.e+6), np.log10(tLogP[-1]/year/1.e+6), np.log10(Y0/Yunit), np.log10(Y1/Yunit)],
            #                   cmap=color_scheme.hist_cmap, norm=matplotlib.colors.LogNorm())
            #    cb = fig.colorbar(im, ax=ax, pad=colorbar_pad)
            #    ax.set_aspect(np.log10(tLogP[-1]/tLogP[0]) / np.log10(Y1/Y0))
            #ax.xaxis.set_major_formatter(fake_log)
            #ax.xaxis.set_major_locator(ticker.MultipleLocator(1.))
            #ax.xaxis.set_minor_formatter(ticker.NullFormatter())
            #ax.xaxis.set_minor_locator(ticker.FixedLocator(minor_ticks))
            #ax.yaxis.set_major_formatter(fake_log)
            #ax.yaxis.set_major_locator(ticker.MultipleLocator(1.))
            #ax.yaxis.set_minor_formatter(ticker.NullFormatter())
            #ax.yaxis.set_minor_locator(ticker.FixedLocator(minor_ticks))
            #ax.grid(which='major', axis='both')
            #ax.grid(which='minor', axis='both', linestyle='--', color='lightgray')
            #Xs = time[iLog]
            #if Ymean is not None:
            #    Ys = Ymean[iLog]
            #    ax.plot(np.log10(Xs[Ys > 0.]/year/1.e+6), np.log10(Ys[Ys > 0.]/Yunit), label=mean_label, color=color_scheme.mean, linewidth=wMean)
            #if sim_args.simulation.nE >= 1 and YmeanE is not None:
            #    Ys = YmeanE[iLog]
            #    ax.plot(np.log10(Xs[Ys > 0.]/year/1.e+6), np.log10(Ys[Ys > 0.]/Yunit), label=embryo_label, color=color_scheme.embryo, linewidth=wEmb)
            ##ax.set_ylim(bottom=np.log10(Y0/Yunit), top=np.log10(Y1/Yunit))
            #if Ymean is not None or (sim_args.simulation.nE >= 1 and YmeanE is not None):
            #    ax.legend()


            #def data(tR):
            #    tdata = 
            #    data = data_by_time.get_group(tR)
            #    mask = data.M != 0
            #    Xs = data.R[mask]/km
            #    Ys = data.a[mask]/AU
            #    S0s = data.M[mask]
            #    Ss = S0s/np.sum(S0s)*1.e+4
            #    Cs = np.sqrt(np.maximum(1.e-12, data.eSq[mask]))
            #    return Xs, Ys, Ss, Cs
#
            #def init():
            #    ax1.set_xscale('log')
            #    ax1.set_xlim(Rmin/km, Rmax/km)
            #    ax1.set_ylim(rmin/AU, rmax/AU)
            #    nonlocal sc
            #    #it = projection.find_nearest_indices(group_times, tLogP[0])
            #    #tR = group_times[it]
            #    Xs, Ys, Ss, Cs = data(times[0])
            #    sc = ax1.scatter(Xs, Ys, c=Cs, s=Ss, cmap=cmap, norm=matplotlib.colors.LogNorm(vmin=emin, vmax=emax))
            #    return sc,
#
            #def update(t):
            #    #it = projection.find_nearest_indices(group_times, t)
            #    #tR = group_times[it]
            #    print("  - update(t = {:.2f} Myr)".format(t/1.e+6/year))
            #    ax1.set_title('t = {:.2f} Myr'.format(t/1.e+6/year))
            #    Xs, Ys, Ss, Cs = data(t)
            #    sc.set_offsets(np.transpose(np.array([Xs, Ys])))
            #    sc.set_sizes(Ss)
            #    sc.set_array(Cs)
            #    return sc,
            #    #sc = ax.scatter(Xs, Ys, c=Cs, s=Ss, cmap=cmap, *args, **kwargs, norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax))
            #    #ax.legend(*sc.legend_elements(num=5), loc="upper left", title="eccentricity")
#
            #ani = matplotlib.animation.FuncAnimation(fig, update, frames=times[1:], init_func=init, blit=True)
            #anim = animation.FuncAnimation(fig, animate, init_func=init,
            #                   frames=200, interval=20, blit=True)
            #return ani

    return Plots()


# Example: write single plot
#plots = simulation.make_plots(...)
#filename = simulation.plot_filename(outdir, datafilename, 'bulk radius')
#print("- plotting bulk radii to '{}'".format(filename))
#fig,ax = plt.subplots(1, 1, figsize=(6, 6))
#ax = fig.add_subplot(*nplots, 1)
#plots.plot_bulk_radius_to(ax)
#fig.savefig(filename)


def analyze_snapshots(sim_args, analysis_args, sim_data):

    from const.cgs import MS, GG

    ΩK = np.sqrt(GG*MS/sim_args.ring.r**3)  # Kepler orbital velocity (1/s)
    vK = sim_args.ring.r*ΩK  # Kepler velocity of planetesimal (cm/s)

    # load snapshot data
    snapshot_data = sim_data.snapshots()

    def duration_to_mass(m):
        """Determines how long it took for at least one of the tracers to exceed the given mass."""
        exc_data = snapshot_data[snapshot_data.m >= m]
        if len(exc_data.index) > 0:
            idx = exc_data.index[0]
            return exc_data.id[idx], exc_data.t[idx]
        return -1, np.nan

    def mass_until_time(t):
        """Determines the maximum mass obtained at the given time."""
        exc_data = snapshot_data[snapshot_data.t >= t]
        if len(exc_data) == 0:
            return -1, np.nan, np.nan, np.nan
        t = exc_data.t.values.min()
        data_at_time = snapshot_data[snapshot_data.t == t]
        idxMaxMass = data_at_time.m.values.argmax()
        idx = data_at_time.index[idxMaxMass]
        return data_at_time.id[idx], t, data_at_time.R[idx], data_at_time.m[idx]

    def max_mass():
        """Returns the largest planetesimal mass obtained."""
        idxMax = snapshot_data.m.values.argmax()
        return snapshot_data.id[idxMax], snapshot_data.R[idxMax], snapshot_data.m[idxMax]

    def min_Δve_vh():
        """Returns the smallest value of  Δvₑ/vₕ ."""
        h = (snapshot_data.m/(3*MS))**(1/3)  # dimensionless Hill radius
        Rh = sim_args.ring.r*h  # Hill radius (cm)
        vh = Rh*ΩK  # Hill velocity (cm/s)
        Δve_vh = snapshot_data.e*(vK/vh)
        return np.min(Δve_vh)

    def max_vEsc_v():
        """Returns the largest value of  vEsc/v ."""
        vEsc = np.sqrt(2*GG*snapshot_data.m/snapshot_data.R)  # escape velocity from planetesimal (cm/s)
        v = vK*np.sqrt(snapshot_data.e**2 + snapshot_data.sininc**2)  # rms velocities (cm/s)
        vEsc_v = vEsc/v
        return np.max(vEsc_v)

    def excess_count():
        def excess_count_for(data):
            Mavg0 = np.average(data.M)
            return np.sum(np.maximum(np.floor(data.M/Mavg0), 1) - 1)
        data_by_time = snapshot_data \
            .groupby(snapshot_data.t, as_index=False)  # TODO: why as_index?
        group_times = np.fromiter(data_by_time.indices.keys(), dtype=float)
        #data0 = data_by_time.get_group(group_times[0])
        #Mavg0 = np.average(data0.M)
        excess_counts = [excess_count_for(data_by_time.get_group(t)) for t in group_times]
        print(excess_counts)
        return np.max(excess_counts)

    result_args = analysis_results.init_defaults()
    id_mMax, result_args.results.planetesimal.RMax, result_args.results.planetesimal.mMax = max_mass()
    id_tMin, result_args.results.planetesimal.tMin_mTh = duration_to_mass(analysis_args.analysis.mTh)
    id, tT, result_args.results.planetesimal.R_mMax_tTh, result_args.results.planetesimal.mMax_tTh = mass_until_time(analysis_args.analysis.tTh)
    result_args.results.planetesimal.min_Δv_vh = min_Δve_vh()
    result_args.results.planetesimal.max_vEsc_v = max_vEsc_v()
    result_args.results.planetesimal.nExcess = excess_count()
    return result_args

def make_inspection_plots(sim_args):

    from matplotlib import cm
    import matplotlib.pyplot as plt
    from const.cgs import year, km, AU, Mea, MS, GG

    import rpmc

    class InspectionPlots:
        def plot_gas_profile(self, ax=None):
            gas_params = rpmc.GasParams(args=sim_args)
            gas_profile_params = rpmc.RadialGasProfileParams(args=sim_args)
            planetary_gap_params = rpmc.PlanetaryGapParams(args=sim_args)

            r_min, r_max = sim_args.ring.r - sim_args.ring.Δr*1.5, sim_args.ring.r + sim_args.ring.Δr*1.5
            r_min = min(r_min, planetary_gap_params.rp - planetary_gap_params.Rh*3)
            r_max = max(r_max, planetary_gap_params.rp + planetary_gap_params.Rh*3)

            rs = np.linspace(start=r_min, stop=r_max, num=256)

            Σs = rpmc.computeGasSurfaceDensity(gas_profile_params, planetary_gap_params, r=rs)
            dΣ_drs = rpmc.computeGasSurfaceDensityGradient(gas_profile_params, planetary_gap_params, r=rs)

            cmap = cm.rainbow
            colors = list(cmap(np.linspace(0., 1., num=4)))
            if ax is None:
                fig, ax = plt.subplots(1, 1, sharex=False, sharey=False, figsize=[9, 6])
                #fig.tight_layout(h_pad=10, w_pad=10)
            else:
                fig = ax.get_figure()
            ax.plot(rs/AU, Σs, label=R'$\Sigma$', color=colors[0])
            ax.set_xlabel("r (AU)")
            ax.set_ylabel("Σ (g/cm²)")
            ax.set_title('radial gas density profile')
            #ax.set_xscale('log')
            ax.set_yscale('log')
            axb = ax.twinx()
            axb.set_ylabel(R'$\mathrm{d\,log\,}\Sigma/\mathrm{d\,log\,}r$')
            axb.plot(rs/AU, rs/Σs*dΣ_drs, label=R'$\mathrm{d\,log\,}\Sigma/\mathrm{d\,log\,}r$', color=colors[1])
            ax.grid(axis='x', linestyle='--')
            ax.grid(axis='y')
            axb.grid(axis='y', linestyle='--')
            #ax.grid(which='minor', axis='x')
            #ax.grid(which='minor', axis='y', linestyle='--', color='lightgray')
            ax.legend(loc="lower left")
            axb.legend(loc="upper right")
            return fig, ax

        def plot_gas_drift_velocity(self, ax=None):
            num_iterations = 3

            gas_params = rpmc.GasParams(args=sim_args)
            gas_profile_params = rpmc.RadialGasProfileParams(args=sim_args)
            planetary_gap_params = rpmc.PlanetaryGapParams(args=sim_args)

            r_min, r_max = sim_args.ring.r - sim_args.ring.Δr*1.5, sim_args.ring.r + sim_args.ring.Δr*1.5
            r_min = min(r_min, planetary_gap_params.rp - planetary_gap_params.Rh*3)
            r_max = max(r_max, planetary_gap_params.rp + planetary_gap_params.Rh*3)

            rs = np.linspace(start=r_min, stop=r_max, num=256)

            ρ = sim_args.planetesimal.ρ
            #Rs = np.array([1.e-2, 1.e-1, 1.e+0, 1.e+1, 5.e+1, 1.e+2, 2.e+2, 1.e+3, 1.e+4])  # (cm)
            #Rs = np.array([1.e+0, 2.e+0, 3.e+0, 5.e+0, 1.e+1, 1.e+2, 1.e+3, 1.e+4])  # (cm)
            Rs = np.array([1.e+0, 2.e+0, 3.e+0, 5.e+0, 1.e+1, 1.e+2])  # (cm)
            ms = 4/3*π*ρ*Rs**3

            cmap = cm.rainbow
            colors = list(cmap(np.linspace(0., 1., num=len(Rs))))
            if ax is None:
                fig, ax = plt.subplots(1, 1, sharex=False, sharey=False, figsize=[9, 6])
                #fig.tight_layout(h_pad=10, w_pad=10)
            else:
                fig = ax.get_figure()
            ax.set_title('radial equilibrium gas drift velocity')
            ax.set_xlabel("r (AU)")
            ax.set_ylabel("radial equilibrium drift velocity (cm/s)")
            ax.set_yscale('symlog', linthresh=1.e+1, linscale=0.3)  # (cm/s)
            ax.grid(axis='both', linestyle='--')
            ax.grid(which='minor', axis='y', linestyle='--', color='lightgray')
            if planetary_gap_params.Rh != 0:
                ax.axvline(x=planetary_gap_params.rp/AU, color='black', label='planet')
                ax.axvspan(xmin=(planetary_gap_params.rp - planetary_gap_params.Rh)/AU, xmax=(planetary_gap_params.rp + planetary_gap_params.Rh)/AU,
                    color='gray', alpha=0.25, label='Hill sphere of planet')
            have_dust_trap = False
            results = [
                rpmc.findParticleEquilibriumDrift(gas_params, gas_profile_params, planetary_gap_params,
                    ρ=ρ, m=ms[i], r=rs, max_num_iterations=num_iterations)
                for i in range(len(Rs))
            ]
            for i, R in enumerate(Rs):
                St, vx, vy = results[i]['St'], results[i]['vx'], results[i]['vy']
                dvx_dr = np.diff(vx)
                roots = np.argwhere((np.sign(vx[:-1]) != np.sign(vx[1:])) & (dvx_dr < 0))
                for ir in roots:
                    ax.axvline(x=rs[ir]/AU, linestyle='--', color=colors[i], alpha=0.4, label='dust trap' if not have_dust_trap else None)
                    have_dust_trap = True
            for i, R in enumerate(Rs):
                St, vx, vy = results[i]['St'], results[i]['vx'], results[i]['vy']
                ax.plot(rs/AU, vx, label=R'R={} m'.format(Rs[i]/1.e+2), color=colors[i])
            ax.legend()
            return fig, ax

    return InspectionPlots()

def make_snapshot_plots_2(sim_args, sim_data: SimulationData):

    import matplotlib
    import matplotlib.offsetbox
    from matplotlib import cm
    import matplotlib.ticker as ticker
    import matplotlib.pyplot as plt
    from const.cgs import year, km, AU, Mea, MS, GG


    class AnchoredVScaleBar(matplotlib.offsetbox.AnchoredOffsetbox):
        """ size: length of bar in data units
            extent : height of bar ends in axes units """
        # taken from https://stackoverflow.com/a/43343934
        def __init__(self, size=1, extent=0.03, label="", loc=2, ax=None,
                    pad=0.4, borderpad=0.5, ppad=0, sep=2, prop=None, 
                    frameon=True, align='center', linekw={}, textkw={}, **kwargs):
            from matplotlib.lines import Line2D
            if not ax:
                ax = plt.gca()
            trans = ax.get_yaxis_transform()
            size_bar = matplotlib.offsetbox.AuxTransformBox(trans)
            line = Line2D([0,0],[0,size], **linekw)
            vline1 = Line2D([-extent/2,extent/2],[0,0], **linekw)
            vline2 = Line2D([-extent/2,extent/2],[size,size], **linekw)
            size_bar.add_artist(line)
            size_bar.add_artist(vline1)
            size_bar.add_artist(vline2)
            txt = matplotlib.offsetbox.TextArea(label, minimumdescent=False, textprops=dict(rotation='vertical', horizontalalignment=align, **textkw))
            self.hpac = matplotlib.offsetbox.HPacker(children=[size_bar,txt],
                                    align=align, pad=ppad, sep=sep)
            matplotlib.offsetbox.AnchoredOffsetbox.__init__(self, loc, pad=pad,
                    borderpad=borderpad, child=self.hpac, prop=prop, frameon=frameon,
                    **kwargs)

    @plt.FuncFormatter
    def fake_log(value, tick_position):
        return '$10^{{{}}}$'.format(int(value))

    def positive_finite(Y, w):
        mask = (Y > 0.) & (w > 0.) & ~np.isinf(Y)
        return Y[mask], w[mask]

    class SnapshotPlots2:
        def plot_masses_and_eccentricities(self, axes=None):
            if axes is None:
                fig, axes = plt.subplots(1, 2, sharex=False, sharey=False, figsize=[15, 6])
                ax1, ax2 = axes
                #fig.tight_layout(h_pad=10, w_pad=10)
            else:
                ax1, ax2 = axes
                fig = ax1.get_figure()

            Ny = 128

            # load snapshot data
            snapshot_data = sim_data.snapshots()
            tLinear, tLog = sim_data.timesteps()['linear'], sim_data.timesteps()['log']

            # group by time
            data_by_time = snapshot_data \
                .groupby(snapshot_data.t, as_index=False)  # TODO: why as_index?

            # Due to loss of precision in serialization of floating-point data, we have to look for the nearest times rather than doing
            # a straight lookup with `get_group()`.
            group_times = np.fromiter(data_by_time.indices.keys(), dtype=float)
            iLinear = projection.find_nearest_indices(group_times, tLinear)
            iLog = projection.find_nearest_indices(group_times, tLog)
            tLinearP = group_times[iLinear]
            tLogP = group_times[iLog]

            minor_ticks = np.array([i + np.log10(k) for k in range(2, 10) for i in range(-20, 15)])

            have_planet = sim_args.dust_trap.mode == 'synchronized-pressure-bump'
            nP = 1 if have_planet else 0

            # Plot masses
            cmap = plt.get_cmap('Blues')
            m_unit = Mea
            R_unit = km
            ρ = sim_args.planetesimal.ρ
            def ldm2ldR(ldm_m0):
                m_m0 = 10**ldm_m0
                m = m_m0*m_unit
                R = (m/(4/3*π*ρ))**(1/3)
                R_R0 = R/R_unit
                return np.log10(R_R0)
            def ldR2ldm(ldR_R0):
                R_R0 = 10**ldR_R0
                R = R_R0*R_unit
                m = 4/3*π*ρ*R**3
                m_m0 = m/m_unit
                return np.log10(m_m0)
            if sim_args.simulation.nPlt > 1:
                offsPlt = nP + sim_args.simulation.nE
                Y_w = lambda data: (data.m[data.id >= offsPlt], data.M[data.id >= offsPlt])
                data_by_tLog = [positive_finite(*Y_w(data_by_time.get_group(t))) for t in tLogP]
                hist, bins = projection.project_log_histograms(data_by_tLog, Ny, nbin_factor=2.)
                Y0, Y1 = bins[0], bins[-1]
                im = ax1.imshow(hist, interpolation='nearest', origin='lower',
                               extent=[np.log10(tLogP[0]/year/1.e+6), np.log10(tLogP[-1]/year/1.e+6), np.log10(Y0/m_unit), np.log10(Y1/m_unit)],
                               cmap=cmap, norm=matplotlib.colors.LogNorm())
                #cb = fig.colorbar(im, ax=ax1, pad=0.1)
                ax1.set_aspect(np.log10(tLogP[-1]/tLogP[0]) / np.log10(Y1/Y0))
            ax1.xaxis.set_major_formatter(fake_log)
            ax1.xaxis.set_major_locator(ticker.MultipleLocator(1.))
            ax1.xaxis.set_minor_formatter(ticker.NullFormatter())
            ax1.xaxis.set_minor_locator(ticker.FixedLocator(minor_ticks))
            ax1.yaxis.set_major_formatter(fake_log)
            ax1.yaxis.set_major_locator(ticker.MultipleLocator(1.))
            ax1b = ax1.secondary_yaxis('right', functions=(ldm2ldR, ldR2ldm))
            ax1b.yaxis.set_major_formatter(fake_log)
            ax1b.yaxis.set_major_locator(ticker.MultipleLocator(1.))
            ax1.grid(which='major', axis='both')
            ax1.grid(which='minor', axis='x', linestyle='--', color='lightgray')
            ax1.set_xlabel('time (Myr)')
            ax1.set_ylabel(R'particle mass ($M_\oplus$)')
            ax1b.set_ylabel('particle radius (km)')
            ax1.set_title('mass distribution')
            #Xs = time[iLog]
            Xs = tLogP
            if have_planet:
                mPs = snapshot_data.m[snapshot_data.id == 0].to_numpy()[iLog]
                ax1.plot(np.log10(Xs/year/1.e+6), np.log10(mPs/m_unit), label='planet', color='tab:orange', linewidth=2.5)
            for i in range(nP, nP + sim_args.simulation.nE):
                mEs = snapshot_data.m[snapshot_data.id == i].to_numpy()[iLog]
                ax1.plot(np.log10(Xs/year/1.e+6), np.log10(mEs/m_unit), label='ovum' if i == nP else None, color='tab:brown', linewidth=1.5)
            if have_planet or sim_args.simulation.nE > 0:
                ax1.legend()

            # Plot eccentricities
            e_unit = 1.
            cmap = plt.get_cmap('Greys')
            if sim_args.simulation.nPlt > 1:
                offsPlt = nP + sim_args.simulation.nE
                Y_w = lambda data: (data.e[data.id >= offsPlt], data.M[data.id >= offsPlt])
                data_by_tLog = [positive_finite(*Y_w(data_by_time.get_group(t))) for t in tLogP]
                hist, bins = projection.project_log_histograms(data_by_tLog, Ny, nbin_factor=2.)
                Y0, Y1 = bins[0], bins[-1]
                im = ax2.imshow(hist, interpolation='nearest', origin='lower',
                               extent=[np.log10(tLogP[0]/year/1.e+6), np.log10(tLogP[-1]/year/1.e+6), np.log10(Y0/e_unit), np.log10(Y1/e_unit)],
                               cmap=cmap, norm=matplotlib.colors.LogNorm())
                #cb = fig.colorbar(im, ax=ax2, pad=0.1)
                ax2.set_aspect(np.log10(tLogP[-1]/tLogP[0]) / np.log10(Y1/Y0))
            ax2.xaxis.set_major_formatter(fake_log)
            ax2.xaxis.set_major_locator(ticker.MultipleLocator(1.))
            ax2.xaxis.set_minor_formatter(ticker.NullFormatter())
            ax2.xaxis.set_minor_locator(ticker.FixedLocator(minor_ticks))
            ax2.yaxis.set_major_formatter(fake_log)
            ax2.yaxis.set_major_locator(ticker.MultipleLocator(1.))
            ax2.yaxis.set_minor_formatter(ticker.NullFormatter())
            ax2.yaxis.set_minor_locator(ticker.FixedLocator(minor_ticks))
            ax2.grid(which='major', axis='both')
            ax2.grid(which='minor', axis='both', linestyle='--', color='lightgray')
            ax2.set_xlabel('time (Myr)')
            ax2.set_ylabel(R'(rms-)eccentricity')
            ax2.set_title('mass-weighted eccentricities')
            #Xs = time[iLog]
            Xs = tLogP
            if have_planet:
                ePs = snapshot_data.e[snapshot_data.id == 0].to_numpy()[iLog]
                mask = ePs > 0
                ax2.plot(np.log10(Xs[mask]/year/1.e+6), np.log10(ePs[mask]/e_unit), label='planet', color='tab:orange', linewidth=2.5)
            for i in range(nP, nP + sim_args.simulation.nE):
                eEs = snapshot_data.e[snapshot_data.id == i].to_numpy()[iLog]
                mask = eEs > 0
                ax2.plot(np.log10(Xs[mask]/year/1.e+6), np.log10(eEs[mask]/e_unit), label='ovum' if i == nP else None, color='tab:brown', linewidth=1.5)
            if have_planet or sim_args.simulation.nE > 0:
                ax2.legend()

            return fig, axes
    return SnapshotPlots2()
