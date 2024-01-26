# Source code repository for Beutel et al., 'Efficient simulation of stochastic interactions among representative Monte Carlo particles'

This repository contains the code used to generate the results used in the publication
'Efficient simulation of stochastic interactions among representative Monte Carlo particles'
by M. Beutel, C. P. Dullemond, & R. Strzodka, accepted for publication in Astronomy & Astrophysics,
henceforth referred to as [Beutel et al. (in press)](#beutel_efficient_2024).

**Contents**  
- [Introduction](#introduction)
- [Build dependencies](#build-dependencies)
- [Usage](#usage)
- [Reproducing the figures from our publication](#reproducing-the-figures-from-our-publication)
- [Troubleshooting](#troubleshooting)
- [References](#references)


## Introduction

Our publication introduces an efficient computational scheme for representative stochastic methods,
dubbed the 'bucketing scheme'. This repository contains two reference implementations of the
extended Representative Particle Monte Carlo method (see [Beutel & Dullemond (2023)](#beutel_improved_2023)):
once with the traditional scheme, which uses discrete inverse transform sampling,
and once with the bucketing scheme, which maintains a dynamic grouping ('buckets')
of representative particles and combines discrete inverse transform sampling with
rejection sampling, extensively relying on interval arithmetic to compute bucket–bucket
interaction rate bounds. For a detailed introduction of the computational scheme, we
refer to [Beutel et al. (in press)](#beutel_efficient_2024).

Our code is written in portable C++20 and was tested on Windows with Microsoft Visual C++ 2022
and Clang 16, and on Linux with GCC 12.2 on various (ancient to modern) x86-64 platforms. The
code is single-threaded; although we have a parallel implementation in the works, it is not quite
ready yet.

The simulation code is entirely written in C++. For convenient use, we have additionally developed
a Python module through which an RPMC simulation can be configured and controlled. We provide a
Python script which may be used to run the tests discussed in our simulation, and to generate the
according plots.


## Build dependencies

This project uses the [CMake](https://cmake.org/) build system. It requires a recent version of
CMake (v3.24 or newer), a conforming C++20 compiler, and the following build dependencies:

- [*gsl-lite*](https://github.com/gsl-lite/gsl-lite), an implementation of the [Guidelines Support Library from the C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#S-gsl)
- [*makeshift*](https://github.com/mbeutel/makeshift), a lightweight metaprogramming library
- [*intervals*](https://github.com/mbeutel/intervals), an implementation of interval arithmetic for C++, cf. [Beutel & Strzodka (2023)](beutel_paradigm_2023)
- [*{fmt}*](https://github.com/fmtlib/fmt), a fast formatting library
- [*pybind11*](https://github.com/pybind/pybind11), a modern interop library for Python
- for testing: [*Catch2*](https://github.com/catchorg/Catch2), a unit testing framework

The supplied CMake configuration script uses the [FetchContent](https://cmake.org/cmake/help/latest/module/FetchContent.html)
module to obtain these source dependencies automatically.

Additionally, a recent version of the [*Python*](https://www.python.org/) programming environment
(Python 3.10 was used in our tests) should be installed and accessible to CMake. The Python scripts
supplied with this repository require the following Python packages:

- [*NumPy*](https://numpy.org/)
- [*SciPy*](https://scipy.org/)
- [*Pandas*](https://pandas.pydata.org/)
- [*Matplotlib*](https://matplotlib.org/)

They may be installed with the [pip](https://pypi.org/project/pip/) package manager using the following command:

```bash
python -m pip install --user numpy scipy matplotlib pandas
```


## Usage

After building the project, a Python module named `rpmc` is available in the build directory
(With CPython 3.10 on 64-bit Windows, the file may be named 'rpmc.cp310-win_amd64.pyd').

Open a command environment (on Windows, we highly recommend using
[PowerShell](https://learn.microsoft.com/en-us/powershell/) rather than the arcane 'cmd.exe' command interpreter)
and navigate to the `script` subdirectory of the source repository.
Then append the build directory, here referred to as '<my-build-dir>', to the `PYTHONPATH` environment variable:

```
# cmd.exe (Windows)
export PYTHONPATH=<my-build-dir>;%PYTHONPATH%

# PowerShell (Windows)
$env:PYTHONPATH = "<my-build-dir>;$env:PYTHONPATH"

# Bash (Linux)
export PYTHONPATH=<my-build-dir>:$PYTHONPATH
```

Python will now be able to locate the `rpmc` module that contains the simulation code. The simulation script,
'bucketing-plots.py', supports the following (simplified) command-line syntax:

```
# bucketing-plots.py --help
usage: bucketing-plots.py [-h] [--log] [--report] [--plot] [--overwrite] [--simulations [<type> ...]]
                          [--outdir <dir>] [PARAMS]
```

The following options can be used to control the command-line output:

```
  -h, --help            show this help message and exit
  --log                 produce log messages
  --report              report simulation parameters
  --plot                generate plots
  --overwrite           overwrite an existing archive (default: false)
  --simulations [<type> ...]
                        simulations to run or for which to plot results ('all', '1.1', ...)
  --outdir <dir>        output directory for data file and plots (default 'data/bucketing plots')
```

For many simulations, the script supports configuring the simulation parameters with the following parameter options:

<details>
<summary>click to expand</summary>
<p>
```
  --method <m>          simulation method to use (rpmc, rpmc-traditional)
  --n <int>             number of representative particles
  --Nth <int>           particle regime threshold
  --Nzones <int>        number of zones to simulate
  --n_zone <int>        number of representative particles per zone
  --tag <str>           tag for directory name
  --star.M <M>          mass of central star (default: MSun)
  --star.L <L>, -L <L>  luminosity of central star (default: LSun)
  --planet.ρ [<ρ>,...], -ρP [<ρ>,...]
                        bulk density of planet
  --planet.m [<m>,...], -mP [<m>,...]
                        mass of planet
  --planet.R [<R>,...], -RP [<R>,...]
                        bulk radius of planet
  --planet.a [<a>,...], -aP [<a>,...]
                        semimajor axis of planet
  --planet.i [<i>,...], -incP [<i>,...]
                        inclination angle of planet
  --planet.sini [<sini>,...], -sinincP [<sini>,...]
                        inclination of planet
  --planet.e [<e>,...], -eP [<e>,...]
                        eccentricity of planet
  --planetesimal.ρ [<ρ>,...], -ρPlt [<ρ>,...]
                        bulk density of planetesimals
  --planetesimal.m [<m>,...], -mPlt [<m>,...]
                        mass of planetesimal
  --planetesimal.M [<M>,...], -MPlt [<M>,...]
                        total mass of planetesimals in ring
  --planetesimal.a [<a>,...], -aPlt [<a>,...]
                        semimajor axis of planetesimals
  --planetesimal.N [<N>,...], -NPlt [<N>,...]
                        number of planetesimals
  --planetesimal.R [<R>,...], -RPlt [<R>,...]
                        bulk radius of planetesimals
  --planetesimal.i [<i>,...], -incPlt [<i>,...]
                        inclination dispersion angle arcsin √<sin² i> of planetesimals
  --planetesimal.sini [<sini>,...], -sinincPlt [<sini>,...]
                        inclination dispersion √<sin² i> of planetesimals
  --planetesimal.e [<e>,...], -ePlt [<e>,...]
                        eccentricity dispersion √<e²> of planetesimals
  --planetesimal.Δv [<Δv>,...], -ΔvPlt [<Δv>,...]
                        velocity dispersion √<Δv²> of planetesimals
  --planetesimal.Δv_vh [<Δv_vh>,...], -Δv_vhPlt [<Δv_vh>,...]
                        velocity dispersion √<Δv²> of planetesimals in units of Hill velocity
  --embryo.ρ [<ρ>,...], -ρE [<ρ>,...]
                        bulk density of embryos
  --embryo.m [<m>,...], -mE [<m>,...]
                        mass of embryo
  --embryo.M [<M>,...], -ME [<M>,...]
                        total mass of embryos in ring
  --embryo.a [<a>,...], -aE [<a>,...]
                        semimajor axis of embryos
  --embryo.N [<N>,...], -NE [<N>,...]
                        number of embryos
  --embryo.R [<R>,...], -RE [<R>,...]
                        bulk radius of embryos
  --embryo.i [<i>,...], -incE [<i>,...]
                        inclination dispersion angle arcsin √<sin² i> of embryos
  --embryo.sini [<sini>,...], -sinincE [<sini>,...]
                        inclination dispersion √<sin² i> of embryos
  --embryo.e [<e>,...], -eE [<e>,...]
                        eccentricity dispersion √<e²> of embryos
  --embryo.Δv [<Δv>,...], -ΔvE [<Δv>,...]
                        velocity dispersion √<Δv²> of embryos
  --embryo.Δv_vh [<Δv_vh>,...], -Δv_vhE [<Δv_vh>,...]
                        velocity dispersion √<Δv²> of embryos in units of Hill velocity
  --ring.r <r>, -r <r>  central radius of planetesimal ring
  --ring.Δr <Δr>, -Δr <Δr>
                        width of planetesimal ring
  --ring.Δr/2 <Δr_2>, -Δr/2 <Δr_2>
                        half-width of planetesimal ring
  --disk.rMin <rMin>, -rMin <rMin>
                        inner disk boundary
  --disk.rMax <rMax>, -rMax <rMax>
                        outer disk boundary
  --gas.Cd <Cd>, -Cd <Cd>
                        gas drag coefficient
  --gas.α <α>, -α <α>   viscosity parameter
  --gas.p <p>, -pg <p>  power-law exponent of radial gas density profile (default: -1.5)
  --gas.r0 <r0>, -r0g <r0>
                        reference radius for gas disk
  --gas.ρ0 <ρ0>, -ρ0g <ρ0>
                        midplane gas density at reference radius r₀
  --gas.Σ0 <Σ0>, -Σ0 <Σ0>, -Σ0g <Σ0>
                        surface density at reference radius r₀
  --gas.cs0 <cs0>, -cs0 <cs0>
                        speed of sound at reference radius r₀
  --gas.Tmin <Tmin>, -Tmin <Tmin>
                        background temperature
  --gas.rmin <rmin>, -rming <rmin>
                        lower gas disk boundary (default: 0 AU)
  --gas.rmax <rmax>, -rmaxg <rmax>
                        upper gas disk boundary
  --gas.M <M>, -Mg <M>  total mass of gas disk from rₗₒ to rₕᵢ
  --gas.T0 <T0>, -T0 <T0>
                        mid-plane temperature at reference radius r₀
  --gas.Taccr0 <Taccr0>, -Taccr0 <Taccr0>
                        mid-plane accretion temperature at reference radius r₀
  --gas.Hp0 <Hp0>, -Hp0 <Hp0>
                        pressure scale height at reference radius r₀
  --gas.hp0 <hp0>, -hp0 <hp0>
                        dimensionless pressure scale height Hₚ₀/r₀ at reference radius r₀
  --gas.φ <φ>, -φg <φ>  flaring angle
  --gas.γ <γ>, -γ <γ>   strength of turbulent density fluctuations
  --gas.profile <profile>
                        gas profile type ('none': no gas, 'power-law': power-law gas profile, 'power-law-with-
                        planetary-gap': power-law gas profile with pressure gap as would be carved by planet of given
                        mass; default: none)
  --gas.rPlanet <rPlanet>, -rPl <rPlanet>
                        orbital radius of gap-carving planet
  --gas.rDustTrap <rDustTrap>, -rTrap <rDustTrap>
                        orbital radius of dust trap for St=1 particles (pebble trap, actually)
  --gas.mPlanet <mPlanet>, -mPl <mPlanet>
                        mass of gap-carving planet
  --zones.ΔrMin <ΔrMin>, -ΔrMin <ΔrMin>
                        minimal width of radial zones
  --simulation.method <method>
                        simulation method ('rpmc': RPMC simulation'hybrid': hybrid N-body/RPMC simulation)
  --simulation.effects <effects>
                        physical effects to simulate ('stirring': simulate viscous stirring by swarm particles as per
                        Ormel et al. (2010), 'friction': simulate dynamical friction by swarm particles as per Ormel
                        et al. (2010), 'collisions': simulate stochastic collisions, 'all': all of the above; default:
                        all)
  --collisions.kernel <kernel>, -kernel <kernel>
                        collision handling ('constant': constant test kernel, 'constant-threshold': constant test
                        kernel with threshold, 'linear': linear test kernel, 'linear-threshold': linear test kernel
                        with threshold, 'product': product test kernel, 'runaway': runaway test kernel, 'geometric':
                        simple estimate assuming homogeneous and isotropic motion in bounded volume as per Ormel et
                        al. (2010))
  --collisions.constant-collision-rate <constant_collision_rate>
                        constant kernel collision rate
  --collisions.constant-threshold-mass <constant_threshold_mass>
                        constant kernel threshold mass (with the 'constant-threshold' kernel, particles interact only
                        if the one has a mass above and the other below the given threshold mass)
  --collisions.linear-threshold-mass <linear_threshold_mass>
                        linear kernel threshold mass (with the 'linear -threshold' kernel, particles interact only if
                        the one has a mass above and the other below the given threshold mass)
  --collisions.linear-collision-rate-coefficient <linear_collision_rate_coefficient>
                        linear kernel collision rate coefficient
  --collisions.product-collision-rate-coefficient <product_collision_rate_coefficient>
                        product kernel collision rate coefficient
  --collisions.runaway-collision-rate-coefficient <runaway_collision_rate_coefficient>
                        runaway kernel collision rate coefficient
  --collisions.runaway-critical-mass <runaway_critical_mass>
                        runaway kernel critical mass
  --collisions.outcomes <outcomes>
                        '+'-delimited list of possible collision outcomes ('coagulation': inelastic collisions:
                        coagulation, 'fragmentation': inelastic collisions: fragmentation; default: none)
  --collisions.ε <ε>, -ε <ε>
                        coefficient of restitution in Ormel et al. (2010) collision model
  --collisions.Rfrag <Rfrag>, -Rfrag <Rfrag>
                        radius of fragments in Ormel et al. (2010) collision model
  --simulation.options <options>
                        configuration options for simulation ('locality': enable locality optimisation for kernels
                        which support it; default: none)
  --simulation.bucket-exhaustion <bucket_exhaustion>
                        controls particle-independent exhaustion of bucket property bounds ('none', 'tracer-mass',
                        'tracer-velocity', 'tracer-position', 'tracer', 'swarm', 'full'; default: none)
  --simulation.N-threshold <N_threshold>, -NTh <N_threshold>
                        swarm particle number threshold for active N-body particles (only particles with a particle
                        number below the threshold are treated as active N-body particles which exert direct
                        gravitational force onto other particles; default: 1.5)
  --simulation.St-NBody-threshold <St_NBody_threshold>, -StNBTh <St_NBody_threshold>
                        Stokes number threshold for N-body kinetics (mutual collision, viscous stirring, and dynamical
                        friction is suppressed if both particles are self-representing and have Stokes numbers above
                        the threshold; default: ∞)
  --simulation.m-NBody-threshold <m_NBody_threshold>, -mNBTh <m_NBody_threshold>
                        mass threshold for N-body kinetics (mutual collision, viscous stirring, and dynamical friction
                        is suppressed if both particles are self-representing and have masses above the threshold;
                        default: ∞)
  --simulation.St-equilibrium-threshold <St_equilibrium_threshold>, -StEqTh <St_equilibrium_threshold>
                        Stokes number threshold for equilibrium gas drag (equilibrium gas drag is suppressed for
                        particles with Stokes numbers above the threshold; default: ∞)
  --simulation.St-dust-threshold <St_dust_threshold>, -StDustTh <St_dust_threshold>
                        Stokes number threshold for dust (collision is suppressed if both particles are below the
                        threshold, and viscous stirring and dynamical friction are suppressed if at least one particle
                        is below the threshold; default: 0)
  --simulation.m-dust-threshold <m_dust_threshold>, -mDustTh <m_dust_threshold>
                        mass threshold for dust (collision is suppressed if both particles are below the threshold,
                        and viscous stirring and dynamical friction are suppressed if at least one particle is below
                        the threshold; default: 1 kg)
  --simulation.relative-change-update-threshold <relative_change_update_threshold>
                        maximal admitted fractional change before update (maximal admitted fractional change maxᵢ{
                        Δa/a, Δe/e, Δi/i } after which the stirring and collision rates should be recomputed; default:
                        0.01)
  --simulation.particle-regime-threshold <particle_regime_threshold>
                        regime threshold for particle number (collisions involving a particle from a swarm with a
                        particle number less than `particle-regime-threshold` are treated in a different regime to
                        properly support oligarchic growth; default: 100)
  --simulation.particle-regime-threshold-for-interaction-rates <particle_regime_threshold_for_interaction_rates>
                        optional separate threshold for the particle regime when computing collision rates;
                        experimental use only (A value of 0 indicates that `particle-regime-threshold` is used
                        instead: the only sensible choice; default: 0)
  --simulation.bin-widening <bin_widening>, -bin-widening <bin_widening>
                        percentual widening of buckets to decrease updating rate (default: 0)
  --simulation.removal-bucket-update-delay <removal_bucket_update_delay>, -removal-bucket-update-delay <removal_bucket_update_delay>
                        percentual update delay of buckets to decrease updating rate on particle removal (default: 1)
  --simulation.rejection-bucket-update-delay <rejection_bucket_update_delay>, -rejection-bucket-update-delay <rejection_bucket_update_delay>
                        percentual update delay of buckets to decrease updating rate on event rejection (default: 1)
  --simulation.subclass-resolution-factor <subclass_resolution_factor>
                        number of subdivisions of the smallest resolved length (a granularity of sorts) (default: 8)
  --simulation.subclass-widening-fraction <subclass_widening_fraction>, -spread <subclass_widening_fraction>
                        widening fraction of sub-bucket widths (default: 0.05)
  --simulation.r-bins <r_bins>, -r-bins <r_bins>
                        number of radial bins in the ring (default: 10)
  --simulation.M-bins-per-decade <M_bins_per_decade>, -M-bins-per-decade <M_bins_per_decade>
                        number of bins per decadic order of magnitude of swarm mass (default: 2 (@ 1.e+24 g))
  --simulation.m-bins-per-decade <m_bins_per_decade>, -m-bins-per-decade <m_bins_per_decade>
                        number of bins per decadic order of magnitude of particle mass (default: 2 (@ 1.e+24 g))
  --simulation.e-bins-per-decade <e_bins_per_decade>, -e-bins-per-decade <e_bins_per_decade>
                        number of bins per decadic order of magnitude of eccentricity (default: 1 (≥ 1.e-6))
  --simulation.sininc-bins-per-decade <sininc_bins_per_decade>, -sininc-bins-per-decade <sininc_bins_per_decade>
                        number of bins per decadic order of magnitude of inclination (default: 1 (≥ 1.e-6))
  --simulation.mass-growth-factor <mass_growth_factor>
                        minimal percentaged mass increase in coagulation (default: 0)
  --simulation.velocity-growth-factor <velocity_growth_factor>
                        minimal percentaged velocity change in stirring/friction (default: 0)
  --simulation.velocity-growth-rate <velocity_growth_rate>
                        maximal interaction rate for stochastic velocity changes (default: 0/yr)
  --simulation.timesteps-per-orbit <timesteps_per_orbit>
                        how many timesteps to take for the smallest orbit represented in the system (default: 20)
  --simulation.min-sync-time <min_sync_time>
                        minimal time interval after which to synchronize N-body state and stochastic simulation state
                        (default: 0 yr)
  --simulation.nPlt <nPlt>, -nPlt <nPlt>
                        number of tracers for planetesimals (default: 1)
  --simulation.nPltR <nPltR>, -nPltR <nPltR>
                        number of unallocated tracers to reserve for planetesimals (default: 0)
  --simulation.nE <nE>, -nE <nE>
                        number of tracers for embryos (default: 0)
  --simulation.hierarchical-ordering-base <hierarchical_ordering_base>
                        base of logarithmic buckets in hierarchical ordering (default: 10)
  --simulation.random-seed <random_seed>
                        random seed (default: 42)
  --simulation.name <name>, -name <name>
                        name of simulation scenario
  --simulation.id <id>, -id <id>
                        id of simulation scenario
  --simulation.NSteps <NSteps>, -NSteps <NSteps>
                        number of time steps (default: 256)
  --simulation.tMinLinear <tMinLinear>
                        minimal time on linear timescale (default: 0 years)
  --simulation.tMinLog <tMinLog>
                        minimal time on logarithmic timescale (default: 100 years)
  --simulation.T <T>, -T <T>
                        total duration of simulation (default: 100 years)
```
</p>
</details>


## Reproducing the figures from our publication

In this section, we briefly describe the pseudocommands necessary to reproduce the figures from our publication.
Note that some figures depict benchmark results and thus require a two-stage execution: running the benchmarks,
then collecting and plotting the results.

All command snippets stated in this section call the 'bucketing-plots.py' script with a variety of command-line arguments.
We used Christoph Klein's excellent [*Measurement Instructor*](https://github.com/codecircuit/minstructor) application,
herein referred to as *minstructor*, to conduct parameter studies and run benchmarks. Some of the pseudocommands in this
section use *minstructor* syntax to state lists and ranges of parameters. For instance, consider the following pseudocommand:
```
bucketing-plots.py --simulations 2.1 --n=[1024,65536] --simulation.random-seed=range(0,3)
```
Here, the bracket syntax in the command-line parameter `n` means that `n` can be `1024` or `65536`,
and the `range()` syntax states that parameter `simulation.random-seed` can be any integer ∈ ℕ₀ ∩ \[0,3\)
(cf. the [*minstructor* manual](https://github.com/codecircuit/minstructor) for a definition of *minstructor*'s
parameter expansion syntax).

A pseudocommand that contains *minstructor*-style list or range arguments is understood to be executed multiple times
so as to realise the full Cartesian product of possible parameter combinations. For the example stated here, the
single pseudocommand is understood to expand to a list of six commands:
```
bucketing-plots.py --simulations 2.1 --n=1024 --simulation.random-seed=0
bucketing-plots.py --simulations 2.1 --n=1024 --simulation.random-seed=1
bucketing-plots.py --simulations 2.1 --n=1024 --simulation.random-seed=2
bucketing-plots.py --simulations 2.1 --n=65536 --simulation.random-seed=0
bucketing-plots.py --simulations 2.1 --n=65536 --simulation.random-seed=1
bucketing-plots.py --simulations 2.1 --n=65536 --simulation.random-seed=2
```
This expansion can be achieved with a scripting language, or most conveniently with the
[*minstructor*](https://github.com/codecircuit/minstructor) tool.

A pseudocommand that contains no list or range arguments is simply taken as a literal command.

In all calls to 'bucketing-plots.py', use the `--log` command to get more frequent intermediate updates,
which is especially useful for long-running simulations. If you wish to have the command-line parameters
echoed back (for example, to be sure they have been parsed correctly), use the `--report` command.


### Figure 6

Figure title: 'Analytical solutions and RPMC simulations for the coagulation test with the linear kernel (Eq. (64)) using the
bucketing scheme.'

This figure can be reproduced by executing the following pseudocommand:

```
bucketing-plots.py --simulations 2.1 --Nth=10 --n=[1024,65536] --simulation.m-bins-per-decade=2 --simulation.M-bins-per-decade=2 \
    --method=rpmc --simulation.bin-widening=0.05 --simulation.random-seed=0 --plot
```


### Figure 7

Figure title: 'RPMC simulations for the coagulation test with the runaway kernel (Eq. (70)) using the bucketing scheme.'

This figure can be reproduced by executing the following pseudocommand:

```
bucketing-plots.py --simulations 1.3 --Nth=10 --n=[2048,65536] --simulation.m-bins-per-decade=2 --simulation.M-bins-per-decade=2 \
    --method=rpmc --simulation.bin-widening=0.05 --simulation.random-seed=0 --plot
```


### Figure 8

Figure title: 'Raw interaction rate λ(𝐪,𝐪') and entity interaction rate λₑₙₜ(𝐪,𝐪',δ) for the runaway kernel (Eq. (70)).'

This figure can be reproduced by executing the following command:

```
bucketing-plots.py --simulations 1.1 --plot
```


### Figure 9

Figure title: 'Snapshots of the entity interaction rates λⱼₖ and corresponding bucket–bucket interaction rate bounds λᴊᴋ for
the runaway kernel (Eq. (70)) in an RPMC simulation at different times t = 8, t = 25.'

This figure can be reproduced by executing the following command:

```
bucketing-plots.py --simulations 1.6 --plot
```


### Figure 11

Figure title: 'Performance analysis of the linear kernel test for the traditional scheme and the bucketing scheme.'

To reproduce this figure, first execute the benchmarks with the following pseudocommands, storing the
command-line output of each run to a text file:

```
bucketing-plots.py --simulations 2.2 --Nth=10 --n=logspace(5,13,9,2) --method=rpmc-traditional \
    --simulation.random-seed=range(0,5) --report
bucketing-plots.py --simulations 2.2 --Nth=10 --n=logspace(5,17,13,2) --simulation.m-bins-per-decade=2 \
    --simulation.M-bins-per-decade=2 --method=rpmc --simulation.random-seed=range(0,5) --report
```

An individual run produces command-line output akin to the following:

```
[bucketing-plots] Running '2.2 linear kernel benchmark'.
n = 32
Nth = 10
method = rpmc-traditional
Running simulation... done.
Elapsed time: 0.40533350000623614 s
```

Using a tool like *mcollector* from the [*minstructor* package](https://github.com/codecircuit/minstructor),
assemble a tab-separated text file from all the text files generated by the individual benchmark runs and save
it as 'script/data/perf/linear-kernel.tsv' relative to the source directory. The tab-separated text file should
have the following format (additional named columns will be tolerated):
```
n	method	time
16384	rpmc	2.231369998306036
4096	rpmc	0.5330711780115962
...
```

Then execute the following command to generate the figure:

```
bucketing-plots.py --simulations 2.3 --plot
```


### Figure 12

Figure title: 'Performance analysis of the runaway kernel test for the traditional scheme and the bucketing scheme.'

To reproduce this figure, first execute the benchmarks with the following pseudocommands, storing the
command-line output of each run to a text file:

```
bucketing-plots.py --simulations 1.4 --Nth=10 --n=logspace(5,13,9,2) --method=rpmc-traditional \
    --simulation.random-seed=range(0,5) --report
bucketing-plots.py --simulations 1.4 --Nth=10 --n=logspace(5,20,16,2) --simulation.m-bins-per-decade=2 \
    --simulation.M-bins-per-decade=2 --method=rpmc --simulation.random-seed=range(0,5) --report
```

An individual run produces command-line output akin to the following:

```
[bucketing-plots] Running '1.4 runaway kernel benchmark'.
n = 32
Nth = 10
method = rpmc
Running simulation... done.
Elapsed time: 0.8392966999672353 s
Statistics:
===========
    simulation-time: 59.1442
    num-events: 2463
    num-rejections: 6364
    num-excess-samplings: 0
    num-buckets-out-of-reach: 0
    num-out-of-reach: 0
    buckets-out-of-reach-probability: 0
    out-of-reach-probability: 0
    avg-acceptance-probability: 0.27903
    num-updates: 4241
    num-bucket-updates: 1714
    num-recomputes: 879
    num-bucket-changes: 1730
    recompute-probability: 0.207262
    bucket-update-probability: 0.40415
    bucket-change-probability: 0.407923
    min-num-buckets: 2
    max-num-buckets: 15
    avg-num-buckets: 7.75964
    min-num-active-particles: 2
    max-num-active-particles: 107
    avg-num-active-particles: 48.6573
```

Using a tool like *mcollector* from the [*minstructor* package](https://github.com/codecircuit/minstructor),
assemble a tab-separated text file from all the text files generated by the individual benchmark runs and save
it as 'script/data/perf/runaway-kernel.tsv' relative to the source directory. The tab-separated text file should
have the following format (additional named columns will be tolerated):
```
n	method	time	simulation-time	num-events	num-rejections	num-excess-samplings	num-buckets-out-of-reach	num-out-of-reach	buckets-out-of-reach-probability	out-of-reach-probability	avg-acceptance-probability	num-updates	num-bucket-updates	num-recomputes	num-bucket-changes	recompute-probability	bucket-update-probability	bucket-change-probability	min-num-buckets	max-num-buckets	avg-num-buckets	min-num-active-particles	max-num-active-particles	avg-num-active-particles
1048576	rpmc	98.0119669130072	67.9007	49969776	160157808	0	0	0	0	0	0.237807	87878637	29980375	20157	33423894	0.000229373	0.341157	0.380342	2	53	22.314	2	5188887	2.75965e+06
1048576	rpmc	99.30565563589334	66.0707	50013537	159578238	0	0	0	0	0	0.238624	87954221	29974197	24555	33414391	0.000279179	0.340793	0.379907	2	54	23.1232	2	5184580	2.75855e+06
...
```

Then execute the following command to generate the figure:

```
bucketing-plots.py --simulations 1.5 --plot
```


### Figure 13

Figure title: 'Results for the simplified Ormel’s model starting with n₀ =32 768 representative particles.'

This figure can be reproduced by executing the following pseudocommand:

```
bucketing-plots.py --simulations [3.1,3.9] \
    --Nth=10 --Nzones=64 --n_zone=512 --simulation.r-bins=0.1 --simulation.e-bins-per-decade="5 (≥ 1.e-5)"
    --simulation.sininc-bins-per-decade="5 (≥ 1.e-5)" --simulation.m-bins-per-decade=6 --simulation.M-bins-per-decade=6
    --method=rpmc --simulation.bin-widening=0 --simulation.subclass-widening-fraction=0.03
    --simulation.velocity-growth-factor=0.02 --simulation.options=locality --simulation.random-seed=0
    --report --plot --overwrite
```


### Figure 14

Figure title: 'Memory cost analysis of simulating Ormel’s model.'

This figure can be reproduced by executing the following command:

```
bucketing-plots.py --simulations 3.6 --plot
```


### Figure 15

Figure title: 'Performance analysis of Ormel's model for the traditional scheme and the bucketing scheme with boosted dynamical friction and viscous stirring.'

To reproduce this figure, first execute the benchmarks with the following pseudocommands, storing the
command-line output of each run to a text file:

```
# without fragmentation
bucketing-plots.py --simulations 3.3 \
    --Nth=10 --Nzones=64 --n_zone=logspace(3,6,4,2) \
    --method=rpmc-traditional \
    --simulation.velocity-growth-factor=0.02 --simulation.options=locality --simulation.random-seed=range(0,3) \
    --report
bucketing-plots.py --simulations 3.3 \
    --Nth=10 --Nzones=64 --n_zone=logspace(3,11,9,2) --simulation.r-bins=0.1 --simulation.e-bins-per-decade="5 (≥ 1.e-5)" \
    --simulation.sininc-bins-per-decade="5 (≥ 1.e-5)" --simulation.m-bins-per-decade=6 --simulation.M-bins-per-decade=6 \
    --method=rpmc --simulation.bin-widening=0 --simulation.subclass-widening-fraction=0.03 \
    --simulation.velocity-growth-factor=0.02 --simulation.options=locality --simulation.random-seed=range(0,3) \
    --report

# with fragmentation
bucketing-plots.py --simulations 3.10 \
    --Nth=10 --Nzones=64 --n_zone=logspace(3,6,4,2) \
    --method=rpmc-traditional \
    --simulation.velocity-growth-factor=0.02 --simulation.options=locality --simulation.random-seed=range(0,3) \
    --report
bucketing-plots.py --simulations 3.10 \
    --Nth=10 --Nzones=64 --n_zone=logspace(3,11,9,2) --simulation.r-bins=0.1 --simulation.e-bins-per-decade="5 (≥ 1.e-5)" \
    --simulation.sininc-bins-per-decade="5 (≥ 1.e-5)" --simulation.m-bins-per-decade=6 --simulation.M-bins-per-decade=6 \
    --method=rpmc --simulation.bin-widening=0 --simulation.subclass-widening-fraction=0.03 \
    --simulation.velocity-growth-factor=0.02 --simulation.options=locality --simulation.random-seed=range(0,3) \
    --report
```

An individual run produces command-line output akin to the following:

```
[bucketing-plots] Running '3.3 1AU stirring benchmark'.
n_zone = 8
Nzones = 64
Nth = 10
method = rpmc
r-bins = 0.1
M-bins-per-decade = 6
m-bins-per-decade = 6
e-bins-per-decade = 5 (≥ 1e-05)
sininc-bins-per-decade = 5 (≥ 1e-05)
velocity-growth-factor = 0.02
options = locality

Simulation:
  - star.M: 1 MSun
  - star.L: 1 LSun
  - planetesimal.ρ: 3 g/cm³
  - planetesimal.m: 8.03536e-10 MEarth
  - planetesimal.M: 0.158501 MEarth
  - planetesimal.N: 197254627
  - planetesimal.R: 7.25566 km
  - planetesimal.i: 0.00521937 °
  - planetesimal.sini: 9.10953e-05
  - planetesimal.e: 0.000128828
  - planetesimal.Δv: 470 cm/s
  - planetesimal.Δv_vh: 16.9651
  - embryo.ρ: 0
  - embryo.m: 0
  - embryo.M: 0
  - embryo.N: 0
  - embryo.R: 0
  - embryo.i: 0
  - embryo.sini: 0
  - embryo.e: 0
  - embryo.Δv: 0
  - embryo.Δv_vh: 0
  - ring.r: 1 AU
  - ring.Δr: 0.04032 AU
  - ring.Δr/2: 0.02016 AU
  - disk.rMin: 0.5 AU
  - disk.rMax: 1.5 AU
  - gas.α: 0.001
  - gas.p: -1
  - gas.r0: 1 AU
  - gas.ρ0: 1.2e-09 g/cm³
  - gas.Σ0: 1510.62 g/cm²
  - gas.cs0: 100000 cm/s
  - gas.Tmin: 271.369 K
  - gas.rmin: 0.1 AU
  - gas.rmax: 400 AU
  - gas.T0: 278.625 K
  - gas.Hp0: 0.0335706 AU
  - gas.hp0: 0.0335706
  - gas.φ: 0.05
  - gas.profile: power-law
  - zones.ΔrMin: 0.00063 AU
  - simulation.method: rpmc
  - simulation.effects: stirring + friction + collisions
  - collisions.kernel: geometric
  - collisions.outcomes: coagulation
  - collisions.ε: 0.01
  - collisions.Rfrag: 1e-05 km
  - simulation.options: locality
  - simulation.particle-regime-threshold: 10
  - simulation.bin-widening: 0
  - simulation.subclass-widening-fraction: 0.03
  - simulation.r-bins: 0.01
  - simulation.M-bins-per-decade: 6
  - simulation.m-bins-per-decade: 6
  - simulation.e-bins-per-decade: 5 (≥ 1e-05)
  - simulation.sininc-bins-per-decade: 5 (≥ 1e-05)
  - simulation.mass-growth-factor: 5 %
  - simulation.velocity-growth-factor: 2 %
  - simulation.nPlt: 512
  - simulation.nPltR: 4608
  - simulation.nE: 0
  - simulation.random-seed: 0
  - simulation.NSteps: 4
  - simulation.tMinLog: 1e-06 Myr
  - simulation.T: 0.001 Myr

Running simulation... done.
Elapsed time: 20.810151600046083 s
Profiling data:
===========
    total-duration: 20.8874s
    sampling-duration: 3.63848s
    updating-duration: 17.1631s
    extra-duration: 0.0858641s

Statistics:
===========
    simulation-time: 3.15571e+10
    num-events: 4058
    num-rejections: 27825
    num-excess-samplings: 0
    num-buckets-out-of-reach: 10779
    num-out-of-reach: 13389
    buckets-out-of-reach-probability: 0.387385
    out-of-reach-probability: 0.481186
    avg-acceptance-probability: 0.127278
    num-updates: 4725
    num-bucket-updates: 4281
    num-recomputes: 1100
    num-bucket-changes: 4281
    recompute-probability: 0.232804
    bucket-update-probability: 0.906032
    bucket-change-probability: 0.906032
    min-num-buckets: 1
    max-num-buckets: 37
    avg-num-buckets: 22.5917
    min-num-active-particles: 512
    max-num-active-particles: 512
    avg-num-active-particles: 512
```

Using a tool like *mcollector* from the [*minstructor* package](https://github.com/codecircuit/minstructor),
assemble a tab-separated text file from all the text files generated by the individual benchmark runs and save
it as 'script/data/perf/stirring-test9-boost.tsv' (for the runs without fragmentation)
and as 'script/data/perf/stirring-test9-frag-boost.tsv' (for the runs with fragmentation)
relative to the source directory. The tab-separated text file should
have the following format (additional named columns will be tolerated):
```
n_zone	Nzones	method	r-bins	M-bins-per-decade	m-bins-per-decade	e-bins-per-decade	sininc-bins-per-decade	velocity-growth-factor	options	nPlt	nPltR	nE	time	total-duration	sampling-duration	updating-duration	extra-duration	simulation-time	num-events	num-rejections	num-excess-samplings	num-buckets-out-of-reach	num-out-of-reach	buckets-out-of-reach-probability	out-of-reach-probability	avg-acceptance-probability	num-updates	num-bucket-updates	num-recomputes	num-bucket-changes	recompute-probability	bucket-update-probability	bucket-change-probability	min-num-buckets	max-num-buckets	avg-num-buckets	min-num-active-particles	max-num-active-particles	avg-num-active-particles
512	64	rpmc	0.01	6	6	5	5	2	locality	32768	294912	0	27709.913713712245	27710.1	23951.7	3758.11	0.27768	4.73363e+12	150861847	10140108964	0	3955839029	5531855119	0.390118	0.545542	0.0146596	199290657	16780093	733104	16795462	0.00367857	0.0841991	0.0842762	2	682	440.186	153	45125	39943.9
512	64	rpmc	0.01	6	6	5	5	2	locality	32768	294912	0	27045.478982854635	27045.7	23399.3	3646.1	0.27422	4.73364e+12	149340748	9916931011	0	3807311883	5351411011	0.38392	0.539624	0.0148358	190937391	16272268	695447	16288189	0.00364228	0.0852231	0.0853064	2	684	444.683	175	44679	39822.7
...
```

Then execute the following command to generate the figure:

```
bucketing-plots.py --simulations 3.5 --plot
```


### Figure 16

Figure title: 'Parameter study of the bucket densities θᵥ, θₘ, θᴍ in Ormel’s model.'

To reproduce this figure, first execute the benchmarks with the following script, storing the
command-line output of each run to a text file
(note that the script contains pseudocommands that need to be expanded, and whose output needs to
be stored to individual text files):

```bash
#!/usr/bin/env bash

e_BINS='3 4 5 6 7'
m_BINS='3 4 5 6 7'
mkdir -p test7/param/a
mkdir -p test7/param/b
for e_bins in $e_BINS; do
    for m_bins in $m_BINS; do
        bucketing-plots.py --simulations 3.3 --Nth=10 --Nzones=64 --n_zone=64 --simulation.r-bins=0.1 \
            --simulation.e-bins-per-decade=\"$e_bins (≥ 1.e-5)\" --simulation.sininc-bins-per-decade=\"$e_bins (≥ 1.e-5)\" \
            --simulation.m-bins-per-decade=$m_bins --simulation.M-bins-per-decade=$m_bins --method=rpmc \
            --simulation.bin-widening=0 --simulation.subclass-widening-fraction=0.03 --simulation.velocity-growth-factor=0.02 \
            --simulation.options=locality --simulation.random-seed=range(0,10) --report
    done
done
```

Benchmark outputs are similar as for [Figure 15](#figure-15) and need to be processed similarly, but the
result needs to be stored in a tab-separated file 'data/perf/stirring-test7-param.tsv' relative to the source directory.

To generate the figure, then execute the following command:

```
bucketing-plots.py --simulations 3.8 --plot
```


### Figure 17

Figure title: 'Parameter study of the widening fraction f in Ormel’s model without boosting.'

To reproduce this figure, first execute the benchmarks with the following pseudocommand:

```
bucketing-plots.py --simulations 3.3 --Nth=10 --Nzones=4 --n_zone=32 --simulation.r-bins=0.1 \
    --simulation.e-bins-per-decade="5 (≥ 1.e-5)" --simulation.sininc-bins-per-decade="5 (≥ 1.e-5)" \
    --simulation.m-bins-per-decade=6 --simulation.M-bins-per-decade=6 --method=rpmc \
    --simulation.bin-widening=linspace(0,0.1,11) --simulation.subclass-widening-fraction=0.03 \
    --simulation.velocity-growth-factor=0 --simulation.options=locality --simulation.random-seed=range(0,5) --report
```

Benchmark outputs are similar as for [Figure 15](#figure-15) and need to be processed similarly, but the
result needs to be stored in a tab-separated file 'data/perf/stirring-test7-widen.tsv' relative to the source directory.

To generate the figure, then execute the following command:

```
bucketing-plots.py --simulations 3.7 --plot
```


## Troubleshooting

- **Python fails to load my module on Windows with some generic error message.**  
  Compilers on Windows tend to prefer static linkage for external libraries, whereas compilers on Linux
  and similar platforms prefer dynamically linked libraries. Thanks to the [rpath](https://en.wikipedia.org/wiki/Rpath),
  the dynamic linking loader on Linux will be able to locate dynamic libraries without additional effort.
  On Windows, where it is more common to redistribute binaries rather than building them on the target machine,
  there is no analogue to the rpath mechanism. Therefore, if any of the build dependencies was linked dynamically
  on Windows, the loader will follow the
  [dynamic-link library search order](https://learn.microsoft.com/en-us/windows/win32/dlls/dynamic-link-library-search-order)
  to locate the dependencies. Usually, the dependencies will thus not be found unless they have been copied to the build
  directory where the Python module resides or their directory has been added to the `PATH` environment variable.  
    
  The easiest remedy is simply to copy the required DLLs to the build directory. We recommend using the
  [*Dependencies*](https://github.com/lucasg/Dependencies) tool to analyse the runtime dependencies of the Python module.


## References

<a name="beutel_efficient_2024">M. Beutel, C. P. Dullemond, and R. Strzodka.
Efficient simulation of stochastic interactions among representative Monte Carlo particles.
Astronomy & Astrophysics, in press.</a>

<a name="beutel_stochastic_2024">M. Beutel.
Stochastic and deterministic methods for simulating the evolution of solid bodies in protoplanetary disks.
Doctoral thesis, submitted to Ruprecht-Karls-Universität Heidelberg, Germany, 2024.</a>

<a name="beutel_improved_2023">M. Beutel and C. P. Dullemond.
An improved Representative Particle Monte Carlo method for the simulation of particle growth.
Astronomy & Astrophysics, 670:A134, February 2023.</a>
ISSN 0004-6361, 1432-0746. doi: 10.1051/0004-6361/202244955.
URL https://www.aanda.org/10.1051/0004‐6361/202244955

<a name="beutel_paradigm_2023">M. Beutel and R. Strzodka.
A Paradigm for Interval-Aware Programming.
In John Gustafson, Siew Hoon Leong, and Marek Michalewicz, editors, Next Generation
Arithmetic, Lecture Notes in Computer Science, pages 38–60, Cham, 2023. Springer
Nature Switzerland. ISBN 978-3-031-32180-1. doi: 10.1007/978-3-031-32180-1_3.</a>
