
import os
import sys
import math
import time
import types
import argparse

import numpy as np
import matplotlib.cm
import matplotlib.colors
import matplotlib.ticker
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import tools.parameters
from tools import projection
from planets import plot
from tools.argparse import add_boolean_argument

import planets.simulation
import planets.plot
import const.cgs as cgs

import rpmc


# Define command-line interface.
parser = argparse.ArgumentParser(
    fromfile_prefix_chars='@',
    description='Make bucketing-related plots.')
parser.add_argument('--log',                 action='store_true',
                    help='produce log messages')
parser.add_argument('--display',             action='store_true',
                    help='render simulation state in dedicated window')
parser.add_argument('--benchmark',           action='store_true',
                    help='only run simulation benchmark: do not store, do not plot, minimise impact of snapshotting during run')
parser.add_argument('--report',              action='store_true',
                    help='report simulation parameters')
parser.add_argument('--dry-run',             action='store_true',
                    help='do not actually run simulation or generate plots')
parser.add_argument('--do-not-store',        action='store_true',
                    help='do not store results to archive after running simulation')
parser.add_argument('--load',                action='store_true',
                    help='load results from archive instead of running simulation')
parser.add_argument('--plot',                action='store_true',
                    help='generate plots')
parser.add_argument('--format',              metavar='<fmt>', type=str,
                    help="data format ('text' or 'binary'; default: 'text')")
add_boolean_argument(parser, '--compress',   default=True,
                    help="whether to compress the output data (default: true)")
add_boolean_argument(parser, '--overwrite',  default=False,
                    help="overwrite an existing archive (default: false)")
parser.add_argument('--simulations',         action='extend', nargs='*', metavar='<type>', type=str, default=[],
                    help="simulations to run or for which to plot results ('all', '1.1', ...)"),
parser.add_argument('--outdir',              default='data/bucketing plots', metavar='<dir>', type=str,
                    help='output directory for data file and plots (default \'data/bucketing plots\')')
parser.add_argument('--method',              metavar='<m>', type=str, default='rpmc',
                    help="simulation method to use (rpmc, rpmc-traditional)")
parser.add_argument('--n',                   metavar='<int>',  type=int, default=1024,
                    help="number of representative particles")
parser.add_argument('--Nth',                 metavar='<int>',  type=int, default=10,
                    help="particle regime threshold")
parser.add_argument('--Nzones',              metavar='<int>',  type=int, default=270,
                    help="number of zones to simulate")
parser.add_argument('--n_zone',              metavar='<int>',  type=int, default=600,
                    help="number of representative particles per zone")
parser.add_argument('--tag',                 metavar='<str>',  type=str, default='',
                    help="tag for directory name")

planets.simulation.params.add_to_argparser(parser)


def report_args(args):
    config = planets.simulation.params.to_configuration(args=args, filter_pred=tools.parameters.FilterPredicates.not_None)
    print('')
    tools.configuration.report(config)
    print('')


def plot_interaction_rates(base_filename, ext, cmd_args, sim_data, t):
    dirpath, basename = os.path.split(base_filename)

    n = cmd_args.n
    #_, data_by_time = sim_data.snapshots_by_time()
    #data = data_by_time(t)
    #n_active = len(data.N[data.N > 0.9])
    state = sim_data.state
    n_active = np.count_nonzero(state.M > 0.9*state.m)
    params = np.array([0], dtype=np.intp)
    rates = np.zeros(shape=[n_active,n_active], dtype=float)
    true_bucket_rates = np.zeros(shape=[n_active,n_active], dtype=float)
    bucket_rates = np.zeros(shape=[n_active,n_active], dtype=float)
    acc_probs = np.zeros(shape=[n_active,n_active], dtype=float)
    sim_data.sim.inspect(dst=rates, quantity='discrete-operator-0/interaction-model-0/interaction rates', params=params)
    #sim_data.sim.inspect(dst=true_bucket_rates, quantity='discrete-operator-0/interaction-model-0/true particle bucket interaction rates', params=params)
    sim_data.sim.inspect(dst=bucket_rates, quantity='discrete-operator-0/interaction-model-0/particle bucket interaction rates', params=params)
    sim_data.sim.inspect(dst=acc_probs, quantity='discrete-operator-0/interaction-model-0/acceptance probabilities', params=params)

    filename = os.path.join(dirpath, '{} interaction-rates n={} t={:g}{}'.format(basename, n, t, ext).replace(' ', '_'))
    print("Writing plot '{}'.".format(filename))
    fig2, ((ax2a, ax2b), (ax2c, ax2d)) = plt.subplots(2, 2, sharex=False, sharey=False, figsize=[10, 10])
    rates_non0 = rates[rates > 0]
    bucket_rates_non0 = bucket_rates[bucket_rates > 0]
    true_bucket_rates_non0 = true_bucket_rates[true_bucket_rates > 0]
    vmax = max(np.max(rates_non0) if len(rates_non0) > 0 else 1., np.max(bucket_rates_non0) if len(bucket_rates_non0) > 0 else 1., np.max(true_bucket_rates_non0) if len(true_bucket_rates_non0) > 0 else 1.)
    vmin = vmax*1.e-8
    im = ax2a.imshow(rates.T, interpolation='nearest', origin='lower',
       extent=[0.5, n_active+0.5, 0.5, n_active+0.5],
       cmap='rainbow', norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax))
    fig2.colorbar(im, ax=ax2a)
    im = ax2c.imshow(bucket_rates.T, interpolation='nearest', origin='lower',
       extent=[0.5, n_active+0.5, 0.5, n_active+0.5],
       cmap='rainbow', norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax))
    fig2.colorbar(im, ax=ax2c)
    im = ax2d.imshow(true_bucket_rates.T, interpolation='nearest', origin='lower',
       extent=[0.5, n_active+0.5, 0.5, n_active+0.5],
       cmap='rainbow', norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax))
    fig2.colorbar(im, ax=ax2d)
    im = ax2b.imshow(acc_probs.T, interpolation='nearest', origin='lower',
       extent=[0.5, n_active+0.5, 0.5, n_active+0.5],
       cmap='rainbow', norm=matplotlib.colors.LogNorm(vmin=1.e-8, vmax=1.))
    fig2.colorbar(im, ax=ax2b)

    ax2a.set_title(r'tracer–swarm interaction rates ($\mathrm{year}^{-1}$)')
    ax2a.set_xlabel('tracer index')
    ax2a.set_ylabel('swarm index')
    ax2b.set_title(r'acceptance probabilities')
    ax2b.set_xlabel('tracer index')
    ax2b.set_ylabel('swarm index')
    ax2c.set_title(r'bucket bounds of tracer–swarm interaction rates ($\mathrm{year}^{-1}$)')
    ax2c.set_xlabel('tracer index')
    ax2c.set_ylabel('swarm index')
    ax2d.set_title(r'bucket bounds of tracer–swarm interaction rates ($\mathrm{year}^{-1}$)')
    ax2d.set_xlabel('tracer index')
    ax2d.set_ylabel('swarm index')

    fig2.savefig(filename)

    indices = np.zeros(shape=[n_active], dtype=np.intp)
    sim_data.sim.inspect(dst=indices, quantity='discrete-operator-0/indices', params=params)
    bms = np.repeat(state.m.to_numpy()[indices], n_active).reshape([n_active, n_active])
    bMs = np.repeat(state.M.to_numpy()[indices], n_active).reshape([n_active, n_active])

    filename = os.path.join(dirpath, '{} interaction rate correlated masses n={} t={:g}{}'.format(basename, n, t, ext).replace(' ', '_'))
    print("Writing plot '{}'.".format(filename))
    fig3, ((ax3a, ax3b), (ax3c, ax3d)) = plt.subplots(2, 2, sharex=False, sharey=False, figsize=[10, 10])
    im = ax3a.imshow(bms.T, interpolation='nearest', origin='lower',
       extent=[0.5, n_active+0.5, 0.5, n_active+0.5],
       cmap='rainbow', norm=matplotlib.colors.LogNorm())
    fig3.colorbar(im, ax=ax3a)
    im = ax3b.imshow(bms, interpolation='nearest', origin='lower',
       extent=[0.5, n_active+0.5, 0.5, n_active+0.5],
       cmap='rainbow', norm=matplotlib.colors.LogNorm())
    fig3.colorbar(im, ax=ax3b)
    im = ax3c.imshow(bMs.T, interpolation='nearest', origin='lower',
       extent=[0.5, n_active+0.5, 0.5, n_active+0.5],
       cmap='rainbow', norm=matplotlib.colors.LogNorm())
    fig3.colorbar(im, ax=ax3c)
    im = ax3d.imshow(bMs, interpolation='nearest', origin='lower',
       extent=[0.5, n_active+0.5, 0.5, n_active+0.5],
       cmap='rainbow', norm=matplotlib.colors.LogNorm())
    fig3.colorbar(im, ax=ax3d)

    ax3a.set_title('$m_j$')
    ax3a.set_xlabel('tracer index')
    ax3a.set_ylabel('swarm index')
    ax3b.set_title('$m_k$')
    ax3b.set_xlabel('tracer index')
    ax3b.set_ylabel('swarm index')
    ax3c.set_title('$M_j$')
    ax3c.set_xlabel('tracer index')
    ax3c.set_ylabel('swarm index')
    ax3d.set_title('$M_k$')
    ax3d.set_xlabel('tracer index')
    ax3d.set_ylabel('swarm index')

    fig3.savefig(filename)



def plot_constant_kernel_tests(base_filename, ext, cmd_args, benchmark=False):

    import planets.simulation
    if not benchmark:
        import planets.plot

    cmd_args_def = planets.simulation.params.copy(src=cmd_args, filter_pred=tools.parameters.FilterPredicates.not_None)
    seed = cmd_args_def.simulation.random_seed

    from numpy.random import default_rng
    rng = default_rng(seed=seed)

    dirpath, basename = os.path.split(base_filename)

    method_labels = {
        'rpmc': 'bucketing',
        'rpmc-traditional': 'traditional'
    }
    n = cmd_args.n
    M = 1.e+20
    Λ = 1.e-11
    args = planets.simulation.params.make_namespace()
    args.simulation.method = cmd_args.method
    args.simulation.effects = 'collisions'
    args.collisions.kernel = 'constant'
    args.collisions.constant_collision_rate = Λ
    args.simulation.mass_growth_factor = 0.05
    args.simulation.particle_regime_threshold = 1
    args.simulation.m_bins_per_decade = 2

    m0 = 1.
    args.planetesimal.m = m0
    args.planetesimal.M = M

    args.simulation.nPlt = n
    args.simulation.nPltR = (cmd_args.Nth - 1)*n
    args.simulation.nE   = 0
    args.simulation.tMinLog = 1
    args.simulation.NSteps = 256 if cmd_args.log or not benchmark else 4

    args.ring.r = 1.
    args.ring.Δr = 0.

    args.simulation.random_seed = seed

    # Select a log-spaced handful of snapshots for the canonical comparison plot.
    if cmd_args.plot or not benchmark:
        t_few = t_few = [1.e-2, 1.e+0, 1.e+2, 1.e+4, 1.e+6]
        args.simulation.T = t_few[-1]
    else:
        t_few = []
        args.simulation.T = 16

    with planets.simulation.run(args=args, rng=rng, log=cmd_args.log, timesteps={ 'few': t_few }) as sim_data:

        if cmd_args.plot:
            _, data_by_time = sim_data.snapshots_by_time()
            for t in t_few:
                data = data_by_time(t)
                weights = data.M.to_numpy()
                mask = weights > 0
                weights = weights[mask]
                print('t={}: {}/{} active tracers'.format(t, len(weights), len(data.M.to_numpy())))

            analytical = planets.plot.analytical_solution_constant_kernel(
                m0=m0, M=M,
                collision_rate=args.collisions.constant_collision_rate)

            fig1, ax1a = plt.subplots(1, 1, sharex=False, sharey=False, figsize=[5, 4])
            planets.plot.plot_snapshots(sim_data, ax1a, analytical=analytical, timescale='log', xrange=1., yrange=1.e+4)
            ax1a.set_xlabel('mass (dimensionless)')
            ax1a.set_ylabel('mass density $m^2 f(m)$')
            # 1.e-2..1.e+16, 1.e+14..1.e+18
            ax1a.text(2.e+13, 1.4e+21, 'constant kernel\n$n = {:_}$'.format(n).replace('_', r'\,'), size=10, ha="center", va="bottom",
                bbox=dict(boxstyle="round", ec=(0.2, 0.2, 0.2), fc=(0.95, 0.95, 0.95)))
            fig1.set_tight_layout(True)
            #ax1a.set_xlim(1.e-2, 1.e+16)
            fig1.savefig(os.path.join(dirpath, '{} method={} n={}{}'.format(basename, method_labels[cmd_args.method], n, ext).replace(' ', '_')))

            #_plot_kernel_snapshots(sim_data, base_filename=base_filename + ' linear kernel', ext=ext)

        if cmd_args.log and cmd_args.method == 'rpmc':
            temp = np.array([0], dtype=np.intp)
            params = np.array([0], dtype=np.intp)
            sim_data.sim.inspect(dst=temp, quantity='discrete-operator-0/statistics', params=params)

        if cmd_args.plot and cmd_args.log and cmd_args.method == 'rpmc':
            plot_interaction_rates(base_filename=base_filename, ext=ext, cmd_args=cmd_args, sim_data=sim_data, t=t_few[-1])



def plot_linear_kernel_tests(base_filename, ext, cmd_args, benchmark=False):
    from scipy.special import lambertw

    import planets.simulation
    if not benchmark:
        import planets.plot

    cmd_args_def = planets.simulation.params.copy(src=cmd_args, filter_pred=tools.parameters.FilterPredicates.not_None)
    seed = cmd_args_def.simulation.random_seed

    from numpy.random import default_rng
    rng = default_rng(seed=seed)

    dirpath, basename = os.path.split(base_filename)

    method_labels = {
        'rpmc': 'bucketing',
        'rpmc-traditional': 'traditional'
    }
    n = cmd_args.n
    M = 1.e+25
    args = planets.simulation.params.make_namespace()
    args.simulation.method = cmd_args.method
    args.simulation.mass_growth_factor = 0.05
    args.simulation.particle_regime_threshold = 1
    args.simulation.m_bins_per_decade = 1

    # Permit overriding some simulation parameters with command-line arguments.
    planets.simulation.params.copy(dest=args, src=cmd_args, filter_pred=tools.parameters.FilterPredicates.not_None)
    planets.simulation.fill_implicit_args(args)
    
    args.simulation.effects = 'collisions'
    args.collisions.kernel = 'linear'
    args.collisions.linear_collision_rate_coefficient = 1/M

    m0avg = 1.
    args.planetesimal.m = -m0avg*(1 + np.real(lambertw(-rng.uniform(size=n)/np.exp(1), k=-1)))
    args.planetesimal.M = M
    m0 = np.min(args.planetesimal.m)

    args.simulation.nPlt = n
    args.simulation.nPltR = (cmd_args.Nth - 1)*n
    args.simulation.nE   = 0
    args.simulation.tMinLog = 1
    args.simulation.NSteps = 256 if cmd_args.log or not benchmark else 4


    args.simulation.random_seed = seed

    # Select a linearly spaced handful of snapshots for the canonical comparison plot.
    if cmd_args.plot or not benchmark:
        t_few = [0, 4, 8, 12, 16]
        #t_few = [0, 1, 2, 3, 4]
        #t_few = [0, 4]
        args.simulation.T = t_few[-1]
    else:
        t_few = []
        args.simulation.T = 16

    with planets.simulation.run(args=args, rng=rng, log=cmd_args.log, timesteps={ 'few': t_few }) as sim_data:

        if cmd_args.plot:
            _, data_by_time = sim_data.snapshots_by_time()
            for t in t_few:
                data = data_by_time(t)
                weights = data.M.to_numpy()
                mask = weights > 0
                weights = weights[mask]
                print('t={}: {}/{} active tracers'.format(t, len(weights), len(data.M.to_numpy())))

            analytical = planets.plot.analytical_solution_linear_kernel(
                m0avg=m0avg, m0=m0, M=M,
                collision_rate_coefficient=args.collisions.linear_collision_rate_coefficient)

            fig1, ax1a = plt.subplots(1, 1, sharex=False, sharey=False, figsize=[5, 4])
            planets.plot.plot_snapshots(sim_data, ax1a, analytical=analytical, timescale='linear', xrange=1., yrange=1.e+4)
            ax1a.set_xlabel('mass (dimensionless)')
            ax1a.set_ylabel('mass density $m^2 f(m)$')
            # 1.e-2..1.e+16, 1.e+14..1.e+18
            ax1a.text(2.e+13, 1.4e+21, 'linear kernel\n$n = {:_}$'.format(n).replace('_', r'\,'), size=10, ha="center", va="bottom",
                bbox=dict(boxstyle="round", ec=(0.2, 0.2, 0.2), fc=(0.95, 0.95, 0.95)))
            fig1.set_tight_layout(True)
            ax1a.set_xlim(1.e-2, 1.e+16)
            fig1.savefig(os.path.join(dirpath, '{} method={} n={}{}'.format(basename, method_labels[cmd_args.method], n, ext).replace(' ', '_')))

            #_plot_kernel_snapshots(sim_data, base_filename=base_filename + ' linear kernel', ext=ext)

        if (cmd_args.log or cmd_args.report) and cmd_args.method == 'rpmc':
            temp = np.array([0], dtype=np.intp)
            params = np.array([0], dtype=np.intp)
            sim_data.sim.inspect(dst=temp, quantity='discrete-operator-0/statistics', params=params)

        if cmd_args.plot and cmd_args.log and cmd_args.method == 'rpmc':
            plot_interaction_rates(base_filename=base_filename, ext=ext, cmd_args=cmd_args, sim_data=sim_data, t=t_few[-1])
#            data = data_by_time(t_few[-1])
#            n_active = len(data.N[data.N > 0.9])
#            rates = np.zeros(shape=[n_active,n_active], dtype=float)
#            true_bucket_rates = np.zeros(shape=[n_active,n_active], dtype=float)
#            bucket_rates = np.zeros(shape=[n_active,n_active], dtype=float)
#            acc_probs = np.zeros(shape=[n_active,n_active], dtype=float)
#            sim_data.sim.inspect(dst=rates, quantity='discrete-operator-0/interaction-model-0/interaction rates', params=params)
#            sim_data.sim.inspect(dst=true_bucket_rates, quantity='discrete-operator-0/interaction-model-0/true particle bucket interaction rates', params=params)
#            sim_data.sim.inspect(dst=bucket_rates, quantity='discrete-operator-0/interaction-model-0/particle bucket interaction rates', params=params)
#            sim_data.sim.inspect(dst=acc_probs, quantity='discrete-operator-0/interaction-model-0/acceptance probabilities', params=params)
#
#            filename = os.path.join(dirpath, '{} interaction-rates n={} t={:g}{}'.format(basename, n, t, ext).replace(' ', '_'))
#            print("Writing plot '{}'.".format(filename))
#            fig2, ((ax2a, ax2b), (ax2c, ax2d)) = plt.subplots(2, 2, sharex=False, sharey=False, figsize=[10, 10])
#            rates_non0 = rates[rates > 0]
#            bucket_rates_non0 = bucket_rates[bucket_rates > 0]
#            true_bucket_rates_non0 = true_bucket_rates[true_bucket_rates > 0]
#            vmax = max(np.max(rates_non0) if len(rates_non0) > 0 else 1., np.max(bucket_rates_non0) if len(bucket_rates_non0) > 0 else 1., np.max(true_bucket_rates_non0) if len(true_bucket_rates_non0) > 0 else 1.)
#            vmin = vmax*1.e-8
#            im = ax2a.imshow(rates.T, interpolation='nearest', origin='lower',
#               extent=[0.5, n_active+0.5, 0.5, n_active+0.5],
#               cmap='rainbow', norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax))
#            fig2.colorbar(im, ax=ax2a)
#            im = ax2c.imshow(bucket_rates.T, interpolation='nearest', origin='lower',
#               extent=[0.5, n_active+0.5, 0.5, n_active+0.5],
#               cmap='rainbow', norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax))
#            fig2.colorbar(im, ax=ax2c)
#            im = ax2d.imshow(true_bucket_rates.T, interpolation='nearest', origin='lower',
#               extent=[0.5, n_active+0.5, 0.5, n_active+0.5],
#               cmap='rainbow', norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax))
#            fig2.colorbar(im, ax=ax2d)
#            im = ax2b.imshow(acc_probs.T, interpolation='nearest', origin='lower',
#               extent=[0.5, n_active+0.5, 0.5, n_active+0.5],
#               cmap='rainbow', norm=matplotlib.colors.LogNorm(vmin=1.e-8, vmax=1.))
#            fig2.colorbar(im, ax=ax2b)
#
#            ax2a.set_title(r'tracer–swarm interaction rates ($\mathrm{year}^{-1}$)')
#            ax2a.set_xlabel('tracer index')
#            ax2a.set_ylabel('swarm index')
#            ax2b.set_title(r'acceptance probabilities')
#            ax2b.set_xlabel('tracer index')
#            ax2b.set_ylabel('swarm index')
#            ax2c.set_title(r'bucket bounds of tracer–swarm interaction rates ($\mathrm{year}^{-1}$)')
#            ax2c.set_xlabel('tracer index')
#            ax2c.set_ylabel('swarm index')
#            ax2d.set_title(r'bucket bounds of tracer–swarm interaction rates ($\mathrm{year}^{-1}$)')
#            ax2d.set_xlabel('tracer index')
#            ax2d.set_ylabel('swarm index')
#
#            fig2.savefig(filename)
#
#            indices = np.zeros(shape=[n_active], dtype=np.intp)
#            sim_data.sim.inspect(dst=indices, quantity='discrete-operator-0/indices', params=params)
#            bms = np.repeat(data.m.to_numpy()[indices], n_active).reshape([n_active, n_active])
#            bMs = np.repeat(data.M.to_numpy()[indices], n_active).reshape([n_active, n_active])
#
#            filename = os.path.join(dirpath, '{} masses n={} t={:g}{}'.format(basename, n, t, ext).replace(' ', '_'))
#            print("Writing plot '{}'.".format(filename))
#            fig3, ((ax3a, ax3b), (ax3c, ax3d)) = plt.subplots(2, 2, sharex=False, sharey=False, figsize=[10, 10])
#            im = ax3a.imshow(bms.T, interpolation='nearest', origin='lower',
#               extent=[0.5, n_active+0.5, 0.5, n_active+0.5],
#               cmap='rainbow', norm=matplotlib.colors.LogNorm())
#            fig3.colorbar(im, ax=ax3a)
#            im = ax3b.imshow(bms, interpolation='nearest', origin='lower',
#               extent=[0.5, n_active+0.5, 0.5, n_active+0.5],
#               cmap='rainbow', norm=matplotlib.colors.LogNorm())
#            fig3.colorbar(im, ax=ax3b)
#            im = ax3c.imshow(bMs.T, interpolation='nearest', origin='lower',
#               extent=[0.5, n_active+0.5, 0.5, n_active+0.5],
#               cmap='rainbow', norm=matplotlib.colors.LogNorm())
#            fig3.colorbar(im, ax=ax3c)
#            im = ax3d.imshow(bMs, interpolation='nearest', origin='lower',
#               extent=[0.5, n_active+0.5, 0.5, n_active+0.5],
#               cmap='rainbow', norm=matplotlib.colors.LogNorm())
#            fig3.colorbar(im, ax=ax3d)
#
#            ax3a.set_title('$m_j$')
#            ax3a.set_xlabel('tracer index')
#            ax3a.set_ylabel('swarm index')
#            ax3b.set_title('$m_k$')
#            ax3b.set_xlabel('tracer index')
#            ax3b.set_ylabel('swarm index')
#            ax3c.set_title('$M_j$')
#            ax3c.set_xlabel('tracer index')
#            ax3c.set_ylabel('swarm index')
#            ax3d.set_title('$M_k$')
#            ax3d.set_xlabel('tracer index')
#            ax3d.set_ylabel('swarm index')
#
#            fig3.savefig(filename)


def plot_runaway_kernel_rates(base_filename, ext, cmd_args):
    from scipy.special import lambertw

    import planets.simulation
    import planets.plot

    cmd_args_def = planets.simulation.params.copy(src=cmd_args, filter_pred=tools.parameters.FilterPredicates.not_None)
    seed = cmd_args_def.simulation.random_seed

    from numpy.random import default_rng
    rng = default_rng(seed=seed)

    dirpath, basename = os.path.split(base_filename)

    method_labels = {
        'rpmc': 'bucketing',
        'rpmc-traditional': 'traditional'
    }

    Nx = 2
    #Ny = 128
    n = 2048
    #n = 4
    Nth = 10

    args = planets.simulation.params.make_namespace()
    args.simulation.random_seed = seed
    #M = 1.e+10
    M = 1.e+11
    args.collisions.runaway_collision_rate_coefficient = 10/M
    args.simulation.method = cmd_args.method
    args.simulation.effects = 'collisions'
    args.collisions.kernel = 'runaway'
    #args.collisions.runaway_critical_mass = 1.e+10
    args.collisions.runaway_critical_mass = 1.e+7
    args.simulation.mass_growth_factor = 0.05
    args.simulation.particle_regime_threshold = Nth
    args.simulation.m_bins_per_decade = 2

    m0avg = 1.
    args.planetesimal.m = -m0avg*(1 + np.real(lambertw(-rng.uniform(size=n)/np.exp(1), k=-1)))
    args.planetesimal.M = M
    m0 = np.min(args.planetesimal.m)

    args.simulation.nPlt = n
    args.simulation.nPltR = (Nth - 1)*n
    args.simulation.nE   = 0
    args.simulation.tMinLog = 1
    args.simulation.NSteps = 2*Nx

    #t_few = [0, 1, 8, 30, 50, 60, 70]
    #t_inspect = [15, 25, 30, 50]
    t_inspect = [8, 25]
    args.simulation.T = t_inspect[-1]

    ns_active = []
    interaction_rates = []
    bucket_interaction_rates = []
    acc_probs = []
    masses = []
    swarm_masses = []
    def inspect_callback(sim, state, i, t):
        params = np.array([0], dtype=float)
        dst0 = np.zeros(shape=[1], dtype=np.intp)
        sim.inspect(dst0, 'discrete-operator-0/num-active-particles', params)
        num_active = dst0[0]
        ns_active.append(num_active)
        dst1 = np.zeros(shape=[num_active, num_active], dtype=float)
        dst2 = np.zeros(shape=[num_active, num_active], dtype=float)
        dst3 = np.zeros(shape=[num_active, num_active], dtype=float)
        indices = np.zeros(shape=[num_active], dtype=np.intp)
        sim.inspect(dst1, 'discrete-operator-0/interaction-model-0/interaction rates', params)
        sim.inspect(dst2, 'discrete-operator-0/interaction-model-0/particle bucket interaction rates', params)
        sim.inspect(dst3, 'discrete-operator-0/interaction-model-0/acceptance probabilities', params)
        sim.inspect(indices, 'discrete-operator-0/indices', params)
        interaction_rates.append(dst1)
        bucket_interaction_rates.append(dst2)
        acc_probs.append(dst3)
        m = state.m.to_numpy(copy=True)[indices]
        masses.append(m)
        M = state.M.to_numpy(copy=True)[indices]
        swarm_masses.append(M)

    with planets.simulation.run(args=args, rng=rng, log=cmd_args.log, timesteps={ 'inspect': t_inspect },
                                inspect_callback=inspect_callback) as sim_data:

        temp = np.array([0], dtype=np.intp)
        params = np.array([0], dtype=np.intp)
        sim_data.sim.inspect(dst=temp, quantity='discrete-operator-0/statistics', params=params)

    scale = 9/18
    width_ratios=(0.5, 8, 8, 0.5, 0.5)
    #              0 1     3 4     6 7     9 10
    height_ratios=(0.5,8, 1, 0.5,8)
    #height_ratios=(1,8, 1, 1,8, 1, 1,8, 1, 1, 8 )
    fig = plt.figure(figsize=(np.sum(width_ratios)*scale, np.sum(height_ratios)*scale))
    gs = fig.add_gridspec(nrows=len(height_ratios), ncols=len(width_ratios), width_ratios=width_ratios, height_ratios=height_ratios,
                          left=0.08, right=0.92, bottom=0.08, top=0.92,
                          wspace=0.02, hspace=0.02)
    ax1a = fig.add_subplot(gs[1, 1])
    ax1a_histx = fig.add_subplot(gs[0, 1], sharex=ax1a)
    ax1a_histy = fig.add_subplot(gs[1, 0], sharey=ax1a)
    ax1b = fig.add_subplot(gs[1, 2])
    ax1b_histx = fig.add_subplot(gs[0, 2], sharex=ax1b)
    ax1cb = fig.add_subplot(gs[1, 4])
    ax2a = fig.add_subplot(gs[4, 1])
    ax2a_histx = fig.add_subplot(gs[3, 1], sharex=ax2a)
    ax2a_histy = fig.add_subplot(gs[4, 0], sharey=ax2a)
    ax2b = fig.add_subplot(gs[4, 2])
    ax2b_histx = fig.add_subplot(gs[3, 2], sharex=ax2b)
    ax2cb = fig.add_subplot(gs[4, 4])
    #ax3a = fig.add_subplot(gs[7, 1])
    #ax3a_histx = fig.add_subplot(gs[6, 1], sharex=ax3a)
    #ax3a_histy = fig.add_subplot(gs[7, 0], sharey=ax3a)
    #ax3b = fig.add_subplot(gs[7, 2])
    #ax3b_histx = fig.add_subplot(gs[6, 2], sharex=ax3b)
    #ax3cb = fig.add_subplot(gs[7, 4])
    #ax4a = fig.add_subplot(gs[10, 1])
    #ax4a_histx = fig.add_subplot(gs[9, 1], sharex=ax4a)
    #ax4a_histy = fig.add_subplot(gs[10, 0], sharey=ax4a)
    #ax4b = fig.add_subplot(gs[10, 2])
    #ax4b_histx = fig.add_subplot(gs[9, 2], sharex=ax4b)
    #ax4cb = fig.add_subplot(gs[10, 4])
    axs = [
        (ax1a, ax1a_histx, ax1a_histy, ax1b, ax1b_histx, ax1cb),
        (ax2a, ax2a_histx, ax2a_histy, ax2b, ax2b_histx, ax2cb),
        #(ax3a, ax3a_histx, ax3a_histy, ax3b, ax3b_histx, ax3cb),
        #(ax4a, ax4a_histx, ax4a_histy, ax4b, ax4b_histx, ax4cb)
    ]

    filename = os.path.join(dirpath, '{} interaction-rates n={}{}'.format(basename, n, ext).replace(' ', '_'))
    print("Writing plot '{}'.".format(filename))
    vmin = np.inf
    vmax = -np.inf
    mmin = np.inf
    mmax = -np.inf
    for rates, bucket_rates, ms, in zip(interaction_rates, bucket_interaction_rates, masses):
        rates_non0 = rates[rates > 0]
        bucket_rates_non0 = bucket_rates[bucket_rates > 0]
        vmin = min(vmin, np.min(rates_non0) if len(rates_non0) > 0 else 1., np.min(bucket_rates_non0) if len(bucket_rates_non0) > 0 else 1.)
        vmax = max(vmax, np.max(rates_non0) if len(rates_non0) > 0 else 1., np.max(bucket_rates_non0) if len(bucket_rates_non0) > 0 else 1.)
        #vmin = vmax*1.e-8
        mmin = min(mmin, np.min(ms))
        mmax = max(mmax, np.max(ms))
    vmin = 4.e-7
    vmax = 4.e-1

    for t, (axa, axa_histx, axa_histy, axb, axb_histx, axcb), n_active, rates, bucket_rates, probs, ms \
            in zip(t_inspect, axs, ns_active, interaction_rates, bucket_interaction_rates, acc_probs, masses):
        im_rate = axa.imshow(rates.T, interpolation='nearest', origin='lower',
           extent=[0.5, n_active+0.5, 0.5, n_active+0.5],
           cmap='plasma', norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax))
        axa.tick_params(axis="y", left=False, labelleft=False)
        im_rate = axb.imshow(bucket_rates.T, interpolation='nearest', origin='lower',
           extent=[0.5, n_active+0.5, 0.5, n_active+0.5],
           cmap='plasma', norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax))
        axb.tick_params(axis="y", left=False, labelleft=False)
        ms_sq = ms[..., np.newaxis] * np.array([1., 1.])[np.newaxis, ...]
        im_mass = axa_histx.imshow(ms_sq.T, interpolation='nearest', origin='lower',
           extent=[0.5, n_active+0.5, 0, 1], aspect='auto',
           cmap='bone_r', norm=matplotlib.colors.LogNorm(vmin=mmin, vmax=mmax))
        axa_histx.tick_params(axis="x", bottom=False, labelbottom=False)
        axa_histx.tick_params(axis="y", left=False, labelleft=False)
        im_mass = axb_histx.imshow(ms_sq.T, interpolation='nearest', origin='lower',
           extent=[0.5, n_active+0.5, 0, 1], aspect='auto',
           cmap='bone_r', norm=matplotlib.colors.LogNorm(vmin=mmin, vmax=mmax))
        axb_histx.tick_params(axis="x", bottom=False, labelbottom=False)
        axb_histx.tick_params(axis="y", left=False, labelleft=False)
        im_mass = axa_histy.imshow(ms_sq, interpolation='nearest', origin='lower',
           extent=[0, 1, 0.5, n_active+0.5], aspect='auto',
           cmap='bone_r', norm=matplotlib.colors.LogNorm(vmin=mmin, vmax=mmax))
        axa_histy.tick_params(axis="x", bottom=False, labelbottom=False)
        axa.text(n_active*0.04, n_active*0.96, 'interaction rate\n$\\lambda_{jk}$', size=10, ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.5", ec=(0.2, 0.2, 0.2), fc=(0.95, 0.95, 0.95)))
        axb.text(n_active*0.04, n_active*0.96, 'interaction rate bound\n$\\lambda_{\\mathrm{JK}}^+$', size=10, ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.5", ec=(0.2, 0.2, 0.2), fc=(0.95, 0.95, 0.95)))
        axb.text(n_active*0.96, n_active*0.04, '$t = {}$'.format(t), size=10, ha="right", va="bottom",
            bbox=dict(boxstyle="round", ec=(0.2, 0.2, 0.2), fc=(0.95, 0.95, 0.95)))

        #axa.set_title(r'RP–swarm interaction rates')
        axa.set_xlabel('RP index $j$')
        axa_histy.set_ylabel('swarm index $k$')
        #axb.set_title(r'acceptance probabilities')
        #axb.set_xlabel('tracer index')
        #axb.set_ylabel('swarm index')
        #axb.set_title(r'bucket bounds of RP–swarm interaction rates')
        axb.set_xlabel('RP index $j$')
        #axb.set_ylabel('swarm index')
        
    cb = fig.colorbar(im_mass, cax=ax1cb)
    cb.set_label('particle mass')
    cb = fig.colorbar(im_rate, cax=ax2cb)
    cb.set_label(r"interaction rate")

    #fig.set_tight_layout(True)
    fig.savefig(filename)
    plt.close(fig)


def plot_runaway_kernel_tests(base_filename, ext, cmd_args, convergence_mode='converge', benchmark=False):
    from scipy.special import lambertw

    import planets.simulation
    import planets.plot

    cmd_args_def = planets.simulation.params.copy(src=cmd_args, filter_pred=tools.parameters.FilterPredicates.not_None)
    seed = cmd_args_def.simulation.random_seed

    from numpy.random import default_rng
    rng = default_rng(seed=seed)

    dirpath, basename = os.path.split(base_filename)

    method_labels = {
        'rpmc': 'bucketing',
        'rpmc-traditional': 'traditional'
    }

    #Ny = 128
    n = cmd_args.n
    Nth = cmd_args.Nth
    num_repetitions = 10 if not benchmark else 1
    Nx = 128 if not benchmark else 2
    #Ny = 128
    Ny = projection.num_histogram_bins(2*n)

    img = np.zeros(shape=[Ny, Nx])
    mmaxnum = np.zeros(shape=[Nx], dtype=int)
    mmaxsum = np.zeros(shape=[Nx])
    mmaxsqsum = np.zeros(shape=[Nx])
    for repetition in range(num_repetitions):
        args = planets.simulation.params.make_namespace()
        args.simulation.random_seed = seed
        seed += 1
        if convergence_mode == 'converge':
            #M = 1.e+10
            M = 1.e+11
        else:
            M = 1.e+12
        args.collisions.runaway_collision_rate_coefficient = 10/M
        args.simulation.method = cmd_args.method
        args.simulation.effects = 'collisions'
        args.collisions.kernel = 'runaway'
        #args.collisions.runaway_critical_mass = 1.e+10
        args.collisions.runaway_critical_mass = 1.e+7
        args.simulation.mass_growth_factor = 0.05
        args.simulation.particle_regime_threshold = Nth
        args.simulation.m_bins_per_decade = 2

        m0avg = 1.
        args.planetesimal.m = -m0avg*(1 + np.real(lambertw(-rng.uniform(size=n)/np.exp(1), k=-1)))
        args.planetesimal.M = M
        m0 = np.min(args.planetesimal.m)

        args.simulation.nPlt = n
        args.simulation.nPltR = (Nth - 1)*n
        args.simulation.nE   = 0
        args.simulation.tMinLog = 1
        args.simulation.NSteps = 2*Nx

        args.ring.r = 1.
        args.ring.Δr = 0.

        #args.simulation.random_seed = rng.integers(np.iinfo(np.uint32).max)

        # Select a linearly spaced handful of snapshots for the canonical comparison plot.
        #t_few = [0, 4, 8, 11, 11.5, 11.6, 11.65, 11.66, 11.67, 11.68]
        if convergence_mode == 'converge':
            #t_few = [0, 4, 8, 11, 11.5, 12, 14, 16, 18, 20, 22, 24, 26]
            #t_few = [0, 1, 8, 16, 32, 48, 56, 60, 120]
            t_few = [0, 1, 8, 30, 50, 60, 70]
            #t_few = [0, 1]
            #t_few = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            #t_few = [0, 0.2, 0.4, 0.6, 0.8, 1]
        else:
            t_few = [0, 4, 8, 11, 11.5]
        args.simulation.T = t_few[-1]
        if not cmd_args.plot:
            t_few = []

        if cmd_args.report and repetition == 0:
            report_args(args)
        if cmd_args.dry_run:
            return

        if cmd_args.plot:
            histograms_edges = []
            mmaxs = np.zeros(shape=[args.simulation.NSteps//2])
            mavgs = np.zeros(shape=[args.simulation.NSteps//2])
            mstds = np.zeros(shape=[args.simulation.NSteps//2])
        with planets.simulation.run(args=args, rng=rng, log=cmd_args.log, timesteps={ 'few': t_few }) as sim_data:

            #if cmd_args.log or not benchmark:
            temp = np.array([0], dtype=np.intp)
            params = np.array([0], dtype=np.intp)
            sim_data.sim.inspect(dst=temp, quantity='discrete-operator-0/statistics', params=params)

            if cmd_args.plot and cmd_args.log and repetition == 0:
                plot_interaction_rates(base_filename=base_filename, ext=ext, cmd_args=cmd_args, sim_data=sim_data, t=t_few[-1])

            if cmd_args.plot:
                _, data_by_time = sim_data.snapshots_by_time()

                ts_linear = sim_data.timesteps()['linear']
                data_by_ts_linear = []
                for i, t in enumerate(ts_linear):
                    data = data_by_time(t)
                    ms = data.m.to_numpy()
                    Ms = data.M.to_numpy()
                    Ns = data.N.to_numpy()
                    mask = Ms > 0
                    mms = ms[mask]
                    mMs = Ms[mask]
                    mNs = Ns[mask]
                    N = np.sum(mNs)
                    logms = np.log10(mms)
                    mean = np.sum(mNs*logms)/N
                    var = np.sum(mNs*np.square(logms - mean))/N
                    mmaxs[i] = np.max(mms)
                    mavgs[i] = mean
                    mstds[i] = np.sqrt(var)
                    imax = np.argmax(ms)
                    mmax = ms[imax]
                    if mmax > 0:
                        mmaxnum[i] += 1
                        mmaxsum[i] += mmax
                        mmaxsqsum[i] += mmax**2
                        Ms[imax] -= mmax
                        Ns[imax] -= 1
                        if Ns[imax] < 0.1:
                            ms[imax] = 0
                            Ms[imax] = 0
                            Ns[imax] = 0
                    data_by_ts_linear.append(planets.plot.positive_finite(ms, Ms))

                for t in t_few:
                    #if t == 0.7:
                    #    _plot_kernel_snapshots(sim_data, base_filename=base_filename + ' constant kernel', ext=ext)
                    data = data_by_time(t)
                    masses = data.m.to_numpy()
                    weights = data.M.to_numpy()
                    mask = weights > 0
                    masses = masses[mask]
                    weights = weights[mask]
                    #print('masses', masses, 'weights', weights)
                    histograms_edges.append(projection.make_log_histogram(masses, weights=weights, nbin_factor=1.))
                    print('t={}: {} active tracers'.format(t, len(weights)))

                #Y_w = lambda data: (data.m, data.M)
                #data_by_ts_linear = [planets.plot.positive_finite(*Y_w(data_by_time(t))) for t in ts_linear]
                hist, edges = projection.project_log_histograms(data_by_ts_linear, Ny, vmin=1., vmax=M)
                #hist, edges = projection.project_log_histograms(data_by_ts_linear, Ny, vmin=1.e-2, vmax=M*1.e-5)
                bins = np.sqrt(edges[1:]*edges[:-1])
                np.add(img, hist, out=img)

    if cmd_args.plot:
        cmap = plt.cm.Set1(np.linspace(0, 1, num=9))
        Ym0, Ym1 = bins[0], bins[-1]
        filename = os.path.join(dirpath, '{} masses n={} Nth={} method={}{}'.format(basename, n, Nth, method_labels[cmd_args.method], ext).replace(' ', '_'))
        print("Writing plot '{}'.".format(filename))
        fig, ax = plt.subplots(1, 1, sharex=False, sharey=False, figsize=[5, 4])
        im = ax.imshow(img/num_repetitions, interpolation='nearest', origin='lower',
                       extent=[ts_linear[0], ts_linear[-1], np.log10(Ym0), np.log10(Ym1)],
                       cmap='Blues', norm=matplotlib.colors.LogNorm(vmin=1.e-6*M, vmax=M))
        mmaxavg = mmaxsum/mmaxnum
        mmaxstd = np.sqrt(mmaxsqsum/mmaxnum - mmaxavg**2)
        #ax.text(3, np.log10(M), r"$n = {}$, $N_{{\mathrm{{th}}}} = {}$".format('{:_}'.format(n).replace('_', r'\,'), Nth), size=10, ha="left", va="top",
        #    bbox=dict(boxstyle="round", ec=(0.2, 0.2, 0.2), fc=(0.95, 0.95, 0.95)))
        ax.text(3, np.log10(M), "$n = {:_}$\n$N^{{\\mathrm{{th}}}} = {:_}$".format(n, Nth).replace('_', r'\,').replace('^', '_'),
            size=10, ha="left", va="top",
            bbox=dict(boxstyle="round", ec=(0.2, 0.2, 0.2), fc=(0.95, 0.95, 0.95)))
        ax.plot(ts_linear, np.log10(mmaxavg), color=cmap[0])
        if num_repetitions >= 3:
            ax.plot(ts_linear, np.log10(mmaxavg - mmaxstd), color=cmap[0], linestyle='--', linewidth=0.75)
            ax.plot(ts_linear, np.log10(mmaxavg + mmaxstd), color=cmap[0], linestyle='--', linewidth=0.75)
            ax.fill_between(ts_linear, np.log10(mmaxavg - mmaxstd), np.log10(mmaxavg + mmaxstd), color=cmap[0], alpha=0.25)
        cb = fig.colorbar(im, ax=ax, pad=0.1)
        cb.set_label('mass density $m^2 f(m)$')
        ax.axhline(np.log10(args.collisions.runaway_critical_mass), linestyle='dotted', color='red')
        ax.axhline(np.log10(M/(n*Nth)), linestyle='dashed', color='orange')
        ax.set_aspect(ts_linear[-1] / np.log10(Ym1/Ym0))
        ax.set_ylim(np.log10(1), np.log10(M*2))
        #ax.set_ylim(np.log10(1.e-2), np.log10(M*2*1.e-5))
        planets.plot.set_fake_log_axes(ax=ax, axes='y', minor_ticks=True)
        #ax.grid(which='major', axis='both')
        #ax.grid(which='minor', axis='y', linestyle='--', color='lightgray')
        ax.set_xlabel('time (dimensionless)')
        ax.set_ylabel(r'mass (dimensionless)')
        fig.set_tight_layout(True)
        fig.savefig(filename)
        plt.close(fig)

        #print(histograms_edges[-1])

        fig1, ax1a = plt.subplots(1, 1, sharex=False, sharey=False, figsize=[5, 4])
        cmap = plt.cm.tab10(np.linspace(0, 1, num=10))
        for i, ((hist, edges), color) in enumerate(zip(histograms_edges, cmap)):
            #bins = np.sqrt(edges[1:]*edges[:-1])
            #ax1a.fill_between(bins, 0., hist, step='mid', color=color, alpha=0.5, label=R'$t = {}$'.format(t_few[i]))
            #ax1a.step(bins, hist, where='mid', color=color, alpha=0.5, label=R'$t = {}$'.format(t_few[i]))
            ax1a.stairs(hist, edges, color=color, label=R'$t = {}$'.format(t_few[i]), fill=True, alpha=0.1)
            ax1a.stairs(hist, edges, color=color, facecolor=color)
            #ax1a.plot(bins, hist, color=color, label=R'$t = {}$'.format(t_few[i]))
        ax1a.text(5.e+10, 1.3*M*1.e-4, r"$n = {:_}$".format(n).replace('_', r'\,'), size=10, ha="right", va="bottom",
            bbox=dict(boxstyle="round", ec=(0.2, 0.2, 0.2), fc=(0.95, 0.95, 0.95)))
        ax1a.legend(loc='lower left')
        ax1a.set_xlabel('mass (dimensionless)')
        ax1a.set_ylabel('mass density $m^2 f(m)$')
        ax1a.set_xscale('log')
        ax1a.set_yscale('log')
        fig1.set_tight_layout(True)
        ##ax1a.set_xlim(1.e-2, M*1.e+1)
        #ax1a.set_xlim(1.e-2, M*1.e-3)
        #ax1a.set_ylim(M*1.e-4, M)
        ax1a.grid(which='major', axis='both')
        ax1a.grid(which='minor', axis='both', linestyle='--', color='lightgray')
        fig1.savefig(os.path.join(dirpath, '{} method={} n={} Nth={}{}'.format(basename, method_labels[cmd_args.method], n, Nth, ext).replace(' ', '_')))


def linreg_1_0(x1,y1):
    a = y1/x1
    return np.array([a])

def linreg_2_0(x1,x2,y1,y2):
    a = y1/(x1*(x1 - x2)) + y2/(x2*(x2 - x1))
    b = -y1*x2/(x1*(x1 - x2)) - y2*x1/(x2*(x2 - x1))
    return np.array([a, b])

def linregmin_n_0(xs, ys, exp):
    if exp < 1:
        as_, = linreg_1_0(xs**exp, ys)
        amin = np.min(as_)
        return lambda x: amin*x**exp
    if exp == 1:
        as_, = linreg_1_0(xs, ys)
        amin = np.min(as_)
        return lambda x: amin*x
    elif exp == 2:
        as_, bs = linreg_2_0(xs[:-1], xs[1:], ys[:-1], ys[1:])
        iamin = np.argmin(as_)
        amin, bmin = as_[iamin], bs[iamin]
        return lambda x: amin*x**2 + bmin*x
    else:
        raise RuntimeError('unsupported exponent')


def plot_linear_kernel_perf(base_filename, ext, cmd_args):
    import pandas as pd
    
    dirpath, basename = os.path.split(base_filename)
    os.makedirs(dirpath, exist_ok=True)

    # Xeon E3-1585 v5
    memL1 = 32.*1024
    memL2c = 256.*1024
    memL2s = 1.*1024*1024
    memL3c = 2.*1024*1024
    memL3s = 8.*1024*1024
    memL4 = 128.*1024*1024

    # mp-media* machines
    memDRAM = 64.*1024*1024*1024

    df = pd.read_csv('data/perf/linear-kernel.tsv', sep='\t')
    dfa = df.groupby(['method', 'n'], as_index=False).agg({ 'time': ['mean', 'std'] })
    #methods = np.unique(dfa.method)
    #print("methods: ", methods)
    methods = ['rpmc-traditional', 'rpmc']

    filename = os.path.join(dirpath, '{}{}'.format(basename, ext).replace(' ', '_'))
    print("Writing plot '{}'.".format(filename))
    fig, ax = plt.subplots(1, 1, sharex=False, sharey=False, figsize=[5, 4])
    labels = {
        'rpmc-traditional': 'traditional scheme',
        'rpmc': 'bucketing scheme'
    }
    scale_exp_labels = {
        'rpmc-traditional': (2, r'$\sim n^2$ cost model'),
        'rpmc': (1, r'$\sim n$ cost model')
    }
    cmap = plt.cm.tab10(np.linspace(0, 1, num=10))
    cmap2 = plt.cm.tab10_r(np.linspace(0, 1, num=10))
    #n0b = np.min(dfa.n)
    #n1b = np.max(dfa.n)
    n0b = 1.e+1
    n1b = 3.e+5
    nrefs = np.logspace(np.log2(n0b), np.log2(n1b), num=512, base=2)
    for method, color in zip(methods, cmap):
        dfaf = dfa[dfa.method == method]
        ns = dfaf.n.to_numpy()
        tmeans, tstds = dfaf.time.to_numpy().T
        #ax.errorbar(ns, tmeans, yerr=tstds, marker='o', color=color, label=labels[method])
        attenuated_color = plot.adjust_lightness(color=color, amount=0.3)
        ax.scatter(ns, tmeans, color=attenuated_color, marker='o')
        ax.errorbar(ns, tmeans, yerr=tstds, color=color, label=labels[method])
        scale_exp, reflabel = scale_exp_labels[method]
        scalef = linregmin_n_0(ns, tmeans, scale_exp)
        trefs = scalef(nrefs)
        ax.plot(nrefs, trefs, linestyle='--', alpha=0.6, color=color, label=reflabel)
    memrefs_traditional = nrefs**2*8  # simplistic memory model
    memrefs_bucketing = nrefs*16  # simplistic memory model
    nmemf = lambda memrefs, mem: nrefs[np.argwhere(memrefs > mem)[0]]
    nL1t = nmemf(memrefs_traditional, memL1)
    nL2ct = nmemf(memrefs_traditional, memL2c)
    nL3st = nmemf(memrefs_traditional, memL3s)
    nDRAMt = nmemf(memrefs_traditional, memDRAM)
    nL1b = nmemf(memrefs_bucketing, memL1)
    nL2cb = nmemf(memrefs_bucketing, memL2c)
    ax.axvline(nL1t,   linestyle='dotted', linewidth=0.6, color=cmap[0], alpha=0.6)
    ax.axvline(nL2ct,  linestyle='dotted', linewidth=0.9, color=cmap[0], alpha=0.6)
    ax.axvline(nL3st,  linestyle='dotted', linewidth=1.2, color=cmap[0], alpha=0.6)
    ax.axvline(nDRAMt, linestyle='dotted', linewidth=3,   color=cmap[0], alpha=0.6)
    ax.axvline(nL1b,   linestyle='dotted', linewidth=0.6, color=cmap[1], alpha=0.6)
    ax.axvline(nL2cb,  linestyle='dotted', linewidth=0.9, color=cmap[1], alpha=0.6)
    ax.text(0.97*nL1t,   4.e-2, "L1\n$32\\,\\mathrm{KiB}$",   size=7.5, ha="right", va="center", color=cmap[0], alpha=0.6)
    ax.text(0.97*nL2ct,  3.e-1, "L2c\n$256\\,\\mathrm{KiB}$", size=7.5, ha="right", va="center", color=cmap[0], alpha=0.6)
    ax.text(0.97*nL3st,  1.e+1, "L3s\n$8\\,\\mathrm{MiB}$",   size=7.5, ha="right", va="center", color=cmap[0], alpha=0.6)
    ax.text(0.94*nDRAMt, 3.e+2, "DRAM\n$64\\,\\mathrm{GiB}$", size=7.5, ha="right", va="center", color=cmap[0], alpha=0.6)
    ax.text(0.97*nL1b,   7.e-1, "L1\n$32\\,\\mathrm{KiB}$",   size=7.5, ha="right", va="center", color=cmap[1], alpha=0.6)
    ax.text(0.97*nL2cb,  7.e+0, "L2c\n$256\\,\\mathrm{KiB}$", size=7.5, ha="right", va="center", color=cmap[1], alpha=0.6)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'number of entities $n$')
    ax.set_ylabel(r'time ($\mathrm{s}$)')
    #ax.axhline(np.log10(mth), label=r'threshold mass $m_{\mathrm{th}}$', linestyle='--', color='black')
    #ax.axvline(np.log10(mth), linestyle='--', color='black')
    #ax.axhline(np.log10(mcrit), label=r'critical mass $m_{\mathrm{crit}} = M/(n N_{\mathrm{th}})$', linestyle='dotted', color='blue')
    #ax.set_ylim(7.e-3, 1.5e+3)
    ax.set_xlim(2.e+1, 2.e+5)
    ax.set_ylim(2.e-3, 6.e+2)
    ax.legend(loc='lower right')
    ax.text(2.5e+1, 4.e+2, r"linear kernel", size=10, ha="left", va="top",
            bbox=dict(boxstyle="round", ec=(0.2, 0.2, 0.2), fc=(0.95, 0.95, 0.95)))
    #ax.legend(loc='upper left')
    #ax.text(7.5e+4, 6.e-3, r"linear kernel", size=10, ha="right", va="bottom",
    #        bbox=dict(boxstyle="round", ec=(0.2, 0.2, 0.2), fc=(0.95, 0.95, 0.95)))
    fig.set_tight_layout(True)
    fig.savefig(filename)
    plt.close(fig)


def plot_runaway_kernel_perf(base_filename, ext, cmd_args):
    import pandas as pd

    # Xeon E3-1585 v5
    memL1 = 32.*1024
    memL2c = 256.*1024
    memL2s = 1.*1024*1024
    memL3c = 2.*1024*1024
    memL3s = 8.*1024*1024
    memL4 = 128.*1024*1024

    # mp-media* machines
    memDRAM = 64.*1024*1024*1024

    dirpath, basename = os.path.split(base_filename)
    os.makedirs(dirpath, exist_ok=True)

    df = pd.read_csv('data/perf/runaway-kernel.tsv', sep='\t')
    dfa = df[df.Nth == 10].groupby(['method', 'n'], as_index=False).agg({
        'time': ['mean', 'std'],
        'num-events': ['mean', 'std'],
        'min-num-buckets': ['mean', 'std'],
        'max-num-buckets': ['mean', 'std'],
        'avg-num-buckets': ['mean', 'std'],
        'min-num-active-particles': ['mean', 'std'],
        'max-num-active-particles': ['mean', 'std'],
        'avg-num-active-particles': ['mean', 'std']
    })
    #methods = np.unique(dfa.method)
    #print("methods: ", methods)
    methods = ['rpmc-traditional', 'rpmc']

    filename = os.path.join(dirpath, '{}{}'.format(basename, ext).replace(' ', '_'))
    print("Writing plot '{}'.".format(filename))
    fig, ax = plt.subplots(1, 1, sharex=False, sharey=False, figsize=[5, 4])
    #ax2 = ax.twinx()
    labels = {
        'rpmc-traditional': 'traditional scheme',
        'rpmc': 'bucketing scheme'
    }
    scale_exp_labels = {
        'rpmc-traditional': (2, r'$\sim n^2$ cost model'),
        'rpmc': (1, r'$\sim n$ cost model')
    }
    cmap = plt.cm.tab10(np.linspace(0, 1, num=10))
    cmap2 = plt.cm.tab10_r(np.linspace(0, 1, num=10))
    #n0b = dfa.n.min()
    #n1b = max(dfa.n.max(), np.max(dfa['avg-num-active-particles'].to_numpy()))
    n0b = 1.
    n1b = 1.e+7
    nrefs = np.logspace(np.log2(n0b), np.log2(n1b), num=512, base=2)
    for method, color in zip(methods, cmap):
        dfaf = dfa[dfa.method == method]
        ns = dfaf.n.to_numpy()
        n_min_active_means,n_min_active_stds = dfaf['min-num-active-particles'].to_numpy().T
        n_max_active_means,n_max_active_stds = dfaf['max-num-active-particles'].to_numpy().T
        n_avg_active_means,n_avg_active_stds = dfaf['avg-num-active-particles'].to_numpy().T
        tmeans, tstds = dfaf.time.to_numpy().T
        #ax.fill_betweenx(tmeans, ns, n_max_active_means, color=color, alpha=0.15)
        #ax.errorbar(ns, tmeans, yerr=tstds, color=color, linestyle='dotted', alpha=0.75)
        attenuated_color = plot.adjust_lightness(color=color, amount=0.3)
        for n0, nmax, t in zip(ns, n_max_active_means, tmeans):
            ax.plot([n0, nmax], [t, t], color=attenuated_color, linewidth=3)
        ax.scatter(ns, tmeans, marker='o', color=attenuated_color)
        ax.scatter(n_max_active_means, tmeans, marker='>', color=attenuated_color)
        ax.errorbar(n_avg_active_means, tmeans, xerr=n_avg_active_stds, yerr=tstds, color=color, label=labels[method])
        #ax.plot(n_avg_active_means, tmeans, color=color, label=labels[method])
        #if method == 'rpmc':
        #    ν_min_active_means,ν_min_active_stds = dfaf['min-num-buckets'].to_numpy().T
        #    ν_max_active_means,ν_max_active_stds = dfaf['max-num-buckets'].to_numpy().T
        #    ν_avg_active_means,ν_avg_active_stds = dfaf['avg-num-buckets'].to_numpy().T
        #    ax2.plot(n_avg_active_means, ν_avg_active_means, color='gray', alpha=0.5, linestyle='--', label='average number of buckets')
        scale_exp, reflabel = scale_exp_labels[method]
        scalef = linregmin_n_0(n_avg_active_means, tmeans, scale_exp)
        trefs = scalef(nrefs)
        ax.plot(nrefs, trefs, linestyle='--', alpha=0.6, color=color, label=reflabel)
    memrefs_traditional = nrefs**2*8  # simplistic memory model
    memrefs_bucketing = nrefs*16  # simplistic memory model
    nmemf = lambda memrefs, mem: nrefs[np.argwhere(memrefs > mem)[0]]
    nL1t = nmemf(memrefs_traditional, memL1)
    nL2ct = nmemf(memrefs_traditional, memL2c)
    nL3st = nmemf(memrefs_traditional, memL3s)
    nDRAMt = nmemf(memrefs_traditional, memDRAM)
    nL1b = nmemf(memrefs_bucketing, memL1)
    nL2cb = nmemf(memrefs_bucketing, memL2c)
    nL3sb = nmemf(memrefs_bucketing, memL3s)
    #nL2cb = nmemf(memrefs_bucketing, memL2c)
    #ax.axvline(nL1t,   linestyle='dotted', linewidth=0.6, color=cmap[0], alpha=0.6)
    ax.axvline(nL2ct,  linestyle='dotted', linewidth=0.9, color=cmap[0], alpha=0.6)
    ax.axvline(nL3st,  linestyle='dotted', linewidth=1.2, color=cmap[0], alpha=0.6)
    ax.axvline(nDRAMt, linestyle='dotted', linewidth=3,   color=cmap[0], alpha=0.6)
    #ax.axvline(nL1b,   linestyle='dotted', linewidth=0.6, color=cmap[1], alpha=0.6)
    ax.axvline(nL2cb,  linestyle='dotted', linewidth=0.9, color=cmap[1], alpha=0.6)
    ax.axvline(nL3sb,  linestyle='dotted', linewidth=1.2, color=cmap[1], alpha=0.6)
    #ax.text(0.97*nL1t,   1.5e+2, "L1\n$32\\,\\mathrm{KiB}$",   size=7.5, ha="right", va="center", color=cmap[0], alpha=0.6)
    ax.text(0.97*nL2ct,  3.e-1, "L2c\n$256\\,\\mathrm{KiB}$", size=7.5, ha="right", va="center", color=cmap[0], alpha=0.6)
    ax.text(0.97*nL3st,  3.e+1, "L3s\n$8\\,\\mathrm{MiB}$",   size=7.5, ha="right", va="center", color=cmap[0], alpha=0.6)
    ax.text(0.94*nDRAMt, 5.e+2, "DRAM\n$64\\,\\mathrm{GiB}$", size=7.5, ha="right", va="center", color=cmap[0], alpha=0.6)
    #ax.text(0.97*nL1b,   3.5e+1, "L1\n$32\\,\\mathrm{KiB}$",   size=7.5, ha="right", va="center", color=cmap[1], alpha=0.6)
    ax.text(0.97*nL2cb,  8.e+0, "L2c\n$256\\,\\mathrm{KiB}$", size=7.5, ha="right", va="center", color=cmap[1], alpha=0.6)
    ax.text(0.97*nL3sb,  2.e+2, "L2c\n$8\\,\\mathrm{MiB}$",   size=7.5, ha="right", va="center", color=cmap[1], alpha=0.6)
    ax.set_xscale('log')
    ax.set_yscale('log')
    #ax2.set_yscale('log')
    ax.set_xlabel(r'number of entities $n$')
    ax.set_ylabel(r'time ($\mathrm{s}$)')
    #ax.set_ylim(7.e-3, 1.5e+3)
    ax.set_xlim(2.e+1, 7.e+6)
    ax.set_ylim(2.e-3, 2.e+3)
    ##ax2.set_ylim(0, 30)
    #ax2.set_ylim(10, 100)
    ax.legend(loc='lower right')
    #ax2.legend(loc='upper right')
    ax.text(3.e+1, 1.2e+3, r"runaway kernel", size=10, ha="left", va="top",
            bbox=dict(boxstyle="round", ec=(0.2, 0.2, 0.2), fc=(0.95, 0.95, 0.95)))
    fig.set_tight_layout(True)
    fig.savefig(filename)
    plt.close(fig)


def mem_model_Orm10(n):
    # RP data (properties; simulation state; interaction rates)
    mem_RP = (14 + 13)*8*n + 10*8*n + (n + 1)*8*n
    return mem_RP

def mem_model_bucketing_Orm10(n,ν):
    # RP data (bucketing indices; properties; simulation state)
    mem_RP = 2*(4 + 4+4)*n + (14 + 13)*8*n + 10*8*n
    # bucket data (ordered refs, bucket data, payload; properties)
    mem_buckets = 2*(4+4 + 2*3*8+4+4+8*ν + 4+4+4+1.5*8+4)*ν + (14 + 13)*2*8*ν
    return mem_RP + mem_buckets


def plot_stirring_mem(base_filename, ext, cmd_args):
    dirpath, basename = os.path.split(base_filename)
    os.makedirs(dirpath, exist_ok=True)
    
    ns = np.logspace(6, 20, base=2, num=128)
    #νs = np.logspace(0, 4, base=10, num=9)
    #νs = [1, 100, 1_000, 10_000]
    νs = 2**np.array([6, 9, 12, 15])

    # Xeon E3-1585 v5
    memL1 = 32.*1024
    memL2c = 256.*1024
    memL2s = 1.*1024*1024
    memL3c = 2.*1024*1024
    memL3s = 8.*1024*1024
    memL4 = 128.*1024*1024

    # mp-media* machines
    memDRAM = 64.*1024*1024*1024

    bytes_traditional = mem_model_Orm10(ns)

    filename = os.path.join(dirpath, '{}{}'.format(basename, ext).replace(' ', '_'))
    print("Writing plot '{}'.".format(filename))
    fig, ax = plt.subplots(1, 1, sharex=False, sharey=False, figsize=[5, 4])
    cmap = plt.cm.tab10(np.linspace(0, 1, num=10))
    #cmap2 = plt.cm.tab10_r(np.linspace(0, 1, num=10))
    cmap2 = cmap
    ax.plot(ns, bytes_traditional/1024, linestyle='--', label='traditional', color=cmap[0])
    cmap_iter = iter(cmap)
    next(cmap_iter)
    for ν, color in zip(νs, cmap_iter):
        bytes = mem_model_bucketing_Orm10(n=ns, ν=np.minimum(ns,ν))
        #ax.plot(ns, bytes/(1024*1024), label=r'${}$ buckets'.format(planets.plot.format_exp(ν, digits=1)), color=color)
        #ax.plot(ns, bytes/(1024*1024), label=r'${}$ buckets'.format(planets.plot.format_exp(ν, digits=1, base=2)), color=color)
        ax.plot(ns, bytes/(1024*1024), label=r'$\leq {:_}$ buckets'.format(ν).replace('_', r'\,'), color=color)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'number of entities $n$')
    ax.set_ylabel(r'working set size ($\mathrm{MiB}$)')
    #ax.axhline(memL1/(1024*1024),   linestyle='dotted', linewidth=0.6, color='black', alpha=0.5)
    ax.axhline(memL2c/(1024*1024),  linestyle='dotted', linewidth=0.9, color='black', alpha=0.5)
    ax.axhline(memL3s/(1024*1024),  linestyle='dotted', linewidth=1.2, color='black', alpha=0.5)
    ax.axhline(memDRAM/(1024*1024), linestyle='dotted', linewidth=3,   color='black', alpha=0.5)
    #ax.text(8.e+5, 1.03*memL1/(1024*1024),   "L1\n$32\\,\\mathrm{KiB}$",   size=7.5, ha="right", va="bottom", color='black', alpha=0.5)
    ax.text(9.e+5, 1.03*memL2c/(1024*1024),  "L2c\n$256\\,\\mathrm{KiB}$", size=7.5, ha="right", va="bottom", color='black', alpha=0.5)
    ax.text(9.e+5, 1.03*memL3s/(1024*1024),  "L3s\n$8\\,\\mathrm{MiB}$",   size=7.5, ha="right", va="bottom", color='black', alpha=0.5)
    ax.text(9.e+5, 1.12*memDRAM/(1024*1024), "DRAM\n$64\\,\\mathrm{GiB}$", size=7.5, ha="right", va="bottom", color='black', alpha=0.5)
    ax.set_xlim(1.e+2, 1.e+6)
    ax.set_ylim(1.e-1, 1.e+9)
    ##ax.axhline(memL1/(1024*1024), linestyle='dotted', color=cmap2[0], label='L1 cache')
    #ax.axhline(memL2c/(1024*1024), linestyle='dotted', linewidth=0.7, color=cmap2[1])
    ##ax.axhline(memL2s/(1024*1024), linestyle='dotted', color=cmap2[2], label='L2 cache (per socket)')
    #ax.axhline(memL3s/(1024*1024), linestyle='dotted', linewidth=1.5, color=cmap2[3])
    ##ax.axhline(memL4/(1024*1024), linestyle='dotted', color=cmap2[4], label='L4 cache')
    #ax.axhline(memDRAM/(1024*1024), linestyle='dotted', linewidth=3, color=cmap2[3])
    ##ax.legend(loc='lower right')
    ax.text(7.5e+5, 4.5e+8, 'stirring test', size=10, ha="right", va="top",
            bbox=dict(boxstyle="round", ec=(0.2, 0.2, 0.2), fc=(0.95, 0.95, 0.95)))
    ax.legend()
    fig.set_tight_layout(True)
    fig.savefig(filename)
    plt.close(fig)



def plot_stirring_paramstudy(base_filename, ext, cmd_args):
    import pandas as pd
    #import seaborn

    dirpath, basename = os.path.split(base_filename)
    os.makedirs(dirpath, exist_ok=True)

    datasets = [
        types.SimpleNamespace(
            file = 'data/perf/stirring-test7-widen.tsv',
            plot_name = '',
            label = 'stirring test',
            label_coords = (9.e+1, 1.8e+5)
        )
    ]

    #'removal-bucket-update-delay'
    #'bin-widening'
    for dataset in datasets:
        df = pd.read_csv(dataset.file, sep='\t')
        df['time_event'] = df['time']/df['num-events']
        df['sampling-duration_event'] = df['sampling-duration']/df['num-events']
        df['updating-duration_event'] = df['updating-duration']/df['num-events']
        df['acceptance-prob'] = df['num-events']/(df['num-events'] + df['num-rejections'])
        #dfp = df.pivot_table(
        #    values=['time', 'time_event'],
        #    index=['removal-bucket-update-delay'],
        #    columns=['bin-widening'],
        #    aggfunc=['mean', 'std'])
        #
        #dfa = df.groupby(['removal-bucket-update-delay', 'bin-widening'], as_index=False).agg({
        #    'time': ['mean', 'std'],
        #    #'time_event': ['mean', 'std'],
        #    #'num-events': ['mean', 'std'],
        #    #'min-num-buckets': ['mean', 'std'],
        #    #'max-num-buckets': ['mean', 'std'],
        #    #'avg-num-buckets': ['mean', 'std'],
        #    #'min-num-active-particles': ['mean', 'std'],
        #    #'max-num-active-particles': ['mean', 'std'],
        #    #'avg-num-active-particles': ['mean', 'std'],
        #    'avg-acceptance-probability': ['mean', 'std']
        #})
        #print(dfp)
        #print(dfp['mean', 'time'])
        #print(dfp['std', 'time']/dfp['mean', 'time'])
        #print(dfp['mean', 'time_event'])
        #print(dfp['std', 'time_event']/dfp['mean', 'time_event'])
        #filename = os.path.join(dirpath, '{}{}{}{}'.format(basename, ' ' if dataset.plot_name != '' else '', dataset.plot_name, ext).replace(' ', '_'))
        #print("Writing plot '{}'.".format(filename))
        #fig, ax = plt.subplots(1, 1, sharex=False, sharey=False, figsize=[5, 4])
        #seaborn.heatmap(ax=ax, data=dfp['mean', 'time_event']*1.e+5, annot=True, norm=matplotlib.colors.LogNorm())
        #fig.set_tight_layout(True)
        #fig.savefig(filename)
        #plt.close(fig)

        df_delay1 = df[df['removal-bucket-update-delay'] == 1.]
        dfw = df_delay1.groupby(['bin-widening'], as_index=False).agg({
            'total-duration': ['mean', 'std'],
            'sampling-duration': ['mean', 'std'],
            'updating-duration': ['mean', 'std'],
            'time_event': ['mean', 'std'],
            'sampling-duration_event': ['mean', 'std'],
            'updating-duration_event': ['mean', 'std'],
            'num-events': ['mean', 'std'],
            'num-rejections': ['mean', 'std'],
            'acceptance-prob': ['mean', 'std'],
            #'min-num-buckets': ['mean', 'std'],
            #'max-num-buckets': ['mean', 'std'],
            #'avg-num-buckets': ['mean', 'std'],
            #'min-num-active-particles': ['mean', 'std'],
            #'max-num-active-particles': ['mean', 'std'],
            #'avg-num-active-particles': ['mean', 'std'],
            'avg-acceptance-probability': ['mean', 'std']
        })

        cmap = plt.cm.Set2(np.linspace(0, 1, num=8))
        filename = os.path.join(dirpath, '{}{}{}{}'.format(basename, ' ' if dataset.plot_name != '' else '', dataset.plot_name, ext).replace(' ', '_'))
        print("Writing plot '{}'.".format(filename))
        fig, ax = plt.subplots(1, 1, sharex=False, sharey=False, figsize=[5, 4])
        ax.fill_between(dfw['bin-widening'], 0., dfw['sampling-duration_event', 'mean']*1.e+3, color=cmap[0], label='sampling time', alpha=0.25)
        #attenuated_color = plot.adjust_lightness(color=color, amount=0.3)
        #ax.scatter(dfw['bin-widening'], tmeans, color=plot.adjust_lightness(color=cmap[0], amount=0.3), marker='o')
        ax.errorbar(dfw['bin-widening'], dfw['sampling-duration_event', 'mean']*1.e+3, yerr=dfw['sampling-duration_event', 'std']*1.e+3, linestyle='none', color=cmap[0])
        ax.errorbar(dfw['bin-widening'], dfw['sampling-duration_event', 'mean']*1.e+3 + dfw['updating-duration_event', 'mean']*1.e+3, yerr=dfw['updating-duration_event', 'std']*1.e+3, linestyle='none', color=cmap[1])
        #ax.scatter(dfw['bin-widening'], dfw['sampling-duration', 'mean'], yerr=dfw['sampling-duration', 'std'], linestyle='none')
        ax.fill_between(dfw['bin-widening'], dfw['sampling-duration_event', 'mean']*1.e+3, dfw['sampling-duration_event', 'mean']*1.e+3 + dfw['updating-duration_event', 'mean']*1.e+3, color=cmap[1], label='updating time', alpha=0.25)
        ax.set_xlabel(r'widening fraction $f$')
        ax.set_ylabel(r'average per-event runtime ($\mathrm{ms})$')
        ax.legend(loc='lower left')
        ax2 = ax.twinx()
        ax2.errorbar(dfw['bin-widening'], dfw['acceptance-prob', 'mean'], yerr=dfw['acceptance-prob', 'std'], linestyle='dashed', color='gray', label="average probability of\nevent acceptance $p$")
        ax2.set_ylabel(r'average acceptance probability $p$')
        ##ax2.set_yscale('log')
        ##ax2.set_ylim(5.e-3, 5.e-1)
        ax2.set_ylim(0, 1)
        ax2.legend(loc='upper right')
        fig.set_tight_layout(True)
        fig.savefig(filename)
        plt.close(fig)


def plot_stirring_paramstudy_buckets(base_filename, ext, cmd_args):
    import pandas as pd
    import seaborn

    dirpath, basename = os.path.split(base_filename)
    os.makedirs(dirpath, exist_ok=True)

    datasets = [
        types.SimpleNamespace(
            file = 'data/perf/stirring-test7-param.tsv',
            plot_name = '',
            label = 'stirring test',
            label_coords = (9.e+1, 1.8e+5)
        )
    ]

    #'removal-bucket-update-delay'
    #'bin-widening'
    for dataset in datasets:
        df = pd.read_csv(dataset.file, sep='\t')
        df['time_event'] = df['time']/df['num-events']
        df['sampling-frac'] = df['sampling-duration']/df['total-duration']
        df['acceptance-prob'] = df['num-events']/(df['num-events'] + df['num-rejections'])
        dfp = df.pivot_table(
            values=['time', 'time_event', 'sampling-frac', 'acceptance-prob'],
            index=['m-bins-per-decade'],
            columns=['e-bins-per-decade'],
            aggfunc=['mean', 'std'])
        #print(dfp)
        #print(dfp['mean', 'time'])
        #print(dfp['std', 'time']/dfp['mean', 'time'])
        #print(dfp['mean', 'time_event'])
        #print(dfp['std', 'time_event']/dfp['mean', 'time_event'])
        filename = os.path.join(dirpath, '{}{} time{}{}'.format(basename, ' ' if dataset.plot_name != '' else '', dataset.plot_name, ext).replace(' ', '_'))
        print("Writing plot '{}'.".format(filename))
        fig, ax = plt.subplots(1, 1, sharex=False, sharey=False, figsize=[5, 4])
        #seaborn.heatmap(ax=ax, data=dfp['mean', 'time_event']*1.e+3, annot=True, norm=matplotlib.colors.LogNorm())
        seaborn.heatmap(ax=ax, data=dfp['mean', 'time_event']*1.e+3, annot=True)
        ax.set_xlabel(r'rms-velocity bucket density $\theta_v$')
        ax.set_ylabel(r'mass bucket densities $\theta_m = \theta_M$')
        ax.collections[0].colorbar.set_label(r'average per-event runtime ($\mathrm{ms})$')
        fig.set_tight_layout(True)
        fig.savefig(filename)
        plt.close(fig)

        filename = os.path.join(dirpath, '{}{} sampling-fraction{}{}'.format(basename, ' ' if dataset.plot_name != '' else '', dataset.plot_name, ext).replace(' ', '_'))
        print("Writing plot '{}'.".format(filename))
        fig, ax = plt.subplots(1, 1, sharex=False, sharey=False, figsize=[5, 4])
        #seaborn.heatmap(ax=ax, data=dfp['mean', 'time_event']*1.e+3, annot=True, norm=matplotlib.colors.LogNorm())
        seaborn.heatmap(ax=ax, data=dfp['mean', 'sampling-frac'], annot=True, fmt='.0%', cmap="crest")
        ax.set_xlabel(r'rms-velocity bucket density $\theta_v$')
        ax.set_ylabel(r'mass bucket densities $\theta_m = \theta_M$')
        ax.collections[0].colorbar.set_label(r'fraction of time spent on sampling events')
        fig.set_tight_layout(True)
        fig.savefig(filename)
        plt.close(fig)

        filename = os.path.join(dirpath, '{}{} acceptance-prob{}{}'.format(basename, ' ' if dataset.plot_name != '' else '', dataset.plot_name, ext).replace(' ', '_'))
        print("Writing plot '{}'.".format(filename))
        fig, ax = plt.subplots(1, 1, sharex=False, sharey=False, figsize=[5, 4])
        #seaborn.heatmap(ax=ax, data=dfp['mean', 'time_event']*1.e+3, annot=True, norm=matplotlib.colors.LogNorm())
        seaborn.heatmap(ax=ax, data=dfp['mean', 'acceptance-prob'], annot=True, fmt='.1%', cmap="crest")
        ax.set_xlabel(r'rms-velocity bucket density $\theta_v$')
        ax.set_ylabel(r'mass bucket densities $\theta_m = \theta_M$')
        ax.collections[0].colorbar.set_label(r'average probability of event acceptance $p$')
        fig.set_tight_layout(True)
        fig.savefig(filename)
        plt.close(fig)


def plot_stirring_perf(base_filename, ext, cmd_args):
    import pandas as pd

    dirpath, basename = os.path.split(base_filename)
    os.makedirs(dirpath, exist_ok=True)

    labels = {
        ('rpmc-traditional', 'none'): 'traditional scheme',
        #('rpmc', 'none'): 'bucketing',
        ('rpmc', 'locality'): 'bucketing scheme'
    }
    bucketlabels = {
        ('rpmc-traditional', 'none'): None,
        #('rpmc', 'none'): r'occupied buckets $\nu$',
        ('rpmc', 'locality'): r'occupied buckets $\nu$'
    }
    scale_exp_labels = {
        ('rpmc-traditional', 'none'): (2, r'$\sim n^2$ cost model'),
        #('rpmc', 'none'): (1, r'$\mathcal{O}(n)$ scaling'),
        ('rpmc', 'locality'): (1, r'$\sim n$ cost model'),
        #('rpmc', 'locality'): (2./3, r'$\sim n^{2/3}$ cost model')
    }
    event_labels = {
        ('rpmc-traditional', 'none'): 'events and updates (traditional)',
        #('rpmc', 'none'): 'events (bucketing)',
        ('rpmc', 'locality'): 'events (bucketing with locality)'
    }
    cmap = plt.cm.tab10(np.linspace(0, 1, num=10))
    cmap2 = plt.cm.tab10_r(np.linspace(0, 1, num=10))

    datasets = [
        #types.SimpleNamespace(
        #    file = 'data/perf/stirring.tsv',
        #    plot_name = '',
        #    label = 'stirring test',
        #    label_coords = (1.e+2, 3.6e+5),
        #    xlim = (9.e+1, 3.e+3),
        #    ylim = (1.e+2, 5.e+5),
        #    ylim_buckets = (5.e+4, 5.e+10)
        #),
        #types.SimpleNamespace(
        #    file = 'data/perf/stirring-fast.tsv',
        #    plot_name = 'fast',
        #    label = 'stirring test with boosting',
        #    label_coords = (1.04e+2, 2.e+4),
        #    xlim = (9.e+1, 1.e+4),
        #    ylim = (5.e+0, 5.e+4),
        #    ylim_buckets = (3.e+4, 3.e+7)
        #),
        #types.SimpleNamespace(
        #    file = 'data/perf/stirring-test7m-boost.tsv',
        #    plot_name = 'fast7m',
        #    label = 'stirring test with boosting',
        #    label_coords = (5.6e+2, 3.6e+4),
        #    label_coords_events = (5.6e+2, 3.6e+4),
        #    label_coords_buckets = (2.05e+4, 240),
        #    label_coords_event_cost = (2.05e+4, 1.1e-1),
        #    xlim = (5.e+2, 1.e+5),
        #    ylim = (1.e+3, 4.e+4),
        #    ylim_events = (3.e+5, 3.e+10),
        #    ylim_buckets = (3.e+5, 3.e+10),
        #    ylim_event_cost = (3.e+5, 3.e+10)
        #),
        #types.SimpleNamespace(
        #    file = 'data/perf/stirring-test7-boost.tsv',
        #    plot_name = 'fast7',
        #    label = 'stirring test with boosting',
        #    label_coords = (5.6e+2, 3.6e+4),
        #    label_coords_events = (5.6e+2, 3.6e+4),
        #    label_coords_buckets = (4.3e+4, 240),
        #    label_coords_event_cost = (4.3e+4, 1.1e-1),
        #    xlim = (5.e+2, 5.e+4),
        #    ylim = (2.e+3, 4.e+4),
        #    ylim_buckets = (3.e+5, 3.e+10)
        #),
        types.SimpleNamespace(
            file = 'data/perf/stirring-test9-boost.tsv',
            plot_name = 'fast9',
            label = 'stirring test with boosting',
            label_coords = (6.2e+2, 3.6e+4),
            label_coords_events = (5.6e+2, 3.6e+4),
            label_coords_buckets = (4.3e+4, 240),
            label_coords_event_cost = (4.3e+4, 1.1e-1),
            xlim = (5.e+2, 5.e+5),
            ylim = (1.e+2, 5.e+4),
            ylim_buckets = (3.e+5, 3.e+10)
        ),
        types.SimpleNamespace(
            file = 'data/perf/stirring-test9-frag-boost.tsv',
            plot_name = 'fast9_frag',
            label = "stirring test with boosting\nand fragmentation",
            label_coords = (6.2e+2, 3.6e+4),
            label_coords_events = (5.6e+2, 3.6e+4),
            label_coords_buckets = (4.3e+4, 240),
            label_coords_event_cost = (4.3e+4, 1.1e-1),
            xlim = (5.e+2, 1.e+5),
            ylim = (1.e+2, 5.e+4),
            ylim_buckets = (3.e+5, 3.e+10)
        ),
        #types.SimpleNamespace(
        #    file = 'data/perf/stirring-fast-alt.tsv',
        #    plot_name = 'fast-alt',
        #    label = 'stirring test with boosting',
        #    label_coords = (1.04e+2, 2.e+4),
        #    xlim = (9.e+1, 1.e+4),
        #    ylim = (5.e+0, 5.e+4),
        #    ylim_buckets = (3.e+4, 3.e+7)
        #),
        #types.SimpleNamespace(
        #    file = 'data/perf/stirring-fast-alt2.tsv',
        #    plot_name = 'fast-alt2',
        #    label = 'stirring test with boosting',
        #    label_coords = (1.04e+2, 2.e+4),
        #    xlim = (9.e+1, 1.e+4),
        #    ylim = (5.e+0, 5.e+4),
        #    ylim_buckets = (3.e+4, 3.e+7)
        #),
        #types.SimpleNamespace(
        #    file = 'data/perf/stirring-fast-alt3.tsv',
        #    plot_name = 'fast-alt3',
        #    label = 'stirring test with boosting',
        #    label_coords = (1.04e+2, 2.e+4),
        #    xlim = (9.e+1, 1.e+4),
        #    ylim = (5.e+0, 5.e+4),
        #    ylim_buckets = (3.e+4, 3.e+7)
        #)
    ]
    for dataset in datasets:
        df = pd.read_csv(dataset.file, sep='\t')
        df['time_event'] = df['time']/df['num-events']
        dfa = df.groupby(['method', 'options', 'nPlt'], as_index=False).agg({
            'time': ['mean', 'std'],
            'time_event': ['mean', 'std'],
            'num-events': ['mean', 'std'],
            'num-bucket-updates': ['mean', 'std'],
            'min-num-buckets': ['mean', 'std'],
            'max-num-buckets': ['mean', 'std'],
            'avg-num-buckets': ['mean', 'std'],
            'min-num-active-particles': ['mean', 'std'],
            'max-num-active-particles': ['mean', 'std'],
            'avg-num-active-particles': ['mean', 'std']
        })
        #methods = np.unique(dfa.method)
        #options = np.unique(dfa.options)
        #print("methods: ", methods)
        #methods = ['rpmc', 'rpmc-traditional']
        #methods = ['rpmc']
        methods_options = [
            ('rpmc-traditional', 'none'),
            #('rpmc', 'none'),
            ('rpmc', 'locality')
        ]
        rpmc_method = 'rpmc'

        dfea = df.groupby(['method', 'options', 'num-events'], as_index=False).agg({
            'time': ['mean', 'std'],
            'nPlt': ['mean', 'std'],
            'min-num-buckets': ['mean', 'std'],
            'max-num-buckets': ['mean', 'std'],
            'avg-num-buckets': ['mean', 'std'],
            'min-num-active-particles': ['mean', 'std'],
            'max-num-active-particles': ['mean', 'std'],
            'avg-num-active-particles': ['mean', 'std']
        })


        filename = os.path.join(dirpath, '{}{}{}{}'.format(basename, ' ' if dataset.plot_name != '' else '', dataset.plot_name, ext).replace(' ', '_'))
        print("Writing plot '{}'.".format(filename))
        fig, ax = plt.subplots(1, 1, sharex=False, sharey=False, figsize=[5, 4])
        #n0b = dfa['nPlt'].min()
        #n1b = max(dfa['nPlt'].max(), np.max(dfa['avg-num-active-particles'].to_numpy()))
        n0b = 7.e+1
        n1b = 1.e+6
        nrefs = np.logspace(np.log2(n0b), np.log2(n1b), num=512, base=2)
        νrefs = 400
        #navgs = {
        #    method: dfa[dfa.method == method]['avg-num-active-particles'].to_numpy().T
        #}
        #νavgs = {
        #    method: dfa[dfa.method == method]['avg-num-buckets'].to_numpy().T
        #}
        for (method, option), color in zip(methods_options, cmap):
            #dfaf = dfa[(dfa.method == method) & (dfa.options == option)]
            dfaf = dfa[dfa.method == method]
            ns = dfaf['nPlt'].to_numpy()
            n_min_active_means,n_min_active_stds = dfaf['min-num-active-particles'].to_numpy().T
            n_max_active_means,n_max_active_stds = dfaf['max-num-active-particles'].to_numpy().T
            n_avg_active_means,n_avg_active_stds = dfaf['avg-num-active-particles'].to_numpy().T
            ν_min_active_means,ν_min_active_stds = dfaf['min-num-buckets'].to_numpy().T
            ν_max_active_means,ν_max_active_stds = dfaf['max-num-buckets'].to_numpy().T
            ν_avg_active_means,ν_avg_active_stds = dfaf['avg-num-buckets'].to_numpy().T
            tmeans, tstds = dfaf.time.to_numpy().T
            attenuated_color = plot.adjust_lightness(color=color, amount=0.3)
            for n0, navg, nmax, t in zip(ns, n_avg_active_means, n_max_active_means, tmeans):
                ax.plot([min(n0, navg), nmax], [t, t], color=attenuated_color, linewidth=3)
            ax.scatter(ns, tmeans, marker='o', color=attenuated_color)
            ax.scatter(n_max_active_means, tmeans, marker='>', color=attenuated_color)
            ax.errorbar(n_avg_active_means, tmeans, xerr=n_avg_active_stds, yerr=tstds, color=color, label=labels[(method,option)])
            #ax.fill_betweenx(tmeans, np.minimum(ns, n_avg_active_means), n_max_active_means, color=color, alpha=0.15)
            #ax.plot(ns, tmeans, color=color, linestyle='dotted', alpha=0.75)
            #ax.errorbar(n_avg_active_means, tmeans, xerr=n_avg_active_stds, yerr=tstds, color=color, label=labels[(method,option)])
            scale_exp, reflabel = scale_exp_labels[(method,option)]
            scalef = linregmin_n_0(n_avg_active_means, tmeans, scale_exp)
            #scalef = linregmin_n_0(ν_avg_active_means if method == 'rpmc' else n_avg_active_means, tmeans, scale_exp)
            trefs = scalef(nrefs)
            ax.plot(nrefs, trefs, linestyle='--', alpha=0.6, color=color, label=reflabel)
            if method == 'rpmc':
                #scalef = linregmin_n_0(n_avg_active_means*ν_avg_active_means, tmeans, scale_exp)
                #trefs = scalef(n_avg_active_means*ν_avg_active_means)
                scalef = linregmin_n_0(n_avg_active_means, tmeans, 2./3)
                trefs = scalef(nrefs)
                #ax.plot(n_avg_active_means, trefs, linestyle='dotted', alpha=0.6, color=color, label=r'$\mathcal{O}(n \cdot \nu)$ scaling' if option == 'none' else None)
                ax.plot(nrefs, trefs, linestyle='dotted', alpha=0.6, color=color, label=r'$\sim n^{2/3}$ cost model')
        if dataset.ylim[1] > 60*60*8:
            ax.axhline(60*60*8, color='black', alpha=0.5, linewidth=3, linestyle='dotted')
            ax.text(dataset.xlim[1]*0.9, 60*60*8*1.05, "$8\,\\mathrm{h}$ timeout", size=7.5, ha="right", va="bottom", color='black', alpha=0.5)

        #memrefs_traditional = nrefs**2*8  # simplistic memory model
        #memrefs_bucketing = mem_model_bucketing_Orm10(n=nrefs, ν=νrefs)  # simplistic memory model
        #memL1 = 32.*1024
        #memL2c = 1.*1024*1024
        #memL2s = 24.*1024*1024
        #memL3c = memL2c + 1.375*1024*1024
        #memL3s = memL2s + 33.*1024*1024
        #nmemf = lambda memrefs, mem: nrefs[np.argwhere(memrefs > mem)[0]]
        #ax.axvline(nmemf(memrefs_traditional, memL2c), linestyle='dotted', linewidth=0.7, color=cmap[1], alpha=0.6)
        #ax.axvline(nmemf(memrefs_traditional, memL3s), linestyle='dotted', linewidth=1.5, color=cmap[1], alpha=0.6)
        #ax.axvline(nmemf(memrefs_bucketing, memL2c), linestyle='dotted', linewidth=0.7, color=cmap[0], alpha=0.6)
        #ax.axvline(nmemf(memrefs_bucketing, memL3s), linestyle='dotted', linewidth=0.7, color=cmap[0], alpha=0.6)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'number of entities $n$')
        ax.set_ylabel(r'time ($\mathrm{s}$)')
        #ax.set_ylim(7.e-3, 1.5e+3)
        ax.set_xlim(*dataset.xlim)
        ax.set_ylim(*dataset.ylim)
        ax.legend(loc='lower right')
        ax.text(*dataset.label_coords, dataset.label, size=10, ha="left", va="top",
                bbox=dict(boxstyle="round", ec=(0.2, 0.2, 0.2), fc=(0.95, 0.95, 0.95)))
        fig.set_tight_layout(True)
        fig.savefig(filename)
        plt.close(fig)

        filename = os.path.join(dirpath, '{}{}{} events{}'.format(basename, ' ' if dataset.plot_name != '' else '', dataset.plot_name, ext).replace(' ', '_'))
        print("Writing plot '{}'.".format(filename))
        fig, ax = plt.subplots(1, 1, sharex=False, sharey=False, figsize=[5, 4])
        #dfaf = dfa[(dfa.method == rpmc_method) & (dfa.options == 'none')]  # TODO: ?
        #n_avg_active_means,n_avg_active_stds = dfaf['avg-num-active-particles'].to_numpy().T
        for (method,option), color in zip(methods_options, cmap):
            #dfaf = dfa[(dfa.method == method) & (dfa.options == option)]
            dfaf = dfa[dfa.method == method]
            n_avg_active_means,n_avg_active_stds = dfaf['avg-num-active-particles'].to_numpy().T
            num_events_means,num_events_stds = dfaf['num-events'].to_numpy().T
            #ax.fill_betweenx(tmeans, np.minimum(ns, n_avg_active_means), n_max_active_means, color=color, alpha=0.15)
            #ax.plot(ns, tmeans, color=color, linestyle='dotted', alpha=0.75)
            ax.errorbar(n_avg_active_means, num_events_means, xerr=n_avg_active_stds, yerr=num_events_stds, color=color, label=event_labels[(method,option)])
            if method == 'rpmc':
                num_updates_means,num_updates_stds = dfaf['num-bucket-updates'].to_numpy().T
                ax.errorbar(n_avg_active_means, num_updates_means, xerr=n_avg_active_stds, yerr=num_updates_stds, color=color, label='bucket updates ({})'.format(labels[(method,option)]), linestyle='dotted')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'number of entities $n$')
        ax.set_ylabel(r'number of events and updates')
        #ax.set_ylim(7.e-3, 1.5e+3)
        #ax.set_ylim(*dataset.ylim_buckets)
        ax.legend(loc='upper left')
        #ax.text(*dataset.label_coords, dataset.label, size=10, ha="left", va="top",
        #        bbox=dict(boxstyle="round", ec=(0.2, 0.2, 0.2), fc=(0.95, 0.95, 0.95)))
        fig.set_tight_layout(True)
        fig.savefig(filename)
        plt.close(fig)

        filename = os.path.join(dirpath, '{}{}{} buckets{}'.format(basename, ' ' if dataset.plot_name != '' else '', dataset.plot_name, ext).replace(' ', '_'))
        print("Writing plot '{}'.".format(filename))
        fig, ax = plt.subplots(1, 1, sharex=False, sharey=False, figsize=[5, 3])
        #dfaf = dfa[(dfa.method == rpmc_method) & (dfa.options == 'none')]  # TODO: ?
        #n_avg_active_means,n_avg_active_stds = dfaf['avg-num-active-particles'].to_numpy().T
        for (method,option), color in zip(methods_options, cmap):
            #dfaf = dfa[(dfa.method == method) & (dfa.options == option)]
            dfaf = dfa[dfa.method == method]
            n_avg_active_means,n_avg_active_stds = dfaf['avg-num-active-particles'].to_numpy().T
            if method == 'rpmc':
                ν_avg_active_means,ν_avg_active_stds = dfaf['avg-num-buckets'].to_numpy().T
                #ax.errorbar(n_avg_active_means, ν_avg_active_means, xerr=n_avg_active_stds, yerr=ν_avg_active_stds, color=color, label=bucketlabels[(method,option)], linestyle='--')
                ax.errorbar(n_avg_active_means, ν_avg_active_means, xerr=n_avg_active_stds, yerr=ν_avg_active_stds, color=color, label=None, linestyle='--')
        ax.set_xscale('log')
        #ax.set_yscale('log')
        ax.set_xlabel(r'average number of entities $n$')
        ax.set_ylabel('average number of\noccupied buckets $\\nu$')
        #ax.legend(loc='lower right')
        #ax.text(dataset.label_coords[0], 460, dataset.label, size=10, ha="left", va="top",
        #ax.text(*dataset.label_coords_buckets, dataset.label, size=10, ha="right", va="bottom",
        #        bbox=dict(boxstyle="round", ec=(0.2, 0.2, 0.2), fc=(0.95, 0.95, 0.95)))
        #ax.text(*dataset.label_coords, dataset.label, size=10, ha="left", va="top",
        #        bbox=dict(boxstyle="round", ec=(0.2, 0.2, 0.2), fc=(0.95, 0.95, 0.95)))
        fig.set_tight_layout(True)
        fig.savefig(filename)
        plt.close(fig)

        filename = os.path.join(dirpath, '{}{}{} event-cost{}'.format(basename, ' ' if dataset.plot_name != '' else '', dataset.plot_name, ext).replace(' ', '_'))
        print("Writing plot '{}'.".format(filename))
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=False, figsize=[5, 5])
        #dfaf = dfa[(dfa.method == rpmc_method) & (dfa.options == 'none')]  # TODO: ?
        #n_avg_active_means,n_avg_active_stds = dfaf['avg-num-active-particles'].to_numpy().T
        for (method,option), color in zip(methods_options, cmap):
            #dfaf = dfa[(dfa.method == method) & (dfa.options == option)]
            dfaf = dfa[dfa.method == method]
            reflabel = labels[(method,option)]
            n_avg_active_means,n_avg_active_stds = dfaf['avg-num-active-particles'].to_numpy().T
            time_event_means,time_event_stds = dfaf['time_event'].to_numpy().T*1.e+3
            ax1.errorbar(n_avg_active_means, time_event_means, xerr=n_avg_active_stds, yerr=time_event_stds, color=color, label=reflabel)
            if method == 'rpmc':
                ν_avg_active_means,ν_avg_active_stds = dfaf['avg-num-buckets'].to_numpy().T
                ax2.errorbar(n_avg_active_means, ν_avg_active_means, xerr=n_avg_active_stds, yerr=ν_avg_active_stds, color=color, label=None, linestyle='--')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        #ax1.set_xlabel(r'average number of entities $n$')
        ax1.set_ylabel('average per-event\nruntime $(\\mathrm{ms})$')
        ax1.legend(loc='upper right')
        ax2.set_xscale('log')
        #ax.set_yscale('log')
        ax2.set_xlabel(r'average number of entities $n$')
        ax2.set_ylabel('average number of\noccupied buckets $\\nu$')
        #ax.text(dataset.label_coords[0], 460, dataset.label, size=10, ha="left", va="top",
        #ax.text(*dataset.label_coords_event_cost, dataset.label, size=10, ha="right", va="bottom",
        #        bbox=dict(boxstyle="round", ec=(0.2, 0.2, 0.2), fc=(0.95, 0.95, 0.95)))
        #ax.text(*dataset.label_coords, dataset.label, size=10, ha="left", va="top",
        #        bbox=dict(boxstyle="round", ec=(0.2, 0.2, 0.2), fc=(0.95, 0.95, 0.95)))
        fig.set_tight_layout(True)
        fig.savefig(filename)
        plt.close(fig)


def runaway_kernel(Λ0, mth):
    rcp_mth23 = 1/mth**(2/3)
    def impl(m1,m2):
        mmax23 = np.maximum(m1,m2)**(2/3)
        return Λ0*mmax23*(1 + mmax23*rcp_mth23)
    return impl

def runaway_kernel_representative(Λ0, mth, Nth, M_n):
    rcp_mth23 = 1/mth**(2/3)
    def impl(m1,m2):
        N2 = np.maximum(1., M_n/m2)
        N2[N2 < 10] = 1.
        mmax23 = np.maximum(m1,m2)**(2/3)
        return N2*Λ0*mmax23*(1 + mmax23*rcp_mth23)
    return impl

def runaway_kernel_weighted(Λ0, mth, Nth):
    rcp_mth23 = 1/mth**(2/3)
    def impl(m1,m2,M2):
        N2 = M2/m2
        mmax23 = np.maximum(m1,m2)**(2/3)
        return N2*Λ0*mmax23*(1 + mmax23*rcp_mth23)
    return impl

def plot_runaway_kernel(base_filename, ext, cmd_args):

    dirpath, basename = os.path.split(base_filename)
    os.makedirs(dirpath, exist_ok=True)

    mth = 1.e+7
    Nth = 10
    #M = 1.e+10
    M = 1.e+11
    n = 2048
    #Nm = 64
    Nm = 512
    Λ0 = 10/M
    m0 = 1.e+4
    m1 = 1.e+8
    mcrit = M/(n*Nth)
    ms0 = np.logspace(np.log10(m0), np.log10(m1), Nm)
    ms1 = ms0[..., np.newaxis]
    ms2 = ms0[np.newaxis, ...]

    vmin = 4.e-7
    vmax = 4.e-1

    filename = os.path.join(dirpath, '{} RP-RP{}'.format(basename, ext).replace(' ', '_'))
    print("Writing plot '{}'.".format(filename))
    fig, ax = plt.subplots(1, 1, sharex=False, sharey=False, figsize=[5, 4])
    λs = runaway_kernel(Λ0=Λ0, mth=mth)(ms1, ms2)
    im = ax.imshow(λs.T, interpolation='bilinear', origin='lower', aspect='auto',
            extent=[np.log10(ms0[0]), np.log10(ms0[-1]), np.log10(ms0[0]), np.log10(ms0[-1])],
            cmap='rainbow', norm=matplotlib.colors.LogNorm())
    cb = fig.colorbar(im, ax=ax, pad=0.1)
    cb.set_label(r"interaction rate")
    ax.text(np.log10(ms0[0]) + np.log10(ms0[-1]/ms0[0])*0.04, np.log10(ms0[0]) + np.log10(ms0[-1]/ms0[0])*0.96, 'raw interaction rate\n$\\lambda(m_1,m_2)$', size=10, ha="left", va="top",
        bbox=dict(boxstyle="round", ec=(0.2, 0.2, 0.2), fc=(0.95, 0.95, 0.95)))
    planets.plot.set_fake_log_axes(ax=ax, axes='both', minor_ticks=True)
    #ax.grid(which='major', axis='both')
    ax.axhline(np.log10(mth), label=r'threshold mass $m_{\mathrm{th}}$', linestyle='--', color='white')
    ax.axvline(np.log10(mth), linestyle='--', color='white')
    ax.set_xlabel('particle mass $m_1$')
    ax.set_ylabel('particle mass $m_2$')
    ax.legend(loc='lower left', framealpha=0.65)
    fig.set_tight_layout(True)
    fig.savefig(filename)
    plt.close(fig)

    filename = os.path.join(dirpath, '{} RP-swarm{}'.format(basename, ext).replace(' ', '_'))
    print("Writing plot '{}'.".format(filename))
    fig, ax = plt.subplots(1, 1, sharex=False, sharey=False, figsize=[5, 4])
    λs = runaway_kernel_representative(Λ0=Λ0, mth=mth, Nth=Nth, M_n=M/n)(ms1, ms2)
    im = ax.imshow(λs.T, interpolation='bilinear', origin='lower', aspect='auto',
            extent=[np.log10(ms0[0]), np.log10(ms0[-1]), np.log10(ms0[0]), np.log10(ms0[-1])],
            cmap='plasma', norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax))
    cb = fig.colorbar(im, ax=ax, pad=0.1)
    cb.set_label(r"interaction rate")
    ax.text(np.log10(ms0[0]) + np.log10(ms0[-1]/ms0[0])*0.04, np.log10(ms0[0]) + np.log10(ms0[-1]/ms0[0])*0.96, 'RP–swarm interaction rate\n$\\lambda_{\\mathrm{ent}}(\\mathrm{\\mathbf{q}}_1,\\mathrm{\\mathbf{q}}_2,0)$', size=10, ha="left", va="top",
        bbox=dict(boxstyle="round", ec=(0.2, 0.2, 0.2), fc=(0.95, 0.95, 0.95)))
    planets.plot.set_fake_log_axes(ax=ax, axes='both', minor_ticks=True)
    #ax.grid(which='major', axis='both')
    ax.axhline(np.log10(mth), label=r'threshold mass $m_{\mathrm{th}}$', linestyle='--', color='white')
    ax.axvline(np.log10(mth), linestyle='--', color='white')
    ax.axhline(np.log10(mcrit), label=r'critical mass $m_{\mathrm{crit}} = M/(n\, N_{\mathrm{th}})$', linestyle='dotted', color='lime')
    ax.set_xlabel('particle mass $m_1$')
    ax.set_ylabel('swarm particle mass $m_2$')
    ax.legend(loc='lower left', framealpha=0.65)
    fig.set_tight_layout(True)
    fig.savefig(filename)
    plt.close(fig)

def plot_runaway_kernel_scatter(base_filename, ext, cmd_args):
    from scipy.special import lambertw

    import planets.simulation
    import planets.plot

    from numpy.random import default_rng
    rng = default_rng(seed=42)

    Nys_ns_Nths = [(48, 1024, 10), (48, 16384, 10)]

    dirpath, basename = os.path.split(base_filename)
    os.makedirs(dirpath, exist_ok=True)

    mth = 1.e+7
    Nth = 10
    m0 = 1.
    M = 1.e+10
    Nm = 64
    Nx = 128
    Λ0 = 1.e-9
    #ms0 = np.logspace(np.log10(1.e+0), np.log10(M), Nm)
    #ms1 = ms0[..., np.newaxis]
    #ms2 = ms0[np.newaxis, ...]

    mavg_mstd_dict = { }
    num_repetitions = 1
    seed = 42
    for Ny, n, Nth in Nys_ns_Nths:
        mcrit = M/(n*Nth)

        img = np.zeros(shape=[Ny, Nx])
        mmaxnum = np.zeros(shape=[Nx], dtype=int)
        mmaxsum = np.zeros(shape=[Nx])
        mmaxsqsum = np.zeros(shape=[Nx])
        for repetition in range(num_repetitions):
            args = planets.simulation.params.make_namespace()
            args.collisions.runaway_collision_rate_coefficient = Λ0
            args.simulation.random_seed = seed
            seed += 1
            args.simulation.method = 'rpmc'
            args.simulation.effects = 'collisions'
            args.collisions.kernel = 'runaway'
            args.collisions.runaway_critical_mass = mth
            args.simulation.mass_growth_factor = 0.05
            args.simulation.particle_regime_threshold = Nth
            args.simulation.m_bins_per_decade = 2

            m0avg = 1.
            args.planetesimal.m = -m0avg*(1 + np.real(lambertw(-rng.uniform(size=n)/np.exp(1), k=-1)))
            args.planetesimal.M = M
            m0 = np.min(args.planetesimal.m)

            args.simulation.nPlt = n
            args.simulation.nPltR = (Nth - 1)*n
            args.simulation.nE   = 0
            args.simulation.tMinLog = 1
            args.simulation.NSteps = 2*Nx

            args.ring.r = 1.
            args.ring.Δr = 0.

            t_snap = [60]
            args.simulation.T = 80

            histograms_edges = []
            mmaxs = np.zeros(shape=[args.simulation.NSteps//2])
            mavgs = np.zeros(shape=[args.simulation.NSteps//2])
            mstds = np.zeros(shape=[args.simulation.NSteps//2])
            mavg_mstd_dict[(n, Nth)] = (mmaxs, mavgs, mstds)
            with planets.simulation.run(args=args, rng=rng, log=cmd_args.log, timesteps={ 'snap': t_snap }) as sim_data:
                _, data_by_time = sim_data.snapshots_by_time()

                if repetition == 0:
                    for t in t_snap:
                        data = data_by_time(t)
                        ms_snap = data.m.to_numpy().copy()
                        Ms_snap = data.M.to_numpy().copy()

                ts_linear = sim_data.timesteps()['linear']
                data_by_ts_linear = []
                for i, t in enumerate(ts_linear):
                    data = data_by_time(t)
                    ms = data.m.to_numpy()
                    Ms = data.M.to_numpy()
                    Ns = data.N.to_numpy()
                    mask = Ms > 0
                    mms = ms[mask]
                    mMs = Ms[mask]
                    mNs = Ns[mask]
                    N = np.sum(mNs)
                    logms = np.log10(mms)
                    mean = np.sum(mNs*logms)/N
                    var = np.sum(mNs*np.square(logms - mean))/N
                    mmaxs[i] = np.max(mms)
                    mavgs[i] = mean
                    mstds[i] = np.sqrt(var)
                    imax = np.argmax(ms)
                    mmax = ms[imax]
                    if mmax > 0:
                        mmaxnum[i] += 1
                        mmaxsum[i] += mmax
                        mmaxsqsum[i] += mmax**2
                        Ms[imax] -= mmax
                        Ns[imax] -= 1
                        if Ns[imax] < 0.1:
                            ms[imax] = 0
                            Ms[imax] = 0
                            Ns[imax] = 0
                    data_by_ts_linear.append(planets.plot.positive_finite(ms, Ms))

                hist, edges = projection.project_log_histograms(data_by_ts_linear, Ny, vmin=1., vmax=M)
                bins = np.sqrt(edges[1:]*edges[:-1])
                np.add(img, hist/M, out=img)

        #cmap = plt.cm.Set1(np.linspace(0, 1, num=9))
        #Ym0, Ym1 = bins[0], bins[-1]
        #filename = os.path.join(dirpath, '{} masses n={} Nth={}{}'.format(basename, n, Nth, ext).replace(' ', '_'))
        #print("Writing plot '{}'.".format(filename))
        #fig, ax = plt.subplots(1, 1, sharex=False, sharey=False, figsize=[5, 4])
        #im = ax.imshow(img/num_repetitions, interpolation='nearest', origin='lower',
        #               extent=[ts_linear[0], ts_linear[-1], np.log10(Ym0), np.log10(Ym1)],
        #               cmap='Blues', norm=matplotlib.colors.LogNorm(vmin=1.e-6, vmax=1.))
        #mmaxavg = mmaxsum/mmaxnum
        #mmaxstd = np.sqrt(mmaxsqsum/mmaxnum - mmaxavg**2)
        #ax.text(3, np.log10(M), r"$n = {:_}$".format(n).replace('_', r'\,'), size=10, ha="left", va="top",
        #    bbox=dict(boxstyle="round", ec=(0.2, 0.2, 0.2), fc=(0.95, 0.95, 0.95)))
        #ax.plot(ts_linear, np.log10(mmaxavg), color=cmap[0])
        #ax.plot(ts_linear, np.log10(mmaxavg - mmaxstd), color=cmap[0], linestyle='--', linewidth=0.75)
        #ax.plot(ts_linear, np.log10(mmaxavg + mmaxstd), color=cmap[0], linestyle='--', linewidth=0.75)
        #ax.fill_between(ts_linear, np.log10(mmaxavg - mmaxstd), np.log10(mmaxavg + mmaxstd), color=cmap[0], alpha=0.25)
        #cb = fig.colorbar(im, ax=ax, pad=0.1)
        #ax.axhline(np.log10(args.collisions.runaway_critical_mass), linestyle='dotted', color='red')
        #ax.axhline(np.log10(M/(n*Nth)), linestyle='dashed', color='orange')
        #ax.set_aspect(ts_linear[-1] / np.log10(Ym1/Ym0))
        #ax.set_ylim(np.log10(1), np.log10(M*2))
        #planets.plot.set_fake_log_axes(ax=ax, axes='y', minor_ticks=True)
        #ax.grid(which='major', axis='both')
        ##ax.grid(which='minor', axis='y', linestyle='--', color='lightgray')
        #ax.set_xlabel('time (dimensionless)')
        #ax.set_ylabel(r'mass (dimensionless)')
        #fig.set_tight_layout(True)
        #fig.savefig(filename)
        #plt.close(fig)

        filename = os.path.join(dirpath, '{} scatter n={} Nth={}{}'.format(basename, n, Nth, ext).replace(' ', '_'))
        print("Writing plot '{}'.".format(filename))
        mask = Ms_snap > 0
        ms0m = ms_snap[mask]
        Ms0m = Ms_snap[mask]
        n_snap = len(ms0m)
        ms1 = ms0m[..., np.newaxis]
        ms2 = ms0m[np.newaxis, ...]
        Ms2 = Ms0m[np.newaxis, ...]
        λs = runaway_kernel_weighted(Λ0=Λ0, mth=mth, Nth=Nth)(ms1, ms2, Ms2)
        fig, ax = plt.subplots(1, 1, sharex=False, sharey=False, figsize=[5, 4])
        m0 = 1.e+3  
        #Nproj = int(np.log10(M/m0))*2
        Nproj = 7*50
        img = np.zeros(shape=[Nproj, Nproj])
        ims0 = np.minimum(Nproj - 1, (np.log10(ms0m/m0)*Nproj/(np.log10(M/m0))))
        ims = ims0.astype(dtype=int)
        print(np.unique(ims))
        #ixs, iys = np.tile(ims, n_snap), np.repeat(ims, n_snap)
        #img[ixs,iys] += np.ravel(λs)
        for im1,ix in enumerate(ims):
            for im2,iy in enumerate(ims):
                img[ix,iy] += λs[im1,im2]
                #if ix > 380:
                    #print(ix,iy,img[ix,iy])
        λmin = 1.e-6
        im = ax.imshow(np.maximum(img.T, λmin), interpolation='antialiased', origin='lower',
                       extent=[np.log10(m0), np.log10(M), np.log10(m0), np.log10(M)],
                       cmap='inferno', norm=matplotlib.colors.LogNorm(vmin=λmin, vmax=None))
        cb = fig.colorbar(im, ax=ax, pad=0.1)
        #ax.scatter(np.tile(ms0m, n_snap), np.repeat(ms0m, n_snap), c=λs.ravel(), s=np.repeat(Ms2/M*n, n_snap), cmap='plasma_r',
        #    norm=matplotlib.colors.LogNorm())
        #ax.set_xscale('log')
        #ax.set_yscale('log')
        planets.plot.set_fake_log_axes(ax=ax, axes='both', minor_ticks=True)
        #ax.grid(which='major', axis='both')
        #ax.axhline(np.log10(mth), label=r'threshold mass $m_{\mathrm{th}}$', linestyle='--', color='black')
        #ax.axvline(np.log10(mth), linestyle='--', color='black')
        ax.set_xlabel('mass $m$')
        ax.set_ylabel('mass $m\'$')
        #ax.legend(loc='lower left')
        fig.set_tight_layout(True)
        fig.savefig(filename)
        plt.close(fig)


def plot_stirring_test(base_filename, ext, cmd_args, benchmark=False, r_AU=1, frag=False):

    from const.cgs import GG, year, MS, LS, AU, Mea, Mju, km

    Myr = 1.e+6*year

    from numpy.random import default_rng

    method_labels = {
        'rpmc': 'bucketing',
        'rpmc-traditional': 'traditional'
    }

    dirpath, basename = os.path.split(base_filename)

    # prepare args
    # load or (simulate and store)
    # plot

    Nth = cmd_args.Nth
    #n0 = cmd_args.n
    n0 = cmd_args.n_zone*cmd_args.Nzones
    nR0 = (Nth - 1)*n0
    #Ny = 64
    Ny = projection.num_histogram_bins(2*n0)
    nbin_factor = 1.5

    #mPs = [10*Mea]

    if cmd_args.load:
        pass  # TODO: implement
    else:
        args = planets.simulation.params.make_namespace()
        # TODO: fill in args

        #args.simulation.effects = 'stirring + friction + collisions + gas-drag'
        args.simulation.effects = 'stirring + friction + collisions'
        #args.simulation.effects = 'collisions + gas-drag'
        args.simulation.method = cmd_args.method
        if frag:
            args.collisions.outcomes = 'coagulation + fragmentation'
        else:
            args.collisions.outcomes = 'coagulation'
        args.collisions.kernel = 'geometric'

        #args.simulation.N_threshold = 
        #args.simulation.St_dust_threshold = 0
        #args.simulation.m_dust_threshold = 1.e+3*kg  # anything below 𝒪(compact car) is considered dust

        args.star.M      = MS
        args.star.L      = LS

        if r_AU == 1:
            args.disk.rMin   = 0.5*AU
            args.disk.rMax   = 1.5*AU
            args.gas.r0      = 1*AU
            args.gas.ρ0      = 1.2e-9  # (g/cm³)
            #args.gas.Tmin    = 20  # (K)
            args.gas.cs0     = 1.0e+5  # (cm/s)
        elif r_AU == 6:
            args.disk.rMin   = 5*AU
            args.disk.rMax   = 7*AU
            args.gas.r0      = 6*AU
            args.gas.ρ0      = 9.5e-12  # (g/cm³)
            #args.gas.Tmin    = 20  # (K)
            args.gas.cs0     = 6.2e+4  # (cm/s)

        args.gas.rmin    = 0.1*AU
        args.gas.rmax    = 400*AU
        args.gas.α       = 1.e-3
        args.gas.φ       = 0.05
        args.gas.p       = -1.

        args.gas.profile     = 'power-law'

        if r_AU == 1:
            args.ring.r             = 1*AU
            args.zones.ΔrMin        = 6.3e-4*AU
            args.planetesimal.ρ     = 3.  # (g/cm³)
            Σ0 = 16.7  # (g/cm²)
            args.planetesimal.Δv    = 4.7e+2  # (4.7 m/s)
            m0 = 4.8e+18  # (g)
            args.planetesimal.m     = m0
        elif r_AU == 6:
            args.ring.r             = 6*AU
            args.zones.ΔrMin        = 6.3e-4*AU  # TODO
            args.planetesimal.ρ     = 1.  # (g/cm³)
            Σ0 = 2.  # (g/cm²)
            args.planetesimal.R     = 7.3*km
            m0 = 4/3*np.pi*args.planetesimal.ρ*args.planetesimal.R**3
            #args.planetesimal.e    = 0.002
            #args.planetesimal.sini = 0.002
            args.planetesimal.Δv    = 2.7e+2  # (2.7 m/s)
        #args.planetesimal.m     = m0/(1 - rng.uniform(size=n0)*(1 - np.sqrt(m0/m_max)))**2
        
        args.ring.Δr            = args.zones.ΔrMin*cmd_args.Nzones
        A = 2*np.pi*args.ring.r*args.ring.Δr
        args.planetesimal.M     = A*Σ0
        Rtot = (args.planetesimal.M/(4/3*np.pi*args.planetesimal.ρ))**(1/3)

        mcrit = args.planetesimal.M/(n0*Nth)
        Rcrit = (mcrit/(4/3*np.pi*args.planetesimal.ρ))**(1/3)

        args.collisions.ε     = 0.01
        args.collisions.Rfrag = 1.  # (cm)

        mfrag = 4/3*np.pi*args.planetesimal.ρ*args.collisions.Rfrag**3

        if r_AU == 1:
            #args.simulation.T = 2.e+5*year
            args.simulation.T = 1.8e+5*year
            #args.simulation.T = 2.5e+4*year
            #args.simulation.T = 2.5e+2*year
            #args.simulation.T = 5.e+3*year
            #args.simulation.T = 0.0267867*Myr
        elif r_AU == 6:
            #args.simulation.T = 2.e+6*year
            args.simulation.T = 1.74118e+6*year
        #args.simulation.T = 4.e+5*year
        args.simulation.tMinLog = 1.e+0*year
        args.simulation.NSteps = 512 if not benchmark else 4
        #args.simulation.NSteps = 32 if not benchmark else 4
        #args.simulation.hierarchical_ordering_base = np.sqrt(10)
        args.simulation.random_seed = 42

        args.simulation.mass_growth_factor = 0.05
        #args.simulation.velocity_growth_factor = 0.05
        #args.simulation.velocity_growth_factor = 0.
        #args.simulation.velocity_growth_factor = 0.005
        #args.simulation.velocity_growth_rate = np.sqrt(GG*args.star.M/args.ring.r**3)
        #args.simulation.m_bins_per_decade = 3
        #args.simulation.e_bins_per_decade = 2
        ##args.simulation.r_bins = 14
        ##args.simulation.r_bins = 0.1

        # Permit overriding simulation parameters with command-line arguments.
        planets.simulation.params.copy(dest=args, src=cmd_args, filter_pred=tools.parameters.FilterPredicates.not_None)
        planets.simulation.fill_implicit_args(args)

        if 'locality' in args.simulation.options:
            args.simulation.r_bins = 0.01  # no radial bucketing if we have radial sub-bucketing

        args.simulation.particle_regime_threshold = Nth
        args.simulation.nPlt = n0
        args.simulation.nPltR = nR0
        args.simulation.nE   = 0

        rng = default_rng(seed=1337+args.simulation.random_seed)

    n = args.simulation.nPlt + args.simulation.nPltR

    if benchmark:
        timesteps = { }
        inspect_callback = None
    else:
        if r_AU == 1:
            #ts_cum = np.array([0., 1.e+5, 2.e+5, 4.e+5])*year
            ts_cum = np.array([0., 5.e+2, 1.e+4, 2.5e+4, 5.e+4, 1.e+5, 1.5e+5])*year
            #ts_snap = np.array([0., 1.e+3, 2.e+3, 4.e+3, 1.e+4, 2.e+4])*year
            ts_snap = np.array([0., 1.e+1, 1.e+2, 2.5e+2, 5.e+2, 1.e+3, 2.5e+3, 5.e+3, 1.e+4, 1.3e+4, 2.5e+4, 3.8e+4, 5.e+4, 1.e+5, 1.5e+5, 1.8e+5, 2.5e+5])*year
            #ts_inspect = np.array([0., 1.e+1, 1.e+2, 1.e+3, 2.e+3, 4.e+3, 1.e+4, 2.e+4])*year
            #ts_inspect = np.array([])
            #ts_inspect = np.array([0.0267867*Myr])
            ts_inspect = np.array([0., 5.e+2, 5.e+3, 1.e+4, 2.5e+4, 5.e+4, 1.e+5, 1.5e+5])*year
            #ts_inspect = np.array([1.5e+5])*year
        elif r_AU == 6:
            ts_cum = np.array([0., 1.e+5, 2.e+5, 4.e+5])*year
            ts_snap = np.array([0., 1.e+3, 2.e+3, 4.e+3, 1.e+4, 2.e+4, 4.e+4, 1.e+5, 2.e+5, 4.e+5, 1.e+6, 1.74118e+6])*year
            #ts_inspect = np.array([0., 1.e+1, 1.e+2, 1.e+3, 2.e+3, 4.e+3, 1.e+4, 2.e+4, 4.e+4, 1.e+5, 2.e+5, 4.e+5, 1.e+6, 2.e+6])*year
            ts_inspect = np.array([0., 1.e+1, 1.e+2, 1.e+3, 2.e+3, 4.e+3, 1.e+4, 2.e+4, 4.e+4, 1.e+5, 2.e+5, 4.e+5, 1.e+6, 1.2e+6, 1.4e+6, 1.5e+6, 1.6e+6, 1.74118e+6])*year
        ts_cum = ts_cum[ts_cum <= args.simulation.T]
        ts_snap = ts_snap[ts_snap <= args.simulation.T]
        ts_inspect = ts_inspect[ts_inspect <= args.simulation.T]
            #ts_inspect = np.array([])
        #ts_cum = np.array([0.])*year
        #ts_cum = np.array([0., 1.e+3, 1.e+4, 1.e+5])*year
        #ts_cum = np.array([0., 1.e+3, 1.e+4, 1.6e+4])*year
        #ts_snap = np.array([0., 1.e+3, 2.e+3, 4.e+3, 1.e+4, 2.e+4, 4.e+4, 1.e+5, 2.e+5, 4.e+5, 1.e+6, 2.e+6])*year
        #ts_snap = np.array([])*year
        #ts_snap = np.array([0., 1.e+3])*year
        #ts_snap = np.array([0.])*year
        #ts_snap = np.array([0., 1.e+3, 2.e+3, 4.e+3, 1.e+4, 1.6e+4])*year
        #ts_inspect = np.array([0., 1.e+1, 1.e+2, 1.e+3, 2.e+3, 4.e+3, 1.e+4, 2.e+4, 4.e+4, 1.e+5, 2.e+5, 4.e+5, 1.e+6, 2.e+6])*year
        #ts_inspect = np.array([0., 1.e+1, 1.e+2, 1.e+3, 2.e+3, 4.e+3, 1.e+4, 2.e+4])*year
        #ts_inspect = np.array([1.e+3])*year
        #ts_inspect = np.array([0.])*year
        #ts_inspect = np.array([0., 1.e+1, 1.e+2, 1.e+3, 2.e+3, 4.e+3, 1.e+4, 2.e+4, 4.e+4, 1.e+5, 2.e+5])*year
        #ts_inspect = np.array([0., 1.e+1, 1.e+2, 1.e+3, 2.e+3, 4.e+3, 1.e+4, 1.6e+4])*year
        if cmd_args.method != 'rpmc' or not cmd_args.log or not cmd_args.plot:
            ts_inspect = []
        timesteps = {
            'cum': ts_cum,
            'snap': ts_snap,
            'inspect': ts_inspect
        }
        interaction_rates = []
        bucket_interaction_rates = []
        bucket_gaps = []
        acc_probs = []
        interaction_radii = []
        bucket_interaction_radii = []
        radius_bucket_gaps = []
        radius_acc_probs = []
        masses = []
        swarm_masses = []
        eccentricities = []
        semimajor_axes = []
        names = ['collision', 'stirring', 'friction']
        def inspect_callback(sim, state, i, t):
            params = np.array([0], dtype=float)
            num_active_a = 0
            num_active_bc = 0
            if 'collisions' in args.simulation.effects:
                dst0a = np.zeros(shape=[1], dtype=np.intp)
                sim.inspect(dst0a, 'discrete-operator-0/num-active-particles', params)
                num_active_a = dst0a[0]
            if 'stirring' in args.simulation.effects:
                dst0bc = np.zeros(shape=[1], dtype=np.intp)
                sim.inspect(dst0bc, 'discrete-operator-1/num-active-particles', params)
                num_active_bc = dst0bc[0]
            dst0bc = np.zeros(shape=[1], dtype=np.intp)
            dst1a = np.zeros(shape=[num_active_a, num_active_a], dtype=float)
            dst1b = np.zeros(shape=[num_active_bc, num_active_bc], dtype=float)
            dst1c = np.zeros(shape=[num_active_bc, num_active_bc], dtype=float)
            dst2a = np.zeros(shape=[num_active_a, num_active_a], dtype=float)
            dst2b = np.zeros(shape=[num_active_bc, num_active_bc], dtype=float)
            dst2c = np.zeros(shape=[num_active_bc, num_active_bc], dtype=float)
            dst3a = np.zeros(shape=[num_active_a, num_active_a], dtype=float)
            dst3b = np.zeros(shape=[num_active_bc, num_active_bc], dtype=float)
            dst3c = np.zeros(shape=[num_active_bc, num_active_bc], dtype=float)
            dst4a = np.zeros(shape=[num_active_a, num_active_a], dtype=float)
            dst4b = np.zeros(shape=[num_active_bc, num_active_bc], dtype=float)
            dst4c = np.zeros(shape=[num_active_bc, num_active_bc], dtype=float)
            if 'locality' in args.simulation.options:
                dst1ar = np.zeros(shape=[num_active_a, num_active_a], dtype=float)
                dst1br = np.zeros(shape=[num_active_bc, num_active_bc], dtype=float)
                dst1cr = np.zeros(shape=[num_active_bc, num_active_bc], dtype=float)
                dst2ar = np.zeros(shape=[num_active_a, num_active_a], dtype=float)
                dst2br = np.zeros(shape=[num_active_bc, num_active_bc], dtype=float)
                dst2cr = np.zeros(shape=[num_active_bc, num_active_bc], dtype=float)
                dst3ar = np.zeros(shape=[num_active_a, num_active_a], dtype=float)
                dst3br = np.zeros(shape=[num_active_bc, num_active_bc], dtype=float)
                dst3cr = np.zeros(shape=[num_active_bc, num_active_bc], dtype=float)
                dst4ar = np.zeros(shape=[num_active_a, num_active_a], dtype=float)
                dst4br = np.zeros(shape=[num_active_bc, num_active_bc], dtype=float)
                dst4cr = np.zeros(shape=[num_active_bc, num_active_bc], dtype=float)
            indicesa = np.zeros(shape=[num_active_a], dtype=np.intp)
            indicesbc = np.zeros(shape=[num_active_bc], dtype=np.intp)
            if 'collisions' in args.simulation.effects:
                sim.inspect(dst1a, 'discrete-operator-0/interaction-model-0/interaction rates', params)
                sim.inspect(dst2a, 'discrete-operator-0/interaction-model-0/particle bucket interaction rates', params)
                sim.inspect(dst3a, 'discrete-operator-0/interaction-model-0/acceptance probabilities', params)
                #sim.inspect(dst4a, 'discrete-operator-0/interaction-model-0/particle bucket gap', params)
                sim.inspect(dst4a, 'discrete-operator-0/interaction-model-0/true particle bucket interaction rates', params)
                sim.inspect(indicesa, 'discrete-operator-0/indices', params)
                #sim.inspect([], 'discrete-operator-0/hierarchical ordering', [])
                if 'locality' in args.simulation.options:
                    sim.inspect(dst1ar, 'discrete-operator-0/interaction-model-0/interaction radii', params)
                    sim.inspect(dst2ar, 'discrete-operator-0/interaction-model-0/particle bucket interaction radii', params)
                    #sim.inspect(dst3ar, 'discrete-operator-0/interaction-model-0/in-reach probabilities', params)
                    sim.inspect(dst3ar, 'discrete-operator-0/interaction-model-0/interaction rates, reverse', params)
                    #sim.inspect(dst4ar, 'discrete-operator-0/interaction-model-0/particle reach bucket gap', params)
                    sim.inspect(dst4ar, 'discrete-operator-0/interaction-model-0/true particle bucket interaction rates, reverse', params)
            if 'stirring' in args.simulation.effects:
                #sim.inspect([], 'discrete-operator-1/hierarchical ordering', [])
                sim.inspect(dst1b, 'discrete-operator-1/interaction-model-0/interaction rates', params)
                sim.inspect(dst1c, 'discrete-operator-1/interaction-model-1/interaction rates', params)
                sim.inspect(dst2b, 'discrete-operator-1/interaction-model-0/particle bucket interaction rates', params)
                sim.inspect(dst2c, 'discrete-operator-1/interaction-model-1/particle bucket interaction rates', params)
                sim.inspect(dst3b, 'discrete-operator-1/interaction-model-0/acceptance probabilities', params)
                sim.inspect(dst3c, 'discrete-operator-1/interaction-model-1/acceptance probabilities', params)
                #sim.inspect(dst4b, 'discrete-operator-1/interaction-model-0/particle bucket gap', params)
                sim.inspect(dst4b, 'discrete-operator-1/interaction-model-0/true particle bucket interaction rates', params)
                #sim.inspect(dst4c, 'discrete-operator-1/interaction-model-1/particle bucket gap', params)
                sim.inspect(dst4c, 'discrete-operator-1/interaction-model-1/true particle bucket interaction rates', params)
                if 'locality' in args.simulation.options:
                    sim.inspect(dst1br, 'discrete-operator-1/interaction-model-0/interaction radii', params)
                    sim.inspect(dst1cr, 'discrete-operator-1/interaction-model-1/interaction radii', params)
                    sim.inspect(dst2br, 'discrete-operator-1/interaction-model-0/particle bucket interaction radii', params)
                    sim.inspect(dst2cr, 'discrete-operator-1/interaction-model-1/particle bucket interaction radii', params)
                    #sim.inspect(dst3br, 'discrete-operator-1/interaction-model-0/in-reach probabilities', params)
                    #sim.inspect(dst3cr, 'discrete-operator-1/interaction-model-1/in-reach probabilities', params)
                    sim.inspect(dst3br, 'discrete-operator-1/interaction-model-0/interaction rates, reverse', params)
                    sim.inspect(dst3cr, 'discrete-operator-1/interaction-model-1/interaction rates, reverse', params)
                    #sim.inspect(dst4br, 'discrete-operator-1/interaction-model-0/particle reach bucket gap', params)
                    #sim.inspect(dst4cr, 'discrete-operator-1/interaction-model-1/particle reach bucket gap', params)
                    sim.inspect(dst4br, 'discrete-operator-1/interaction-model-0/true particle bucket interaction rates, reverse', params)
                    sim.inspect(dst4cr, 'discrete-operator-1/interaction-model-1/true particle bucket interaction rates, reverse', params)
                sim.inspect(indicesbc, 'discrete-operator-1/indices', params)
            interaction_rates.append((dst1a, dst1b, dst1c))
            bucket_interaction_rates.append((dst2a, dst2b, dst2c))
            acc_probs.append((dst3a, dst3b, dst3c))
            bucket_gaps.append((dst4a, dst4b, dst4c))
            if 'locality' in args.simulation.options:
                interaction_radii.append((dst1ar, dst1br, dst1cr))
                bucket_interaction_radii.append((dst2ar, dst2br, dst2cr))
                radius_acc_probs.append((dst3ar, dst3br, dst3cr))
                radius_bucket_gaps.append((dst4ar, dst4br, dst4cr))
            else:
                interaction_radii.append((None, None, None))
                bucket_interaction_radii.append((None, None, None))
                radius_acc_probs.append((None, None, None))
                radius_bucket_gaps.append((None, None, None))

            m_all = state.m.to_numpy(copy=True)
            ma = m_all[indicesa]
            mbc = m_all[indicesbc]
            masses.append((ma, mbc, mbc))
            M_all = state.M.to_numpy(copy=True)
            Ma = M_all[indicesa]
            Mbc = M_all[indicesbc]
            swarm_masses.append((Ma, Mbc, Mbc))
            e_all = state.e.to_numpy(copy=True)
            ea = e_all[indicesa]
            ebc = e_all[indicesbc]
            eccentricities.append((ea, ebc, ebc))
            a_all = state.a.to_numpy(copy=True)
            aa = a_all[indicesa]
            abc = a_all[indicesbc]
            semimajor_axes.append((aa, abc, abc))

    if cmd_args.report:
        report_args(args)

    if not cmd_args.dry_run and not cmd_args.load:
        with planets.simulation.run(args=args, rng=rng, log=cmd_args.log, display=cmd_args.display,
                                    timesteps=timesteps, inspect_callback=inspect_callback) as sim_data:

            dummy = np.array([], dtype=np.intp)
            sim_data.sim.inspect(dummy, 'profiling-data', dummy)
            if 'collision' in args.simulation.effects and cmd_args.log or not benchmark:
                sim_data.sim.inspect(dummy, 'discrete-operator-0/statistics', dummy)
            if 'stirring' in args.simulation.effects:
                sim_data.sim.inspect(dummy, 'discrete-operator-1/statistics', dummy)

            if not benchmark:
                lbasename = '{} r={}AU Nzones={} n={} method={}{}{}{}'.format(basename, r_AU, cmd_args.Nzones, n, method_labels[cmd_args.method], ' loc' if 'locality' in args.simulation.options else '', ' ' if cmd_args.tag != '' else '', cmd_args.tag).replace(' ', '_')
                datadirpath = os.path.join(dirpath, lbasename)
                #plotdirpath = os.path.join(datadirpath, 'plots')
                plotdirpath = datadirpath

                # Save results.
                if not (cmd_args.load or cmd_args.do_not_store):
                    print("Saving snapshot data to '{}/'.".format(datadirpath))
                    config = planets.simulation.params.to_configuration(args=args, filter_pred=tools.parameters.FilterPredicates.not_None)
                    with tools.data.DirectoryArchiveWriter(dirname=datadirpath, config=config,
                                                           overwrite=cmd_args.overwrite, compress=cmd_args.compress,
                                                           prefer_binary=cmd_args.format == 'binary') as archive_writer:
                        planets.simulation.SimulationData.save(archive_writer, sim_data=sim_data)
                else:
                    os.makedirs(plotdirpath, exist_ok=cmd_args.overwrite)

                #_, data_by_time = sim_data.snapshots_by_time()
                ts_log = sim_data.timesteps()['log']
                data = sim_data.snapshots()
                rh = (data.m/(3*args.star.M))**(1/3)
                data['v_vh'] = data.e/rh
                data['vz_vh'] = data.sininc/rh
                data_by_time = data.groupby(data.t, as_index=False)

                data_mean = data_by_time.agg('mean')
                v_vh = data_mean.v_vh
                vz_vh = data_mean.vz_vh
                β = vz_vh/v_vh
                if cmd_args.plot:
                    filename = os.path.join(plotdirpath, 'velocities{}'.format(ext).replace(' ', '_'))
                    print("Writing plot '{}'.".format(filename))
                    fig, ax = plt.subplots(1, 1, sharex=False, sharey=False, figsize=[5, 4])
                    ax.plot(data_mean.t/year, v_vh, label='eccentricity')
                    ax.plot(data_mean.t/year, vz_vh, label='inclination')
                    ax.axhline(2.5, linestyle='dotted', label=r'$2.5 v_{\mathrm{h}}$', color='black')
                    ax.plot(data_mean.t/year, β, label=r'$\beta$', linestyle='--')
                    ax.set_xscale('log')
                    ax.set_yscale('log')
                    #ax.set_ylim(1.e-2, 3.e+1)
                    #ax.set_ylim(1.e-2, None)
                    ax.set_xlabel('time (yr)')
                    ax.set_ylabel(r'$v/v_{\mathrm{h}}$, $v_z/v_{\mathrm{h}}$')
                    ax.grid(axis='both', which='major')
                    ax.grid(axis='both', which='minor', linestyle='--', color='lightgray')
                    ax.legend()
                    fig.set_tight_layout(True)
                    fig.savefig(filename)
                    plt.close(fig)

                Y_w = lambda data: (data.m, data.M)
                data_by_ts_log = [planets.plot.positive_finite(*Y_w(data_by_time.get_group(t))) for t in ts_log]
                hist, bins = tools.projection.project_log_histograms(data_by_ts_log, Ny, nbin_factor=nbin_factor)
                Ym0, Ym1 = bins[0], bins[-1]
                mmin = m0 if not frag else mfrag
                #t0 = 1.e+4*year
                t0 = ts_log[0]
                if cmd_args.plot:
                    filename = os.path.join(plotdirpath, 'masses{}'.format(ext).replace(' ', '_'))
                    print("Writing plot '{}'.".format(filename))
                    fig, ax = plt.subplots(1, 1, sharex=False, sharey=False, figsize=[5, 4])
                    im = ax.imshow(hist/args.planetesimal.M, interpolation='nearest', origin='lower',
                                   extent=[np.log10(ts_log[0]/year), np.log10(ts_log[-1]/year), np.log10(Ym0/Mea), np.log10(Ym1/Mea)],
                                   cmap='Blues', norm=matplotlib.colors.LogNorm())
                    cb = fig.colorbar(im, ax=ax, pad=0.1)
                    ax.set_aspect(np.log10(ts_log[-1]/t0) / np.log10(args.planetesimal.M/mmin))
                    ax.set_xlim(np.log10(t0/year), None)
                    ax.set_ylim(np.log10(mmin/Mea), np.log10(args.planetesimal.M/Mea))
                    planets.plot.set_fake_log_axes(ax=ax, axes='both', minor_ticks=True)
                    ax.grid(which='major', axis='both')
                    if not frag:
                        ax.grid(which='minor', axis='y', linestyle='--', color='lightgray')
                    ax.set_xlabel('time (years)')
                    ax.set_ylabel(r'mass ($\mathrm{M_\oplus}$)')
                    fig.set_tight_layout(True)
                    fig.savefig(filename)
                    plt.close(fig)

                Y_w = lambda data: (data.M, data.M)
                data_by_ts_log = [planets.plot.positive_finite(*Y_w(data_by_time.get_group(t))) for t in ts_log]
                hist, bins = tools.projection.project_log_histograms(data_by_ts_log, Ny, nbin_factor=nbin_factor)
                Ym0, Ym1 = bins[0], bins[-1]
                mmin = m0 if not frag else mfrag
                #t0 = 1.e+4*year
                t0 = ts_log[0]
                if cmd_args.plot:
                    filename = os.path.join(plotdirpath, 'swarm-masses{}'.format(ext).replace(' ', '_'))
                    print("Writing plot '{}'.".format(filename))
                    fig, ax = plt.subplots(1, 1, sharex=False, sharey=False, figsize=[5, 4])
                    im = ax.imshow(hist/args.planetesimal.M, interpolation='nearest', origin='lower',
                                   extent=[np.log10(ts_log[0]/year), np.log10(ts_log[-1]/year), np.log10(Ym0/Mea), np.log10(Ym1/Mea)],
                                   cmap='Blues', norm=matplotlib.colors.LogNorm())
                    cb = fig.colorbar(im, ax=ax, pad=0.1)
                    ax.set_aspect(np.log10(ts_log[-1]/t0) / np.log10(args.planetesimal.M/mmin))
                    ax.set_xlim(np.log10(t0/year), None)
                    ax.set_ylim(np.log10(mmin/Mea), np.log10(args.planetesimal.M/Mea))
                    planets.plot.set_fake_log_axes(ax=ax, axes='both', minor_ticks=True)
                    ax.grid(which='major', axis='both')
                    if not frag:
                        ax.grid(which='minor', axis='y', linestyle='--', color='lightgray')
                    ax.set_xlabel('time (years)')
                    ax.set_ylabel(r'mass ($\mathrm{M_\oplus}$)')
                    fig.set_tight_layout(True)
                    fig.savefig(filename)
                    plt.close(fig)

                Y_w = lambda data: (data.R, data.M)
                data_by_ts_log = [planets.plot.positive_finite(*Y_w(data_by_time.get_group(t))) for t in ts_log]
                hist, bins = tools.projection.project_log_histograms(data_by_ts_log, Ny, nbin_factor=nbin_factor)
                Ym0, Ym1 = bins[0], bins[-1]
                Rmin = args.planetesimal.R if not frag else args.collisions.Rfrag
                #t0 = 1.e+4*year
                t0 = ts_log[0]
                if cmd_args.plot:
                    filename = os.path.join(plotdirpath, 'radii{}'.format(ext).replace(' ', '_'))
                    print("Writing plot '{}'.".format(filename))
                    fig, ax = plt.subplots(1, 1, sharex=False, sharey=False, figsize=[5, 4])
                    im = ax.imshow(hist/args.planetesimal.M, interpolation='nearest', origin='lower',
                                   extent=[np.log10(ts_log[0]/year), np.log10(ts_log[-1]/year), np.log10(Ym0/km), np.log10(Ym1/km)],
                                   cmap='Blues', norm=matplotlib.colors.LogNorm())
                    cb = fig.colorbar(im, ax=ax, pad=0.1)
                    ax.set_aspect(np.log10(ts_log[-1]/t0) / np.log10(Rtot/Rmin))
                    ax.set_xlim(np.log10(t0/year), None)
                    ax.set_ylim(np.log10(Rmin/km), np.log10(Rtot/km))
                    planets.plot.set_fake_log_axes(ax=ax, axes='both', minor_ticks=True)
                    ax.grid(which='major', axis='both')
                    ax.grid(which='minor', axis='y', linestyle='--', color='lightgray')
                    ax.set_xlabel('time (years)')
                    ax.set_ylabel(r'particle radius $R$ ($\mathrm{km}$)')
                    fig.set_tight_layout(True)
                    fig.savefig(filename)
                    plt.close(fig)

                Y_w = lambda data: (data.e, data.M)
                data_by_ts_log = [planets.plot.positive_finite(*Y_w(data_by_time.get_group(t))) for t in ts_log]
                hist, bins = tools.projection.project_log_histograms(data_by_ts_log, Ny, nbin_factor=nbin_factor)
                Ym0, Ym1 = bins[0], bins[-1]
                t0 = ts_log[0]
                e0 = 1.e-7
                if cmd_args.plot:
                    filename = os.path.join(plotdirpath, 'eccentricities{}'.format(ext).replace(' ', '_'))
                    print("Writing plot '{}'.".format(filename))
                    fig, ax = plt.subplots(1, 1, sharex=False, sharey=False, figsize=[5, 4])
                    im = ax.imshow(hist/args.planetesimal.M, interpolation='nearest', origin='lower',
                                   extent=[np.log10(ts_log[0]/year), np.log10(ts_log[-1]/year), np.log10(Ym0), np.log10(Ym1)],
                                   cmap='bone_r', norm=matplotlib.colors.LogNorm())
                    cb = fig.colorbar(im, ax=ax, pad=0.1)
                    ax.set_aspect(np.log10(ts_log[-1]/t0) / np.log10(1./e0))
                    ax.set_xlim(np.log10(t0/year), None)
                    ax.set_ylim(np.log10(e0), np.log10(1.))
                    planets.plot.set_fake_log_axes(ax=ax, axes='both', minor_ticks=True)
                    ax.grid(which='major', axis='both')
                    ax.grid(which='minor', axis='y', linestyle='--', color='lightgray')
                    ax.set_xlabel('time (years)')
                    ax.set_ylabel(r'eccentricity')
                    fig.set_tight_layout(True)
                    fig.savefig(filename)
                    plt.close(fig)

                Y_w = lambda data: (data.sininc, data.M)
                data_by_ts_log = [planets.plot.positive_finite(*Y_w(data_by_time.get_group(t))) for t in ts_log]
                hist, bins = tools.projection.project_log_histograms(data_by_ts_log, Ny, nbin_factor=nbin_factor)
                Ym0, Ym1 = bins[0], bins[-1]
                t0 = ts_log[0]
                sininc0 = 1.e-7
                if cmd_args.plot:
                    filename = os.path.join(plotdirpath, 'inclinations{}'.format(ext).replace(' ', '_'))
                    print("Writing plot '{}'.".format(filename))
                    fig, ax = plt.subplots(1, 1, sharex=False, sharey=False, figsize=[5, 4])
                    im = ax.imshow(hist/args.planetesimal.M, interpolation='nearest', origin='lower',
                                   extent=[np.log10(ts_log[0]/year), np.log10(ts_log[-1]/year), np.log10(Ym0), np.log10(Ym1)],
                                   cmap='bone_r', norm=matplotlib.colors.LogNorm())
                    cb = fig.colorbar(im, ax=ax, pad=0.1)
                    ax.set_aspect(np.log10(ts_log[-1]/t0) / np.log10(1./sininc0))
                    ax.set_xlim(np.log10(t0/year), None)
                    ax.set_ylim(np.log10(sininc0), np.log10(1.))
                    planets.plot.set_fake_log_axes(ax=ax, axes='both', minor_ticks=True)
                    ax.grid(which='major', axis='both')
                    ax.grid(which='minor', axis='y', linestyle='--', color='lightgray')
                    ax.set_xlabel('time (years)')
                    ax.set_ylabel(r'inclination ($\sin i$)')
                    fig.set_tight_layout(True)
                    fig.savefig(filename)
                    plt.close(fig)

                m_m0s = np.logspace(np.log10(1.), np.log10(2000.), Ny)
                Ncss = []
                for t in ts_cum:
                    data = data_by_time.get_group(t)
                    Ns = data.N.to_numpy()
                    ms = data.m.to_numpy()
                    Ncs = np.sum(Ns[..., np.newaxis]*(ms[..., np.newaxis] >= m0*m_m0s[np.newaxis, ...]), axis=0)
                    Ncss.append(Ncs)
                if cmd_args.plot:
                    filename = os.path.join(plotdirpath, 'cumulative-masses{}'.format(ext).replace(' ', '_'))
                    print("Writing plot '{}'.".format(filename))
                    fig, ax = plt.subplots(1, 1, sharex=False, sharey=False, figsize=[5, 4])
                    ax.set_xscale('log')
                    ax.set_yscale('log')
                    for t, Ncs in zip(ts_cum, Ncss):
                        ax.plot(m_m0s, Ncs, label=r'$t = {}\,\mathrm{{years}}$'.format(planets.plot.format_exp(t/year, digits=1)))
                    ax.grid(which='major', axis='both')
                    ax.grid(which='minor', axis='y', linestyle='--', color='lightgray')
                    ax.set_xlabel('time (years)')
                    ax.set_ylabel(r'cumulative number')
                    ax.legend()
                    fig.set_tight_layout(True)
                    fig.savefig(filename)
                    plt.close(fig)

                _minor_ticks_positions = np.concatenate([
                    np.array([k for k in range(3, 10)]),
                    np.array([k*10**i for k in range(2, 10) for i in range(1, 10)])])

                rss = []
                Rss = []
                Sss = []
                Css = []
                rmin = np.inf
                rmax = -np.inf
                Rmin = np.inf
                Rmax = -np.inf
                Cmin = np.inf
                Cmax = -np.inf
                for t in ts_snap:
                    data = data_by_time.get_group(t)
                    mask_active = data.M != 0
                    mask = mask_active & ((data.e != 0) | (data.sininc != 0))
                    mask_dust = mask_active & ((data.e == 0) & (data.sininc == 0))
                    rs, rs_dust = data[mask].a.to_numpy(), data[mask_dust].a.to_numpy()
                    Rs, Rs_dust = data[mask].R.to_numpy(), data[mask_dust].R.to_numpy()
                    Ss, Ss_dust = data[mask].M.to_numpy(), data[mask_dust].M.to_numpy()
                    es = data[mask].e.to_numpy()
                    sinincs = data[mask].sininc.to_numpy()
                    #ms, ms_dust = data[mask].m.to_numpy(), data[mask_dust].m.to_numpy()
                    #m_max = np.max(ms)
                    #rh_max = (m_max/(3*args.star.M))**(1/3)
                    #Cs = es/rh_max
                    #Cs = es
                    Cs = es
                    rss.append((rs, rs_dust))
                    Rss.append((Rs, Rs_dust))
                    Sss.append((Ss, Ss_dust))
                    Css.append(Cs)
                    rmin = min(rmin, np.min(rs, initial=np.inf), np.min(rs_dust, initial=np.inf))
                    rmax = max(rmax, np.max(rs, initial=0), np.max(rs_dust, initial=0))
                    #Rmin = min(Rmin, np.min(Rs, initial=np.inf), np.min(Rs_dust, initial=np.inf))
                    Rmin = min(Rmin, np.min(Rs, initial=np.inf))
                    #Rmax = max(Rmax, np.max(Rs))
                    Rmax = Rtot
                    #Cmin = min(Cmin, np.min(Cs))
                    #Cmax = max(Cmax, np.max(Cs))
                    Cmin = 3.e-5
                    Cmax = 1.e-1
                    color_norm = matplotlib.colors.LogNorm(vmin=Cmin, vmax=Cmax)
                    #color_norm = matplotlib.colors.Normalize(vmin=Cmin, vmax=Cmax)
                if cmd_args.plot:
                    for t, (rs, rs_dust), (Rs, Rs_dust), (Ss, Ss_dust), Cs in zip(ts_snap, rss, Rss, Sss, Css):
                        filename = os.path.join(plotdirpath, 'R-r t={}yr{}'.format(t/year, ext).replace(' ', '_'))
                        print("Writing plot '{}'.".format(filename))
                        xmin, xmax = Rmin/1.15/km, Rmax*1.15/km
                        ymin0, ymax0 = args.ring.r - args.ring.Δr/2, args.ring.r + args.ring.Δr/2
                        ymin, ymax = ymin0/AU, ymax0/AU
                        fig, ax = plt.subplots(1, 1, sharex=False, sharey=False, figsize=[5, 4])
                        ax.set_xlim(9.e-1, xmax)
                        ax.set_ylim(ymin, ymax)
                        ax.set_title(r'$t = {}\,\mathrm{{years}}$'.format(planets.plot.format_exp(t/year, digits=1)))
                        ax.set_xlabel('particle radius $R$ (km)')
                        ax.set_ylabel('orbital radius $r$ (AU)')
                        ax.set_xscale('log')
                        sc = ax.scatter(Rs/km, rs/AU, c=Cs, s=Ss/args.planetesimal.M*1.e+3, cmap='coolwarm', norm=color_norm, rasterized=True)
                        cb = fig.colorbar(sc, ax=ax)
                        #cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=color_norm, cmap='coolwarm'), ax=ax)
                        if len(Rs_dust) > 0:
                            sc_dust = ax.scatter(Rs_dust/km*1.e+5, rs_dust/AU, c='black', s=Ss_dust/args.planetesimal.M*1.e+3, rasterized=True)
                        ax.axvline(x=Rcrit/km, linestyle='--', color='gray')
                        cb.set_label(r'rms-eccentricity $\sqrt{\langle e^2 \rangle}$')
                        if frag:
                            dxf = 1.1
                            dy = 0.0006  # how big to make the diagonal lines in axes coordinates
                            xl = 1.4e+0
                            xr = 2.1e+0
                            kwargs = dict(color='k', clip_on=False, zorder=4)
                            ax.plot((xl, xr), (ymin, ymin), color='white', linewidth=1.5, clip_on=False, zorder=3)
                            ax.plot((xl, xr), (ymax, ymax), color='white', linewidth=1.5, clip_on=False, zorder=3)
                            ax.plot((xl/dxf, xl*dxf), (ymin+dy, ymin-dy), **kwargs)
                            ax.plot((xl/dxf, xl*dxf), (ymax+dy, ymax-dy), **kwargs)
                            ax.plot((xr/dxf, xr*dxf), (ymin+dy, ymin-dy), **kwargs)
                            ax.plot((xr/dxf, xr*dxf), (ymax+dy, ymax-dy), **kwargs)
                            ax.set_xticks([1.e+0, 1.e+1, 1.e+2, 1.e+3], ['$10^{-5}$', '$10^1$', '$10^2$', '$10^3$'])
                            ax.xaxis.set_minor_locator(matplotlib.ticker.FixedLocator(_minor_ticks_positions))
                        fig.set_tight_layout(True)
                        fig.savefig(filename, dpi=300)
                        plt.close(fig)

                if cmd_args.plot and cmd_args.log:
                    for t, dst1s, dst2s, dst3s, dst4s, dst1rs, dst2rs, dst3rs, dst4rs, mss, Mss, ess, ass, in zip(ts_inspect, interaction_rates, bucket_interaction_rates, acc_probs, bucket_gaps, interaction_radii, bucket_interaction_radii, radius_acc_probs, radius_bucket_gaps, masses, swarm_masses, eccentricities, semimajor_axes):
                        for dst1, dst2, dst3, dst4, dst1r, dst2r, dst3r, dst4r, ms, Ms, es, as_, name in zip(dst1s, dst2s, dst3s, dst4s, dst1rs, dst2rs, dst3rs, dst4rs, mss, Mss, ess, ass, names):
                            n_active = len(Ms)

                            filename = os.path.join(plotdirpath, 'interaction-rates {} t={}yr{}'.format(name, int(t/year), ext).replace(' ', '_'))
                            print("Writing plot '{}'.".format(filename))
                            fig2, ((ax2a, ax2b, ax2c, ax2d), (ax2ra, ax2rb, ax2rc, ax2rd), (ax2e, ax2f, ax2g, ax2h)) = plt.subplots(3, 4, sharex=False, sharey=False, figsize=[20, 12])
                            dst1_non0 = dst1[dst1 > 0]
                            dst2_non0 = dst2[dst2 > 0]
                            #vmin = min(np.min(dst1_non0) if len(dst1_non0) > 0 else 1., np.min(dst2_non0) if len(dst2_non0) > 0 else 1.)
                            vmax = max(np.max(dst1_non0) if len(dst1_non0) > 0 else 1., np.max(dst2_non0) if len(dst2_non0) > 0 else 1.)
                            vmin = vmax*1.e-8
                            #print("vmin: ", vmin, "vmax: ", vmax)
                            if vmin != vmax:
                                im = ax2a.imshow(dst1.T*year, interpolation='antialiased', origin='lower',
                                   extent=[0.5, n_active+0.5, 0.5, n_active+0.5],
                                   cmap='viridis', norm=matplotlib.colors.LogNorm(vmin=vmin*year, vmax=vmax*year))
                                fig2.colorbar(im, ax=ax2a)
                                im = ax2b.imshow(dst2.T*year, interpolation='antialiased', origin='lower',
                                   extent=[0.5, n_active+0.5, 0.5, n_active+0.5],
                                   cmap='viridis', norm=matplotlib.colors.LogNorm(vmin=vmin*year, vmax=vmax*year))
                                fig2.colorbar(im, ax=ax2b)
                                dst3[dst3 <= 0] = np.nan
                                im = ax2c.imshow(dst3.T, interpolation='antialiased', origin='lower',
                                   extent=[0.5, n_active+0.5, 0.5, n_active+0.5],
                                   cmap='inferno_r', norm=matplotlib.colors.LogNorm(vmin=1.e-2, vmax=1.))
                                fig2.colorbar(im, ax=ax2c)
                                #dst4[dst4 <= 0] = np.nan
                                #im = ax2d.imshow(dst4.T, interpolation='antialiased', origin='lower',
                                im = ax2d.imshow(dst4.T*year, interpolation='antialiased', origin='lower',
                                   extent=[0.5, n_active+0.5, 0.5, n_active+0.5],
                                #   cmap='inferno_r', norm=matplotlib.colors.LogNorm(vmin=1.e-2, vmax=1.))
                                   cmap='viridis', norm=matplotlib.colors.LogNorm(vmin=vmin*year, vmax=vmax*year))
                                fig2.colorbar(im, ax=ax2d)

                            ax2a.set_title(r'tracer–swarm interaction rates ($\mathrm{year}^{-1}$)')
                            ax2a.set_xlabel('tracer index')
                            ax2a.set_ylabel('swarm index')
                            ax2b.set_title(r'bucket bounds of tracer–swarm interaction rates ($\mathrm{year}^{-1}$)')
                            ax2b.set_xlabel('tracer index')
                            ax2b.set_ylabel('swarm index')
                            ax2c.set_title(r'acceptance probabilities')
                            ax2c.set_xlabel('tracer index')
                            ax2c.set_ylabel('swarm index')
                            #ax2d.set_title(r'particle bucket gap')
                            ax2d.set_title(r'true bucket bounds of tracer–swarm interaction rates ($\mathrm{year}^{-1}$)')
                            ax2d.set_xlabel('tracer index')
                            ax2d.set_ylabel('swarm index')

                            if 'locality' in args.simulation.options:
                                dst1r_non0 = dst1r[dst1r > 0]
                                dst2r_non0 = dst2r[dst2r > 0]
                                #vmin = min(np.min(dst1r_non0) if len(dst1r_non0) > 0 else 1., np.min(dst2r_non0) if len(dst2r_non0) > 0 else 1.)
                                vrmax = max(np.max(dst1r_non0) if len(dst1r_non0) > 0 else 1., np.max(dst2r_non0) if len(dst2r_non0) > 0 else 1.)
                                vrmin = vmax*1.e-8
                                #print("vmin: ", vmin, "vmax: ", vmax)
                                if vmin != vmax:
                                    im = ax2ra.imshow(dst1r.T, interpolation='antialiased', origin='lower',
                                       extent=[0.5, n_active+0.5, 0.5, n_active+0.5],
                                       cmap='viridis_r', norm=matplotlib.colors.LogNorm(vmin=vrmin, vmax=vrmax))
                                    fig2.colorbar(im, ax=ax2ra)
                                    im = ax2rb.imshow(dst2r.T, interpolation='antialiased', origin='lower',
                                       extent=[0.5, n_active+0.5, 0.5, n_active+0.5],
                                       cmap='viridis_r', norm=matplotlib.colors.LogNorm(vmin=vrmin, vmax=vrmax))
                                    fig2.colorbar(im, ax=ax2rb)
                                    #dst3r[dst3r <= 0] = np.nan
                                    #im = ax2rc.imshow(dst3r.T, interpolation='antialiased', origin='lower',
                                    im = ax2rc.imshow(dst3r.T*year, interpolation='antialiased', origin='lower',
                                       extent=[0.5, n_active+0.5, 0.5, n_active+0.5],
                                       #cmap='inferno_r', norm=matplotlib.colors.LogNorm(vmin=1.e-2, vmax=1.))
                                       cmap='viridis', norm=matplotlib.colors.LogNorm(vmin=vmin*year, vmax=vmax*year))
                                    fig2.colorbar(im, ax=ax2rc)
                                    #dst4r[dst4r <= 0] = np.nan
                                    #im = ax2rd.imshow(dst4r.T, interpolation='antialiased', origin='lower',
                                    im = ax2rd.imshow(dst4r.T*year, interpolation='antialiased', origin='lower',
                                       extent=[0.5, n_active+0.5, 0.5, n_active+0.5],
                                       #cmap='inferno_r', norm=matplotlib.colors.LogNorm(vmin=1.e-2, vmax=1.))
                                       cmap='viridis', norm=matplotlib.colors.LogNorm(vmin=vmin*year, vmax=vmax*year))
                                    fig2.colorbar(im, ax=ax2rd)

                                ax2ra.set_title(r'tracer–swarm interaction radii ($\mathrm{AU}$)')
                                ax2ra.set_xlabel('tracer index')
                                ax2ra.set_ylabel('swarm index')
                                ax2rb.set_title(r'bucket bounds of tracer–swarm interaction radii ($\mathrm{AU}$)')
                                ax2rb.set_xlabel('tracer index')
                                ax2rb.set_ylabel('swarm index')
                                #ax2rc.set_title(r'in-reach probabilities')
                                ax2rc.set_title(r'tracer–swarm interaction rates, reverse ($\mathrm{year}^{-1}$)')
                                ax2rc.set_xlabel('tracer index')
                                ax2rc.set_ylabel('swarm index')
                                #ax2rd.set_title(r'particle reach bucket gap')
                                ax2rd.set_title(r'true bucket bounds of tracer–swarm interaction rates, reverse ($\mathrm{year}^{-1}$)')
                                ax2rd.set_xlabel('tracer index')
                                ax2rd.set_ylabel('swarm index')

                            bMs = np.repeat(Ms, n_active).reshape([n_active, n_active])
                            bms = np.repeat(ms, n_active).reshape([n_active, n_active])
                            bes = np.repeat(es, n_active).reshape([n_active, n_active])
                            bas_AU = np.repeat(as_/AU, n_active).reshape([n_active, n_active])

                            im = ax2e.imshow(bMs.T, interpolation='antialiased', origin='lower',
                               extent=[0.5, n_active+0.5, 0.5, n_active+0.5],
                               cmap='Greys', norm=matplotlib.colors.LogNorm())
                            fig2.colorbar(im, ax=ax2e)
                            im = ax2f.imshow(bms.T, interpolation='antialiased', origin='lower',
                               extent=[0.5, n_active+0.5, 0.5, n_active+0.5],
                               cmap='Blues', norm=matplotlib.colors.LogNorm())
                            fig2.colorbar(im, ax=ax2f)
                            im = ax2g.imshow(bas_AU.T, interpolation='antialiased', origin='lower',
                               extent=[0.5, n_active+0.5, 0.5, n_active+0.5],
                               cmap='ocean', norm=matplotlib.colors.LogNorm())
                            fig2.colorbar(im, ax=ax2g)
                            im = ax2h.imshow(bes.T, interpolation='antialiased', origin='lower',
                               extent=[0.5, n_active+0.5, 0.5, n_active+0.5],
                               cmap='coolwarm', norm=matplotlib.colors.LogNorm())
                            fig2.colorbar(im, ax=ax2h)

                            ax2e.set_title('$M_j$')
                            ax2e.set_xlabel('tracer index')
                            ax2e.set_ylabel('swarm index')
                            ax2f.set_title('$m_j$')
                            ax2f.set_xlabel('tracer index')
                            ax2f.set_ylabel('swarm index')
                            ax2g.set_title('$a_j$ (AU)')
                            ax2g.set_xlabel('tracer index')
                            ax2g.set_ylabel('swarm index')
                            ax2h.set_title('$e_j$')
                            ax2h.set_xlabel('tracer index')
                            ax2h.set_ylabel('swarm index')

                            fig2.suptitle(r't = ${}$ years'.format(t/year))
                            fig2.set_tight_layout(True)
                            fig2.savefig(filename)
                            plt.close(fig2)


cmd_args = None
try:
    from const.cgs import Mea

    # Parse command line.
    if len(sys.argv) == 1:
        parser.print_usage()
        sys.exit(1)
    cmd_args = parser.parse_args()

    # Read parameter files and initialize default arguments.
    args = planets.simulation.params.make_namespace()
    planets.simulation.params.copy(dest=args, src=cmd_args, filter_pred=tools.parameters.FilterPredicates.not_None)

    # Check parameters.
    if len(cmd_args.simulations) == 0:
        raise RuntimeError('no simulations given')

    # Make plots.
    plots_dir = cmd_args.outdir
    ext = '.pdf'
    all = 'all' in cmd_args.simulations

    if all or '1' in cmd_args.simulations or '1.1' in cmd_args.simulations:
        caption = '1.1 runaway kernel'
        plot_filename = os.path.join(plots_dir, caption)
        print("[bucketing-plots] Plot '{}':".format(caption))
        plot_runaway_kernel(base_filename=plot_filename, ext=ext, cmd_args=cmd_args)

    if all or '1' in cmd_args.simulations or '1.2' in cmd_args.simulations:
        caption = '1.2 runaway kernel scatter'
        plot_filename = os.path.join(plots_dir, caption)
        print("[bucketing-plots] Plot '{}':".format(caption))
        plot_runaway_kernel_scatter(base_filename=plot_filename, ext=ext, cmd_args=cmd_args)

    if all or '1' in cmd_args.simulations or '1.3' in cmd_args.simulations:
        caption = '1.3 runaway kernel results'
        plot_filename = os.path.join(plots_dir, caption)
        print("[bucketing-plots] Writing plot '{}'.".format(caption))
        plot_runaway_kernel_tests(base_filename=plot_filename, ext=ext, cmd_args=cmd_args, benchmark=False)

    if all or '1' in cmd_args.simulations or '1.4' in cmd_args.simulations:
        caption = '1.4 runaway kernel benchmark'
        plot_filename = os.path.join(plots_dir, caption)
        print("[bucketing-plots] Running '{}'.".format(caption))
        print("n = {}".format(cmd_args.n))
        print("Nth = {}".format(cmd_args.Nth))
        print("method = {}".format(cmd_args.method))
        plot_runaway_kernel_tests(base_filename=plot_filename, ext=ext, cmd_args=cmd_args, benchmark=True)

    if all or '1' in cmd_args.simulations or '1.5' in cmd_args.simulations:
        caption = '1.5 runaway kernel perf plot'
        plot_filename = os.path.join(plots_dir, caption)
        print("[bucketing-plots] Plot '{}':".format(caption))
        plot_runaway_kernel_perf(base_filename=plot_filename, ext=ext, cmd_args=cmd_args)

    if all or '1' in cmd_args.simulations or '1.6' in cmd_args.simulations:
        caption = '1.6 runaway kernel inspection'
        plot_filename = os.path.join(plots_dir, caption)
        print("[bucketing-plots] Plot '{}':".format(caption))
        plot_runaway_kernel_rates(base_filename=plot_filename, ext=ext, cmd_args=cmd_args)

    if all or '2' in cmd_args.simulations or '2.1' in cmd_args.simulations:
        caption = '2.1 linear kernel results'
        plot_filename = os.path.join(plots_dir, caption)
        print("[bucketing-plots] Writing plot '{}'.".format(caption))
        plot_linear_kernel_tests(base_filename=plot_filename, ext=ext, cmd_args=cmd_args, benchmark=False)

    if all or '2' in cmd_args.simulations or '2.2' in cmd_args.simulations:
        caption = '2.2 linear kernel benchmark'
        plot_filename = os.path.join(plots_dir, caption)
        print("[bucketing-plots] Running '{}'.".format(caption))
        print("n = {}".format(cmd_args.n))
        print("Nth = {}".format(cmd_args.Nth))
        print("method = {}".format(cmd_args.method))
        plot_linear_kernel_tests(base_filename=plot_filename, ext=ext, cmd_args=cmd_args, benchmark=True)

    if all or '2' in cmd_args.simulations or '2.3' in cmd_args.simulations:
        caption = '2.3 linear kernel perf plot'
        plot_filename = os.path.join(plots_dir, caption)
        print("[bucketing-plots] Plot '{}':".format(caption))
        plot_linear_kernel_perf(base_filename=plot_filename, ext=ext, cmd_args=cmd_args)

    if all or '2' in cmd_args.simulations or '2.4' in cmd_args.simulations:
        caption = '2.4 constant kernel results'
        plot_filename = os.path.join(plots_dir, caption)
        print("[bucketing-plots] Writing plot '{}'.".format(caption))
        plot_constant_kernel_tests(base_filename=plot_filename, ext=ext, cmd_args=cmd_args, benchmark=False)

    if all or '3' in cmd_args.simulations or '3.1' in cmd_args.simulations:
        caption = '3.1 1AU stirring results'
        plot_filename = os.path.join(plots_dir, caption)
        print("[bucketing-plots] Writing plot '{}'.".format(caption))
        plot_stirring_test(base_filename=plot_filename, ext=ext, cmd_args=cmd_args, benchmark=False, r_AU=1)

    if all or '3' in cmd_args.simulations or '3.2' in cmd_args.simulations:
        caption = '3.2 6AU stirring results'
        plot_filename = os.path.join(plots_dir, caption)
        print("[bucketing-plots] Writing plot '{}'.".format(caption))
        plot_stirring_test(base_filename=plot_filename, ext=ext, cmd_args=cmd_args, benchmark=False, r_AU=6)

    if all or '3' in cmd_args.simulations or '3.3' in cmd_args.simulations:
        caption = '3.3 1AU stirring benchmark'
        plot_filename = os.path.join(plots_dir, caption)
        print("[bucketing-plots] Running '{}'.".format(caption))
        print("n_zone = {}".format(cmd_args.n_zone))
        print("Nzones = {}".format(cmd_args.Nzones))
        print("Nth = {}".format(cmd_args.Nth))
        print("method = {}".format(cmd_args.method))
        print("r-bins = {}".format(args.simulation.r_bins))
        print("M-bins-per-decade = {}".format(args.simulation.M_bins_per_decade))
        print("m-bins-per-decade = {}".format(args.simulation.m_bins_per_decade))
        print("e-bins-per-decade = {}".format(args.simulation.e_bins_per_decade))
        print("sininc-bins-per-decade = {}".format(args.simulation.sininc_bins_per_decade))
        print("velocity-growth-factor = {}".format(args.simulation.velocity_growth_factor))
        print("options = {}".format(args.simulation.options))
        plot_stirring_test(base_filename=plot_filename, ext=ext, cmd_args=cmd_args, benchmark=True, r_AU=1)

    if all or '3' in cmd_args.simulations or '3.4' in cmd_args.simulations:
        caption = '3.4 6AU stirring benchmark'
        plot_filename = os.path.join(plots_dir, caption)
        print("[bucketing-plots] Running '{}'.".format(caption))
        print("n = {}".format(cmd_args.n))
        print("Nth = {}".format(cmd_args.Nth))
        print("method = {}".format(cmd_args.method))
        print("r-bins = {}".format(args.simulation.r_bins))
        print("M-bins-per-decade = {}".format(args.simulation.M_bins_per_decade))
        print("m-bins-per-decade = {}".format(args.simulation.m_bins_per_decade))
        print("e-bins-per-decade = {}".format(args.simulation.e_bins_per_decade))
        print("sininc-bins-per-decade = {}".format(args.simulation.sininc_bins_per_decade))
        print("options = {}".format(args.simulation.options))
        plot_stirring_test(base_filename=plot_filename, ext=ext, cmd_args=cmd_args, benchmark=True, r_AU=6)

    if all or '3' in cmd_args.simulations or '3.5' in cmd_args.simulations:
        caption = '3.5 stirring perf plot'
        plot_filename = os.path.join(plots_dir, caption)
        print("[bucketing-plots] Plot '{}':".format(caption))
        plot_stirring_perf(base_filename=plot_filename, ext=ext, cmd_args=cmd_args)

    if all or '3' in cmd_args.simulations or '3.6' in cmd_args.simulations:
        caption = '3.6 stirring mem plot'
        plot_filename = os.path.join(plots_dir, caption)
        print("[bucketing-plots] Plot '{}':".format(caption))
        plot_stirring_mem(base_filename=plot_filename, ext=ext, cmd_args=cmd_args)

    if all or '3' in cmd_args.simulations or '3.7' in cmd_args.simulations:
        caption = '3.7 stirring widening parameter study plot'
        plot_filename = os.path.join(plots_dir, caption)
        print("[bucketing-plots] Plot '{}':".format(caption))
        plot_stirring_paramstudy(base_filename=plot_filename, ext=ext, cmd_args=cmd_args)

    if all or '3' in cmd_args.simulations or '3.8' in cmd_args.simulations:
        caption = '3.8 stirring bucket parameter study plot'
        plot_filename = os.path.join(plots_dir, caption)
        print("[bucketing-plots] Plot '{}':".format(caption))
        plot_stirring_paramstudy_buckets(base_filename=plot_filename, ext=ext, cmd_args=cmd_args)

    if all or '3' in cmd_args.simulations or '3.9' in cmd_args.simulations:
        caption = '3.9 1AU stirring with fragmentation results'
        plot_filename = os.path.join(plots_dir, caption)
        print("[bucketing-plots] Writing plot '{}'.".format(caption))
        plot_stirring_test(base_filename=plot_filename, ext=ext, cmd_args=cmd_args, benchmark=False, r_AU=1, frag=True)

    if all or '3' in cmd_args.simulations or '3.10' in cmd_args.simulations:
        caption = '3.10 1AU stirring with fragmentation benchmark'
        plot_filename = os.path.join(plots_dir, caption)
        print("[bucketing-plots] Running '{}'.".format(caption))
        print("n_zone = {}".format(cmd_args.n_zone))
        print("Nzones = {}".format(cmd_args.Nzones))
        print("Nth = {}".format(cmd_args.Nth))
        print("method = {}".format(cmd_args.method))
        print("r-bins = {}".format(args.simulation.r_bins))
        print("M-bins-per-decade = {}".format(args.simulation.M_bins_per_decade))
        print("m-bins-per-decade = {}".format(args.simulation.m_bins_per_decade))
        print("e-bins-per-decade = {}".format(args.simulation.e_bins_per_decade))
        print("sininc-bins-per-decade = {}".format(args.simulation.sininc_bins_per_decade))
        print("velocity-growth-factor = {}".format(args.simulation.velocity_growth_factor))
        print("options = {}".format(args.simulation.options))
        plot_stirring_test(base_filename=plot_filename, ext=ext, cmd_args=cmd_args, benchmark=True, r_AU=1, frag=True)


except RuntimeError as e:
    print('bucketing-plots.py: Error:', e, file=sys.stderr)
    if cmd_args is not None and cmd_args.log:
        import traceback
        traceback.print_exc()
