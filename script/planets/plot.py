
import math

import numpy as np

import scipy.special

import matplotlib
import matplotlib.pyplot as plt

import tools.projection


def sig(num, digits=3):
    "Return number formatted for significant digits"
    ## Taken from https://stackoverflow.com/a/67587629
    #if num == 0:
    #    return 0
    #negative = '-' if num < 0 else ''
    #num = abs(float(num))
    #power = math.log(num, 10)
    #if num < 1:
    #    step = int(10**(-int(power) + digits) * num)
    #    return negative + '0.' + '0' * -int(power) + str(int(step)).rstrip('0')
    #elif power < digits - 1:
    #    return negative + ('{0:.' + str(digits) + 'g}').format(num)
    #else:
    #    return negative + str(int(num))
    return np.format_float_positional(num, precision=digits)

def format_exp(x, base=10, digits=0, simplify=True):
    assert base > 1
    assert digits >= 0

    if x == 0.:
        return '0'
    #if x == 1.:
    #    return '1'
    exp = int(np.floor(np.log(np.abs(x*1.0000000000001))/np.log(base)))  # to avoid "10×10²"
    mantissa = x/base**exp
    mantissa_str = np.format_float_positional(mantissa, precision=digits, fractional=True, trim='-')
    if mantissa_str == '1' and simplify:
        return r'{}^{{{}}}'.format(base, exp)
    else:
        return r'{} \times {}^{{{}}}'.format(mantissa_str, base, exp)


# https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib
def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, 1 - amount * (1 - c[1]))), c[2])


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
def _fake_log(value, tick_position):
    return '$10^{{{}}}$'.format(int(value))

def positive_finite(Y, w):
    mask = (Y > 0.) & (w > 0.) & ~np.isinf(Y)
    return Y[mask], w[mask]

_minor_ticks_positions = np.array([i + np.log10(k) for k in range(2, 10) for i in range(-10, 10)])
def set_fake_log_axes(ax, axes='both', minor_ticks=True):
    if axes == 'both' or axes == 'x':
        ax.xaxis.set_major_formatter(_fake_log)
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1.))
        if minor_ticks:
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.xaxis.set_minor_locator(matplotlib.ticker.FixedLocator(_minor_ticks_positions))
    if axes == 'both' or axes == 'y':
        ax.yaxis.set_major_formatter(_fake_log)
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        if minor_ticks:
            ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1.))
            ax.yaxis.set_minor_locator(matplotlib.ticker.FixedLocator(_minor_ticks_positions))


def plot_snapshots(sim_data, ax, xscale = 'log', timescale = 'log', analytical = None, minor_ticks = True, yrange = 1.e+2, xrange = 4/3, Nhist = 256):
    t_all, group_by_time = sim_data.snapshots_by_time()
    t_few = sim_data.timesteps()['few']

    # Create histogram of snapshot times.
    mass_seq_few = [positive_finite(data.m, data.M) for data in (group_by_time(t) for t in t_few)]
    mass_hist_few, mass_bins_few = tools.projection.project_log_histograms(mass_seq_few, Nhist=Nhist, nbin_factor=1., homogeneous_bins=False)
    bin_centers = np.sqrt(mass_bins_few[1:]*mass_bins_few[:-1])
    bin_centers_2 = np.concatenate([bin_centers, (bin_centers*bin_centers[-1]/bin_centers[0])[1:int((xrange - 1.)*len(bin_centers))]])

    # Plot snapshots.
    max_density = 10**(np.ceil(np.log10(np.max(mass_hist_few))))
    #ax.set_ylim((max_density/1.e+2, max_density))
    ax.set_ylim((max_density/yrange, max_density))
    #ax1.set_xlim((1., bins[-1]*1.e2))
    ax.set_xscale(xscale)
    ax.set_yscale('log')
    cmap = plt.cm.rainbow(np.linspace(0, 1, len(t_few)))
    for t, hist, color in zip(t_few, mass_hist_few.transpose(), cmap):
        # Plot RPMC histogram.
        ax.fill_between(bin_centers, 0., hist, step='mid', color=color, alpha=0.15)

        # Plot analytical solution.
        if analytical is not None:
            fs_a = analytical(bin_centers_2, t)
            ax.plot(bin_centers_2, fs_a*bin_centers_2**2, ls='-', color=color)

    # Draw legend and axis labels.
    legend_elements = ([
        matplotlib.lines.Line2D([0], [0], marker='_', color='black', markersize=12)
    ] if analytical is not None else []) + [
        matplotlib.lines.Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=12)
    ] + ([
        (matplotlib.lines.Line2D([0], [0], marker='_', color=color, markersize=12), matplotlib.lines.Line2D([0], [0], marker='s', color='w', markerfacecolor=color, alpha=0.15, markersize=12))
        for color in cmap
    ] if analytical is not None else [
        matplotlib.lines.Line2D([0], [0], marker='s', color='w', markerfacecolor=color, alpha=0.15, markersize=12)
        for color in cmap
    ])
    legend_labels = ([
        'analytical solution'
    ] if analytical is not None else []) + [
        'RPMC'
    ] + ([
        '$t = 10^{{{:.0f}}}$'.format(np.rint(np.log10(t)))
        for t in t_few
    ] if timescale == 'log' else [
        '$t = {{{:.0f}}}$'.format(np.rint(t))
        for t in t_few
    ] if timescale == 'linear' else [
        '$t = {{{}}}$'.format(t)
        for t in t_few
    ])
    ax.legend(legend_elements, legend_labels, loc='lower left')
    #ax.grid(which='major', axis='both')
    #if minor_ticks:
    #    ax.grid(which='minor', axis='y', linestyle='--', color='lightgray')

def plot_histogram(sim_data, ax, Y_w, cmap, timescale, minor_ticks = True, colorbar_pad = 0.1):
    t_all, group_by_time = sim_data.snapshots_by_time()
    ts = sim_data.timesteps()[timescale]
    Nhist = len(ts)

    # Create histograms.
    seq = [positive_finite(*Y_w(data)) for data in (group_by_time(t) for t in ts)]
    hist_log, bins_log = tools.projection.project_log_histograms(seq, Nhist=Nhist, nbin_factor=1.)

    # Plot histograms.
    Y0, Y1 = bins_log[0], bins_log[-1]
    X0, X1 = (np.log10(ts[0]), np.log10(ts[-1])) if timescale == 'log' else (ts[0], ts[-1])
    im = ax.imshow(hist_log, interpolation='nearest', origin='lower',
        extent=[X0, X1, np.log10(Y0), np.log10(Y1)],
        cmap=cmap, norm=matplotlib.colors.LogNorm())
    cb = ax.get_figure().colorbar(im, ax=ax, pad=0.1)
    ax.set_aspect((np.log10(ts[-1]/ts[0]) if timescale == 'log' else ts[-1] - ts[0]) / np.log10(Y1/Y0))
    set_fake_log_axes(ax, axes='both' if timescale == 'log' else 'y', minor_ticks=minor_ticks)
    ax.grid(which='major', axis='both')
    if minor_ticks:
        ax.grid(which='minor', axis='y', linestyle='--', color='lightgray')
    return im, cb

def plot_mass_histogram(sim_data, ax, timescale, minor_ticks = True, colorbar_pad = 0.1):
    plot_histogram(sim_data=sim_data, ax=ax, Y_w=lambda data: (data.m, data.M), cmap=plt.get_cmap('Blues'),
        timescale=timescale, minor_ticks=minor_ticks, colorbar_pad=colorbar_pad)

def plot_num_histogram(sim_data, ax, timescale, minor_ticks = True, colorbar_pad = 0.1):
    plot_histogram(sim_data=sim_data, ax=ax, Y_w=lambda data: (data.N, data.M), cmap=plt.get_cmap('Greys'),
        timescale=timescale, minor_ticks=minor_ticks, colorbar_pad=colorbar_pad)


def analytical_solution_constant_kernel(m0, M, collision_rate, t0 = 0.):
    def func(m, t):
        τ = collision_rate*M/m0*(t - t0)
        g = 1/(1 + τ/2)
        k = m/m0
        N0 = M/m0
        f = N0*g**2*(1 - g)**(k - 1)
        return f
    return func

def analytical_solution_linear_kernel(m0avg, m0, M, collision_rate_coefficient, t0 = 0.):
    def func(m, t):
        λ = 1/m0avg
        #n0 = np.exp(-m0)
        n0 = M/m0avg*np.exp(-m0)
        if t == 0.:
            # To avoid division by 0, simply return the initial conditions for  t = 0 .
            C = n0/m0avg
            f = C*np.exp(-λ*m)
        else:
            τ = collision_rate_coefficient*M*(t - t0)
            g = np.exp(-τ)
            A = n0*g/(m*np.sqrt(1 - g))
            x = 2*λ*m*np.sqrt(1 - g)

            # Instead of the modified Bessel function of the 1st kind  Iₖ(x) , use the exponentially scaled variant
            #
            #     Λₖ(x) = e⁻ˣ Iₖ(x)
            #
            # for increased numerical reach:
            #
            #E = np.exp(-λ*m*(2 - g))
            #B = scipy.special.iv(1, x)
            #E = np.exp(-λ*m*(2 - g) + x)
            if g >= 1.e-4:
                arg = λ*m*(-2 + g + 2*np.sqrt(1 - g))
            else:
                arg = λ*m*(-1/4*g**2 - 1/8*g**3)  # 3rd-order Taylor series expansion, where 0th-order and 1st-order terms cancel
            E = np.exp(arg)
            B = scipy.special.ive(1, x)

            # Even the exponentially scaled variant of the modified Bessel function of the 1st kind  Λₖ(x)  yields NaNs
            # for too large arguments. In this case, resort to the large-argument approximation
            #
            #     Λₖ(x) ≈ 1/√(2π⋅x)    for  x → ∞
            #
            # cf. N. G. Lehtinen, "(Everything a physicist needs to know about) Bessel functions  Jₙ(x)  of integer order
            # (and also Hankel functions  Hₙ⁽¹ ²⁾ )", 2020, §5.2 (http://nlpc.stanford.edu/nleht/Science/reference/bessel.pdf).
            mask = np.isnan(B)
            B[mask] = 1/np.sqrt(2*np.pi*x[mask])

            f = A*E*B
        return f
    return func

def positive_finite(Y, w):
    mask = (Y > 0.) & (w > 0.) & ~np.isinf(Y)
    return Y[mask], w[mask]
