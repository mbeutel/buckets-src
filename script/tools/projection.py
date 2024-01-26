
import sys
import math
import numpy as np
import matplotlib.pyplot as plt


def histogram_bins(val_range, N):
    min_bin = 0.
    max_bin = np.max(val_range)
    delta = (max_bin - min_bin)/N
    return np.linspace(start=0., stop=max_bin + delta/2, num=N + 1)

def log_histogram_bins(val_range, N):
    min_bin = math.floor(math.log10(np.min(val_range)))
    max_bin = math.ceil(math.log10(np.max(val_range)))
    return np.logspace(start=min_bin, stop=max_bin, num=N + 1)

def bin_centers(bins):
    return bins[:-1] + 0.5 * np.diff(bins)

def num_histogram_bins(N):
    # Rice rule
    return math.ceil(2*N**(1/3))

def project_histogram_to_array(array, array_bounds, histogram, histogram_bounds):
    assert len(array_bounds) == len(array) + 1 or len(array_bounds) == 2
    assert len(histogram_bounds) == len(histogram) + 1 or len(histogram_bounds) == 2
    
    a0, a1 = array_bounds[0], array_bounds[-1]
    h0, h1 = histogram_bounds[0], histogram_bounds[-1]
    if h0 == h1:  # this can happen if only the single value 0 is present in the histogram
        return
    Na = len(array)
    Nh = len(histogram)
    Δa = (a1 - a0)/Na
    Δh = (h1 - h0)/Nh
    idx_a0 = min(Na, max(0, math.ceil((h0 - a0)/Δa - 0.5)))
    idx_a1 = min(Na, max(idx_a0, math.floor((h1 - a0)/Δa - 0.5))) + 1
    indices_a = np.array(range(idx_a0, idx_a1))
    indices_h = np.minimum(np.floor((a0 - h0 + Δa*(0.5 + indices_a))/Δh).astype(int), Nh - 1)

    #array[:] = 0.
    array[indices_a] = histogram[indices_h]

def project_log_histogram_to_array(array, array_bounds, histogram, histogram_bounds):
    return project_histogram_to_array(array, np.log10(array_bounds), histogram, np.log10(histogram_bounds))

def make_histogram(data, weights, nbin_max=None, nbin_factor=1., nbins=None):
    N = nbins if nbins is not None else int(num_histogram_bins(len(data))*nbin_factor)
    if nbin_max is not None:
        N = min(N, nbin_max)
    if N == 0:
        return None, None
    bins = histogram_bins(data, N)
    return np.histogram(data, bins=bins, weights=weights, density=False)

def make_log_histogram(data, weights, nbin_max=None, nbin_factor=1., nbins=None, bin_edges=None):
    if bin_edges is not None:
        N = len(bin_edges) - 1
        bin_edges = np.log10(bin_edges)
    else:
        N = nbins if nbins is not None else int(num_histogram_bins(len(data))*nbin_factor)
        if nbin_max is not None:
            N = min(N, nbin_max)
        if N == 0:
            return None, None
        bin_edges = N
    #bins = log_histogram_bins(data, N)
    #hist, edges = np.histogram(data, bins=bins, weights=weights, density=False)
    hist, edges = np.histogram(np.log10(data), bins=bin_edges, weights=weights)
    dldx = (edges[-1] - edges[0])/N
    return hist/(np.log(10)*dldx), 10**edges

def project_histograms(data_seq, Nhist, nbin_factor=1., homogeneous_bins=True):
    Nseq = len(data_seq)
    histograms = [None]*Nseq
    histogram_bounds = [None]*Nseq
    min_val = sys.float_info.max
    max_val = sys.float_info.min
    if homogeneous_bins:
        for i, (data, weights) in enumerate(data_seq):
            edges = np.histogram_bin_edges(np.log10(data), bins=Nhist)
            min_val = min(min_val, edges[0])
            max_val = max(max_val, edges[-1])
        bin_edges = np.logspace(min_val, max_val, base=10, num=Nhist + 1)
    else:
        bins = None
    for i, (data, weights) in enumerate(data_seq):
        hist, bins = make_histogram(data, weights, nbin_factor=nbin_factor, bins=bins)
        histograms[i] = hist
        if hist is not None:
            histogram_bounds[i] = bins[0], bins[-1]
            min_val = min(min_val, bins[0])
            max_val = max(max_val, bins[-1])
    img = np.zeros(shape=[Nhist, Nseq], dtype=float)
    img_vbounds = min_val, max_val
    for i in range(Nseq):
        if histograms[i] is not None:
            project_histogram_to_array(img[:, i], img_vbounds, histograms[i], histogram_bounds[i])
    img_vbins = histogram_bins(img_vbounds, Nhist)
    return img, img_vbins

def project_log_histograms(data_seq, Nhist, vmin=None, vmax=None, nbin_factor=1., homogeneous_bins=True):
    Nseq = len(data_seq)
    min_val = sys.float_info.max
    max_val = sys.float_info.min
    if homogeneous_bins:
        for i, (data, weights) in enumerate(data_seq):
            edges = np.histogram_bin_edges(np.log10(data), bins=Nhist)
            min_val = min(min_val, edges[0])
            max_val = max(max_val, edges[-1])
        if vmin is not None:
            min_val = np.log10(vmin)
        if vmax is not None:
            max_val = np.log10(vmax)
        bin_edges = np.logspace(min_val, max_val, base=10, num=Nhist + 1)
        bins = np.sqrt(bin_edges[1:]*bin_edges[:-1])
        #min_val = bins[0]
        #max_val = bins[-1]
        img = np.zeros(shape=[Nhist, Nseq], dtype=float)
        for i, (data, weights) in enumerate(data_seq):
            hist, _bins = make_log_histogram(data, weights, bin_edges=bin_edges)
            img[:,i] = hist
        #return img, bin_edges
        return img, bin_edges
    else:
        histograms = [None]*Nseq
        histogram_bounds = [None]*Nseq
        for i, (data, weights) in enumerate(data_seq):
            hist, bins = make_log_histogram(data, weights, nbin_factor=nbin_factor)
            histograms[i] = hist
            if hist is not None:
                histogram_bounds[i] = bins[0], bins[-1]
                min_val = min(min_val, bins[0])
                max_val = max(max_val, bins[-1])
        img = np.zeros(shape=[Nhist, Nseq], dtype=float)
        img_vbounds = min_val, max_val
        for i in range(Nseq):
            if histograms[i] is not None:
                project_log_histogram_to_array(img[:, i], img_vbounds, histograms[i], histogram_bounds[i])
        #img_vbins = log_histogram_bins(img_vbounds, Nhist)
        img_vedges = np.logspace(start=np.log10(min_val), stop=np.log10(max_val), num=Nhist + 1)
        return img, img_vedges

def find_nearest_indices(array, values):
    avalues = np.atleast_1d(values)
    li = np.searchsorted(array, avalues, side='left')
    ri_bounded = np.maximum(li - 1, 0)
    li_bounded = np.minimum(li, len(array) - 1)
    take_right = np.abs(array[ri_bounded] - avalues) < np.abs(array[li_bounded] - avalues)
    indices = li_bounded
    indices[take_right] = ri_bounded[take_right]
    return indices if not np.isscalar(values) else indices.item()

def project_nearest(Xs, Ys, Zs, Nh, Nv):
    assert len(Xs)*len(Ys) == len(Zs.ravel())
    # also, `Xs` and `Ys` need to be sorted

    Xmin, Xmax = Xs[0], Xs[-1]
    Ymin, Ymax = Ys[0], Ys[-1]
    NX = len(Xs)
    NY = len(Ys)
    dH_2 = (Xmax - Xmin)/(2*(NX - 1))
    dV_2 = (Ymax - Ymin)/(2*(NY - 1))
    Hs = np.linspace(start=Xmin - dH_2, stop=Xmax + dH_2, num=Nh)
    Vs = np.linspace(start=Ymin - dV_2, stop=Ymax + dV_2, num=Nv)
    His = find_nearest_indices(Xs, Hs).reshape([Nh, 1])
    Vis = find_nearest_indices(Ys, Vs).reshape([1, Nv])
    indices = (His + Vis*NX).reshape([Nh*Nv])
    return Hs, Vs, Zs.ravel()[indices].reshape([Nh, Nv])
