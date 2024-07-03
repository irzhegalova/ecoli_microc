# %%
import multiprocess
import re
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import gridspec

from scipy.stats import bootstrap

from pybedtools import BedTool

import cooler
import cooltools.api.snipping as clsnip
from cooltools.lib.numutils import LazyToeplitz
import cooltools.lib.plotting
import cooltools.api.snipping as clsnip
from cooltools.lib.numutils import LazyToeplitz
import cooltools.lib.plotting
from coolpuppy import coolpup
# from plotpuppy import plotpup

import bioframe
from bioframe import count_overlaps

import bbi

import pyBigWig

time_dic = {"V": [0, "Vegetative"], "S": [5, "Streaming"], "M": [8, "Mound"]}
names_mergedIDR_dic = {
    "V": ["H3K27ac_idr", "H3K4me1_idrPR", "r123.mergeIDR", "H3K4me3_idr"],
    "S": ["H3K27ac_idrPR", "H3K4me1_idr", "r123.mergeIDR", "H3K4me3_idr"],
    "M": ["H3K27ac_idr", "H3K4me1_idrPR", "r1234.mergeIDR", "H3K4me3_idr"],
}

from matplotlib import font_manager
font_dirs = ["/home/izhegalova/fonts/"]  # The path to the custom font file.
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)

# %% Create bins
# load chrom sizes
# dicty_chrom = pd.read_table(
#     "~/projects/dicty/hic_loop_study/data/genome/dicty.chrom.sizes",
#     header=None,
#     index_col=0,
# )
# dicty_chrom.columns = ["length"]
# dicty_chrom_ser = dicty_chrom.squeeze()
# dicty_binned = bioframe.binnify(dicty_chrom_ser, 50)


# functions
def wrapper_stackup(
    i,
    pathes,
    nbins,
    window=10000,
    resolution=2000,
    chrom_file=None,
    bed_list=[
        "results/long_loops/loops_leftFlames0.8.bedpe",
        "results/long_loops/0AB_regular_loops.bedpe",
        "results/long_loops/loops_rightFlames0.2.bedpe",
        "data/loops_quantifyChromosight/0AB_chromosight_quantifyMarkedGood.bedpe",
    ],
    flip_negative_strand=False,
):
    """
    use bed_list and pathes (bw_list) to create stackup

    Parameters
    ----------
    i: int
        i in len(bw_list)
        Select the bed_list[i] based on iteration i
    pathes: list
        Pathes to bw files
    nbins: int
        Number of bins for the whole stackup
    window: int
        Length of before and after windows in bp
    resolution: int
        Resolution of each bin in bp
        Default is 2000 bp
    chrom_file: pandas.DataFrame
        Chromosome sizes file
    bed_list: list
        list of bedpe files to use for stackup
    flip_negative_strand: boolean
        Whether to flip negative strand elements

    Returns
    stackup: pandas.DataFrame
        Stackup to plot

    """
    import pathlib
    import re

    schema = re.sub("\.", "", pathlib.Path(bed_list[i]).suffix)
    df_bed = bioframe.read_table(bed_list[i], schema=schema)
    # if df_bed.shape[1] == 3:
    #     df_bed.columns = ['chrom', 'start', 'end']
    # elif df_bed.shape[1] == 6:
    #     df_bed.columns = ['chrom1', 'start1', 'end1', 'chrom2', 'start2', 'end2']
    if schema == "bedpe":
        df_bed = df_bed[["chrom1", "start1", "end2"]]
        df_bed.columns = ["chrom", "start", "end"]
    # else:
    #     raise AssertionError("Not bed?")
    # drop duplicates
    df_bed = df_bed.drop_duplicates()
    # chrom_file['length'] = chrom_file.end
    tmp = df_bed.set_index("chrom").join(chrom_file, how="outer")

    df_bed = tmp[((tmp.start - window) > 0) & (tmp.end + window < tmp.length)].drop(
        "length", axis=1
    )
    df_bed = df_bed.rename_axis("chrom").reset_index()
    # df_bed.columns = ['chrom', 'start', 'end']
    # chroms = df_bed.chrom
    # starts = df_bed['end'].astype(int) - window
    # ends = df_bed['end'].astype(int) + window
    # before
    chroms = df_bed.chrom
    starts = df_bed["start"].astype(int) - window
    ends = df_bed["start"].astype(int)
    s1 = bbi.stackup(pathes[i], chroms, starts, ends, bins=window // resolution)
    # loop
    chroms = df_bed.chrom
    starts = df_bed["start"].astype(int)
    ends = df_bed["end"].astype(int)
    nbins_s2 = nbins - 2 * (window // resolution)
    s2 = bbi.stackup(pathes[i], chroms, starts, ends, bins=nbins_s2)
    # after
    chroms = df_bed.chrom
    starts = df_bed["end"].astype(int)
    ends = df_bed["end"].astype(int) + window
    s3 = bbi.stackup(pathes[i], chroms, starts, ends, bins=window // resolution)

    # Creation of numpy 2D array with annotation snippets
    # s = bbi.stackup(pathes[i], chroms, starts, ends, bins=nbins)
    s = np.concatenate((s1, s2, s3), axis=1)
    if flip_negative_strand:
        if 'strand' in df_bed.columns:
            neg_index = df_bed.loc[df_bed.strand == "-", :].index.tolist()
            s[neg_index,] = np.flip(s[neg_index,], axis=1)
    # s[np.isnan(s)] = 0

    return s


def wrapper_stackup_createMergedAnchors(
    i,
    pathes,
    nbins,
    window=10000,
    resolution=2000,
    chrom_file=None,
    bed_list=[
        "results/long_loops/loops_leftFlames0.8.bedpe",
        "results/long_loops/0AB_regular_loops.bedpe",
        "results/long_loops/loops_rightFlames0.2.bedpe",
        "data/loops_quantifyChromosight/0AB_chromosight_quantifyMarkedGood.bedpe",
    ],
):
    """
    use bed or bedpe-file to create stackup from anchors

    Parameters
    ----------
    i: int
    pathes: list
    nbins: int
    window: int
    resolution: int
    chrom_file: path
    bed_list: list
    """
    df_bed = pd.read_table(bed_list[i], header=None)
    if df_bed.shape[1] == 3:
        df_bed.columns = ["chrom", "start", "end"]

    elif df_bed.shape[1] == 6:
        df_bed.columns = ["chrom1", "start1", "end1", "chrom2", "start2", "end2"]
        df_bed = df_bed[["chrom1", "start1", "end2"]]
        df_bed.columns = ["chrom", "start", "end"]
    else:
        raise AssertionError("Not bed?")

    # initialize data of lists.
    data = {"chrom": df_bed.chrom, "start": df_bed.start, "end": df_bed.start + 2000}
    # Create DataFrame
    df_bed_left = pd.DataFrame(data)
    data = {"chrom": df_bed.chrom, "start": df_bed.end - 2000, "end": df_bed.end}
    df_bed_right = pd.DataFrame(data)

    # Create DataFrame
    df_bed = pd.concat([df_bed_left, df_bed_right])
    # drop duplicates
    df_bed = df_bed.drop_duplicates()

    chroms = df_bed.chrom
    starts = df_bed.start - window
    ends = df_bed.end + window

    # Creation of numpy 2D array with annotation snippets
    s = bbi.stackup(pathes[i], chroms, starts, ends, bins=nbins)
    s[np.isnan(s)] = 0
    return s


def wrapper_stackup_mergedAnchors(i, pathes, nbins, window=10000):
    """
    use file of merges anchors to stackup them
    """
    df_bed = pd.read_table(bed_list[i], header=None)
    df_bed.columns = ["chrom", "start", "end"]
    chroms = df_bed.chrom
    starts = df_bed.start - window
    ends = df_bed.start + 2000 + window

    # Creation of numpy 2D array with annotation snippets
    s = bbi.stackup(pathes[i], chroms, starts, ends, bins=nbins)
    s[np.isnan(s)] = 0
    return s


def wrapper_stackup_difBw(i, pathes, nbins, window=10000):
    """
    ??? can see no differences with previous one
    """
    df_bed = pd.read_table(bed_list[i], header=None)
    df_bed.columns = ["chrom", "start", "end"]
    df_bed = df_bed.drop_duplicates()
    chroms = df_bed.chrom
    starts = df_bed.start - window
    ends = df_bed.start + 2000 + window

    # Creation of numpy 2D array with annotation snippets
    s = bbi.stackup(pathes[i], chroms, starts, ends, bins=nbins)
    s[np.isnan(s)] = 0
    return s


def round_half_up(n, decimals=0):
    """
    round numbers
    """
    multiplier = 10**decimals
    return math.floor(n * multiplier + 0.5) / multiplier


def _create_stack(
    paired_sites,
    clr,
    df_chromsizes,
    windows,
    flank=10000,
    nthreads=32,
    cell_cycle=False,
    use_oe=True,
):
    """
    Call cooltools.snipping and then stack it

    Returns
    -------
    3D array with third dimension being different snips
    """
    if use_oe:
        # Calculate expected interactions for chromosomes

        expected = cooltools.expected_cis(
            clr, view_df=df_chromsizes, nproc=2, chunksize=1_000_000
        )

        # O/E
        oe_snipper = ObsExpSnipperTriu(clr, expected)

        # create the stack of snips:
        with multiprocess.Pool(nthreads) as pool:
            stack = clsnip.pileup(
                windows, oe_snipper.select, oe_snipper.snip, map=pool.map
            )
    else:
        # Create the snipper object:
        snipper = clsnip.CoolerSnipper(clr)

        stack = clsnip.pileup(windows, snipper.select, snipper.snip)
    return stack


def create_stack(
    paired_sites,
    clr,
    df_chromsizes,
    flank=10000,
    nthreads=32,
    remove_na=True,
    cell_cycle=False,
    use_max=False,
    use_oe=True,
    use_localExpected=False,
    use_mean=False,
):
    """
    Create the right format of windows, then call inner horse to create stack
    """

    supports = df_chromsizes[["chrom", "start", "end"]].values
    resolution = clr.resolution
    windows1 = clsnip.make_bin_aligned_windows(
        resolution, paired_sites["chrom1"], paired_sites["start1"], flank_bp=flank
    )

    windows2 = clsnip.make_bin_aligned_windows(
        resolution, paired_sites["chrom2"], paired_sites["start2"], flank_bp=flank
    )

    windows = pd.merge(
        windows1, windows2, left_index=True, right_index=True, suffixes=("1", "2")
    )
    windows["region"] = windows["chrom1"]
    # create snip and then stack it
    stack = _create_stack(
        paired_sites,
        clr,
        df_chromsizes,
        windows=windows,
        flank=flank,
        nthreads=nthreads,
        cell_cycle=cell_cycle,
        use_oe=use_oe,
    )
    return stack


def plot_avLoop(bed_file, clr, resolution, df_chromsizes, flank, vmax=0.9, vmin=-0.35):
    """
    Plot average loop using cooltools
    """
    # av loop
    df_bed = pd.read_table(bed_file, header=None)
    if df_bed.shape[1] == 3:
        df_bed.columns = ["chrom1", "start1", "end2"]
        df_bed["chrom2"] = df_bed["chrom1"]
        df_bed["end2"] = df_bed["end2"] + 1
        df_bed["end1"] = df_bed["start1"] + 5000
        df_bed["start2"] = df_bed["end2"] - 5000
        df_bed = df_bed[["chrom1", "start1", "end1", "chrom2", "start2", "end2"]]
    elif df_bed.shape[1] == 6:
        df_bed.columns = ["chrom1", "start1", "end1", "chrom2", "start2", "end2"]
    else:
        raise AssertionError("Not bed?")

    paired_sites = df_bed[
        df_bed["end2"] - df_bed["start1"] > resolution * 2
    ]  # [cluster_labels == k_ist]

    # av_loop_ax = plt.subplot(gs[6])
    # oe_stack = create_stack(paired_sites, clr, df_chromsizes, flank=flank, remove_na = False)
    expected = cooltools.expected_cis(
        clr, view_df=df_chromsizes, nproc=2, chunksize=1_000_000
    )
    oe_stack = cooltools.pileup(
        clr, paired_sites, view_df=df_chromsizes, expected_df=expected, flank=flank
    )

    oe_mtx = np.nanmedian(oe_stack, axis=2)
    im = plt.imshow(
        np.log2(oe_mtx), vmax=vmax, vmin=vmin, cmap="coolwarm", interpolation="none"
    )
    # add middle pixel as number
    which_middle = oe_mtx.shape[0] // 2
    enr = np.nanmedian(
        np.nanmedian(
            np.log2(oe_mtx)[
                which_middle - 1 : which_middle + 2, which_middle - 1 : which_middle + 1
            ],
            axis=0,
        ),
        axis=0,
    )

    plt.text(
        s=np.round(enr, 3),
        y=0.95,
        x=0.05,
        ha="left",
        va="top",
        size="small",
        # transform=plt.transAxes,
    )
    ticks_pixels = np.linspace(0, flank * 2 // resolution, 5)
    ticks_kbp = ((ticks_pixels - ticks_pixels[-1] / 2) * resolution // 1000).astype(int)
    plt.xticks(ticks_pixels, ticks_kbp)
    plt.yticks(ticks_pixels, ticks_kbp)
    plt.xlabel("relative position, kbp")
    plt.ylabel("relative position, kbp")


def my_zs(a, axis=1):
    """
    compute z-scores
    """
    b = np.array(a).swapaxes(axis, 1)
    mu = np.nanmean(b, axis=1)[..., np.newaxis]
    sigma = np.nanstd(b, axis=1)[..., np.newaxis]
    return (b - mu) / sigma


def interpolate_nans(X):
    """Overwrite NaNs with column value interpolations."""
    for j in range(X.shape[1]):
        mask_j = np.isnan(X[:, j])
        X[mask_j, j] = np.interp(
            np.flatnonzero(mask_j), np.flatnonzero(~mask_j), X[~mask_j, j]
        )
    return X


def sliding_window(seq=None, half_window_size=1):
    # seq=rnaseq_binned_vec1['value'].tolist()
    rnaseq_binned_vec_slidWin = []
    for i in range(len(seq)):
        if i < half_window_size:
            rnaseq_binned_vec_slidWin.append(None)
            continue
        if i == len(seq) - half_window_size:
            rnaseq_binned_vec_slidWin.append(None)
            continue
        rnaseq_binned_vec_slidWin.append(
            np.mean(seq[i - half_window_size : i + half_window_size + 1])
        )
    return rnaseq_binned_vec_slidWin


# window size (one-sided)
# number of bins in the window
def plot_around_loop(
    path_bw,
    plot_name,
    nbins=30,
    resolution=2000,
    chrom_file=None,
    window=10000,
    mode="median",
    ymin=None,
    ymax=None,
    vmin=-3,
    vmax=3,
    norm=False,
    fill=True,
    how_far_from_edge=10,
    bed_list=[
        "results/long_loops/0AB_loops_rightFlames0.2.bedpe",  # "results/long_loops/loops_leftFlames0.8.bedpe",
        "results/long_loops/0AB_regular_loops.bedpe",
        "results/long_loops/0AB_loops_leftFlames0.8.bedpe",  # "results/long_loops/loops_rightFlames0.2.bedpe",
        "data/loops_quantifyChromosight/0AB_chromosight_quantifyMarkedGood.bedpe",
    ],
    name_list=None,
    flip_negative_strand=False,
    return_matrix=False,
    pic_path="results/pics/paper/",
    multipage=False
):
    """
    Plot bw file around bedpe files

    Parameters
    ----------
    path_bw : str
        Name of the bigWig file.
    plot_name : str
        Name of the plot.
    nbins : int
        Number of bins in the stack.
    resolution : int
        Resolution of bins.
    chrom_file : dict
        Chromosome sizes file
    window : int
        Size of the window before and after the loop.
    mode : str
        Whether to use the median or the mean.
    ymin : float
        Maximum y-value on lineplot
    ymax : float
        Minimum y-value on lineplot
    vmin : float
        Minimum y-value on stackup
    vmax : float
        Maximum y-value on stackup
    norm : bool
        Whether to normalize the y-values.
        Normalization is done by dividing the y-values by the y-value of relevant coverage file.
        May be set to True only for coverage files using DNA strands.
    fill : bool
        Whether to fill the stackup with NaNs.
    how_far_from_edge : int
        Define how far from the edge loop anchor must be in order to be included in the plot.
        Default is 10.
    bed_list : list
        List of bedpe files.
    flip_negative_strand: bool
        Whether to flip negative strand
    return_matrix: bool
        Whether return matrix as skip plotting 
        (useful to build figures with subplots)

    Returns
    -------
    pdf file at results/pics/paper directory

    """

    Timing = [0]  # 2, 5, 8
    Timing_cc = [0, 1, 3, 6]
    # Timing = [45]
    pathes_global = [path_bw]

    pathes = []
    for p in range(len(pathes_global)):
        for time in Timing:
            for k in range(len(bed_list)):
                match = re.search("%s", pathes_global[p])
                if match:
                    pathes.append(pathes_global[p] % (time))
                else:
                    pathes.append(pathes_global[p])
    # create outer grid
    outer_grid = gridspec.GridSpec(
        1, len(bed_list), width_ratios=[5,5,5,5,6]
    )  # gridspec with two adjacent horizontal cells
    fig = plt.figure(figsize=[1.5 * len(bed_list), 4.5])
    fig.suptitle(plot_name, y=0.99)
    s_list = []
    # min_matr = []
    # max_matr = []
    # create matr and vector containing max & min for y-axis
    for i in range(len(bed_list)):
        s_list.append(
            wrapper_stackup(
                i,
                pathes=pathes,
                nbins=nbins,
                resolution=resolution,
                window=window,
                bed_list=bed_list,
                chrom_file=chrom_file,
                flip_negative_strand=flip_negative_strand,
            )
        )
        s_list[i][s_list[i] == -np.inf] = np.nan

    if norm:
        # create 'expected' (a line to divide by)
        s_list_ref = []
        upper_median_ref = []
        pathes_global_ref = ["bw/cov_by_minusGenes.bw", "bw/cov_by_plusGenes.bw"]
        for p in range(len(pathes_global_ref)):
            pathes_ref = []
            for i in range(len(bed_list)):
                pathes_ref.append(pathes_global_ref[p])
                stackup_tmp = wrapper_stackup(i, pathes=pathes_ref, nbins=nbins, window=window, chrom_file=chrom_file)
                s_list_ref.append(stackup_tmp)
                upper_median_ref.append(np.nanmean(stackup_tmp, axis=0))

    for i in range(len(bed_list)):
        if norm:
            match_min = re.search("minus", path_bw)
            if match_min:
                s_list[i] = s_list[i] / upper_median_ref[i][None, :]
            match_pl = re.search("plus", path_bw)
            if match_pl:
                s_list[i] = s_list[i] / upper_median_ref[i + len(bed_list)][None, :]

        # Ordering
        if mode == "mean":
            for_order = np.nanmean(
                s_list[i][
                    :,
                    [(int(nbins / 2 - nbins * 0.15)), (int(nbins / 2 + nbins * 0.15))],
                ],
                axis=1,
            )  # ordering by central values of stack
        else:
            for_order = np.nanmedian(
                s_list[i][
                    :, (int(nbins / 2 - nbins * 0.15)) : (int(nbins / 2 + nbins * 0.15))
                ],
                axis=1,
            )  # ordering by central values of stack
        order = np.argsort(for_order)[::-1]
        if return_matrix:
            return s_list, order
        # Plotting and tuning the plot
        # fig = plt.figure(figsize=[2,15])
        the_most_left_cell = outer_grid[0, i]  # the left SubplotSpec within outer_grid
        # gs = gridspec.GridSpecFromSubplotSpec(
        #     2,
        #     3,
        #     the_most_left_cell,
        #     width_ratios=[5, 1, 1],
        #     height_ratios=[1, 2],
        #     wspace=0.05,
        #     hspace=0.05,
        # )
        if i in [0,1,2]:
            pileup_ticks = ['border', 'border']
            # pileup_ticks = ['' for i in range(20)] + ['border'] + ['' for i in range(40)] + ['border']# + ['' for j in range(20)]
        else:
            pileup_ticks = ['TSS', 'TES']
            # pileup_ticks = ['' for i in range(20)] + ['TSS'] + ['' for i in range(40)] + ['TES']
        if i == len(bed_list)-1:
            gs = gridspec.GridSpecFromSubplotSpec(
                2,
                3,
                the_most_left_cell,
                width_ratios=[5, 0.5, 0.5],
                height_ratios=[1, 2],
                wspace=0.05,
                hspace=0.05,
            )
            ax_col_marg = plt.subplot(gs[0])  # upper profile
            # ax_row_marg = plt.subplot(gs[5]) # right marginal distribution of rows or custom numbers
            # if i == (len(bed_list) - 1):
            ax_bar = plt.subplot(gs[5])  # colorbar
            ax_heatmap = plt.subplot(gs[3])  # heatmap

            sns.heatmap(
                s_list[i][order, :],
                ax=ax_heatmap,
                cbar=True,
                cbar_ax=ax_bar,
                yticklabels=False,
                xticklabels=pileup_ticks,
                cmap="coolwarm",
                vmin=vmin,
                vmax=vmax,
            )
            ax_heatmap.xaxis.tick_bottom()
            ax_heatmap.set_xticks([20,60])
            ax_heatmap.set_xticklabels(pileup_ticks, rotation=360)
            for spine in ax_heatmap.spines.items():
                spine[1].set(visible=True, lw=.5, edgecolor="black")
            ax_bar.spines["outline"].set(visible=True, lw=.5, edgecolor="black")
        else:
            gs = gridspec.GridSpecFromSubplotSpec(
                2,
                1,
                the_most_left_cell,
                width_ratios=[5],
                height_ratios=[1, 2],
                wspace=0.0,
                hspace=0.05,
            )
            ax_col_marg = plt.subplot(gs[0])  # upper profile
            ax_heatmap = plt.subplot(gs[1])  # heatmap

            sns.heatmap(
                s_list[i][order, :],
                ax=ax_heatmap,
                cbar=False,
                #cbar_ax=ax_bar,
                yticklabels=False,
                xticklabels=pileup_ticks,
                cmap="coolwarm",
                vmin=vmin,
                vmax=vmax,
            )
            ax_heatmap.xaxis.tick_bottom()
            ax_heatmap.set_xticks([20,60])
            ax_heatmap.set_xticklabels(pileup_ticks, rotation=360)
            for spine in ax_heatmap.spines.items(): 
                spine[1].set(visible=True, lw=.5, edgecolor="black")
            
        # ax_heatmap.set_yticks([how_far_from_edge, nbins - how_far_from_edge -1])
        ax_col_marg.set_title(name_list[i])
        if mode == "mean":
            upper_median = np.nanmean(s_list[i][order, :], axis=0)
            # tmp = sliding_window(seq=np.mean(s_list[i][order, :], axis=0), half_window_size=2)
            ax_col_marg.plot(upper_median)
            ax_col_marg.set_ylim([ymin, ymax])
            # err_style
            dyfit = np.nanstd(s_list[i][order, :], axis=0) / np.sqrt(
                np.size(s_list[i][order, :], axis=0)
            )
            if fill:
                ax_col_marg.fill_between(
                    range(nbins),
                    upper_median - dyfit,
                    upper_median + dyfit,
                    color="blue",
                    alpha=0.3,
                )
            # ax_col_marg.vlines(x=[nbins * 0.25, nbins * 0.75], ls='--', ymin=min(upper_median - dyfit),
            #                    ymax=max(upper_median + dyfit))  # [6.5, 22.5]

            # ax_col_marg.set_ylim([min(upper_median - dyfit), max(upper_median + dyfit)])
        else:
            upper_median = np.nanmedian(s_list[i][order, :], axis=0)
            # tmp = sliding_window(seq=np.mean(s_list[i][order, :], axis=0), half_window_size=2)
            ax_col_marg.plot(upper_median)
            if i+1 == len(bed_list):
                ax_col_marg.set(xticklabels=[])
                ax_col_marg.yaxis.tick_right()
                # ax_col_marg.tick_params(axis='y', which='both', labelleft='off', labelright='on', labelfontfamily='Arial', left=False, right=True)
            else:
                ax_col_marg.set(xticklabels=[], yticklabels=[])
                ax_col_marg.tick_params(axis='y', which='major', right=True)
                
            # ax_col_marg.set_ylim([min(min_vec), max(max_vec)])
            # err_style
            to_bootstrap = (s_list[i][order, :],)
            rng = np.random.default_rng()
            res = bootstrap(
                to_bootstrap,
                np.nanstd,
                n_resamples=1000,
                confidence_level=0.9,  # vectorized=True,
                axis=0,
                random_state=rng,
                method="basic",
            )
            dyfit = res.standard_error
            # dyfit = np.std(s_list[i][order, :], axis=0) / np.sqrt(np.size(s_list[i][order, :], axis=0))
            if fill:
                ax_col_marg.fill_between(
                    range(nbins),
                    upper_median - dyfit,
                    upper_median + dyfit,
                    color="blue",
                    alpha=0.3,
                )
            # ax_col_marg.vlines(x=[nbins * 0.25, nbins * 0.75], ls='--', ymin=min(upper_median - dyfit),
            #                    ymax=max(upper_median + dyfit))  # [6.5, 22.5]
            if ymin is None:
                ymin = np.quantile(np.nanmin(s_list[i][order, :], axis=0), 0.05)
                # if ymin < 0:
                #     ymin *=1.1
                # else:
                #     ymin *= 0.9
            if ymax is None:
                ymax = np.quantile(np.nanmax(s_list[i][order, :], axis=0), 0.95)
                # if ymax < 0:
                #     ymax *= 0.9
                # else:
                #     ymax *= 1.1
            ax_col_marg.set_ylim([ymin, ymax])
        ax_col_marg.vlines(
            x=[how_far_from_edge, nbins - how_far_from_edge -1],
            ls="--",
            ymin=ymin,
            ymax=ymax,
        )  # [6.5, 22.5]
        # ax_col_marg.set_ylim([min(upper_median - dyfit), max(upper_median + dyfit)])
        # ax_row_marg.plot(for_order[order[::-1]], np.arange(s.shape[0]))
        # title = os.path.split(pathes_global[p])[1]
        # fig.suptitle("{}".format(title), x=0.5, y=0.92)
    if not multipage:
        fig.savefig(pic_path + plot_name + '.pdf', dpi=100, bbox_inches="tight")
        plt.show()
    else:
        return fig


def plot_around_anchors(
    path_bw,
    plot_name,
    nbins=30,
    resolution=2000,
    chrom_file=None,
    window=10000,
    mode="median",
    ymin=-0.4,
    ymax=0.4,
    vmin=-3,
    vmax=3,
    norm=False,
    fill=True,
    how_far_from_edge=10,
    bed_list=[
        "results/long_loops/loops_leftFlames0.8.bedpe",
        "results/long_loops/0AB_regular_loops.bed",
        "results/long_loops/loops_rightFlames0.2.bedpe",
        "data/loops_quantifyChromosight/0AB_chromosight_quantifyMarkedGood.bedpe",
    ],
):
    """
    Plot bw file around bedpe files

    Parameters
    ----------
    path_bw : str
        Name of the bigWig file.
    plot_name : str
        Name of the plot.
    nbins : int
        Number of bins in the stack.
    resolution : int
        Resolution of bins.
    chrom_file : dict
        Chromosome sizes file
    window : int
        Size of the window before and after the loop.
    mode : str
        Whether to use the median or the mean.
    ymin : float
        Maximum y-value on lineplot
    ymax : float
        Minimum y-value on lineplot
    vmin : float
        Minimum y-value on stackup
    vmax : float
        Maximum y-value on stackup
    norm : bool
        Whether to normalize the y-values.
        Normalization is done by dividing the y-values by the y-value of relevant coverage file.
        May be set to True only for coverage files using DNA strands.
    fill : bool
        Whether to fill the stackup with NaNs.
    how_far_from_edge : int
        Define how far from the edge loop anchor must be in order to be included in the plot.
        Default is 10.
    bed_list : list
        List of bed or bedpe files.

    Returns
    -------
    pdf file at results/pics/paper directory

    """

    Timing = [0]  # 2, 5, 8
    Timing_cc = [0, 1, 3, 6]
    pathes_global = [path_bw]

    pathes = []
    for p in range(len(pathes_global)):
        for time in Timing:
            for k in range(len(bed_list)):
                match = re.search("%s", pathes_global[p])
                if match:
                    pathes.append(pathes_global[p] % (time))
                else:
                    pathes.append(pathes_global[p])
    # create outer grid
    outer_grid = gridspec.GridSpec(
        1, len(bed_list)
    )  # gridspec with two adjacent horizontal cells
    fig = plt.figure(figsize=[3 * len(bed_list), 9])

    s_list = []
    min_matr = []
    max_matr = []
    # create matr and vector containing max & min for y-axis
    for i in range(len(bed_list)):
        s_list.append(
            wrapper_stackup_createMergedAnchors(
                i,
                pathes=pathes,
                nbins=nbins,
                resolution=resolution,
                window=window,
                bed_list=bed_list,
                chrom_file=chrom_file,
            )
        )
        s_list[i][s_list[i] == -np.inf] = np.nan
        # s_list[i] = my_zs(s_list[i])
        # Ordering
        if mode == "mean":
            for_order = np.nanmean(
                s_list[i][
                    :, (int(nbins / 2)) : (int(nbins / 2)) + 1
                ],  # (int(nbins / 2 + nbins * 0.15))
                axis=1,
            )
        else:
            for_order = np.nanmedian(
                s_list[i][:, (int(nbins / 2)) : (int(nbins / 2)) + 1], axis=1
            )  # ordering by central values of stack
        order = np.argsort(for_order)[::-1]

        min_matr.append(s_list[i].min())
        max_matr.append(s_list[i].max())

    if norm:
        # create 'expected' (a line to divide by)
        s_list_ref = []
        upper_median_ref = []
        pathes_global_ref = ["bw/cov_by_minusGenes.bw", "bw/cov_by_plusGenes.bw"]
        for p in range(len(pathes_global_ref)):
            pathes_ref = []
            for i in range(len(bed_list)):
                pathes_ref.append(pathes_global_ref[p])
                stackup_tmp = wrapper_stackup_createMergedAnchors(
                    i,
                    pathes=pathes_ref,
                    nbins=nbins,
                    window=window,
                    chrom_file=chrom_file,
                )
                s_list_ref.append(stackup_tmp)
                upper_median_ref.append(np.nanmean(stackup_tmp, axis=0))

    for i in range(len(bed_list)):
        if norm:
            match_min = re.search("minus", path_bw)
            if match_min:
                s_list[i] = s_list[i] / upper_median_ref[i][None, :]
            match_pl = re.search("plus", path_bw)
            if match_pl:
                s_list[i] = s_list[i] / upper_median_ref[i + len(bed_list)][None, :]

        # Ordering
        if mode == "mean":
            for_order = np.nanmean(
                s_list[i][:, (int(nbins / 2)) : (int(nbins / 2)) + 1], axis=1
            )  # ordering by central values of stack
        else:
            for_order = np.nanmedian(
                s_list[i][:, (int(nbins / 2)) : (int(nbins / 2)) + 1], axis=1
            )  # ordering by central values of stack
        order = np.argsort(for_order)[::-1]

        # Plotting and tuning the plot
        # fig = plt.figure(figsize=[2,15])
        the_most_left_cell = outer_grid[0, i]  # the left SubplotSpec within outer_grid
        gs = gridspec.GridSpecFromSubplotSpec(
            2,
            3,
            the_most_left_cell,
            width_ratios=[5, 1, 1],
            height_ratios=[1, 2],
            wspace=0.0,
            hspace=0.05,
        )

        ax_col_marg = plt.subplot(gs[0])  # upper profile
        # ax_row_marg = plt.subplot(gs[5]) # right marginal distribution of rows or custom numbers
        # if i == (len(bed_list) - 1):
        ax_bar = plt.subplot(gs[5])  # colorbar
        ax_heatmap = plt.subplot(gs[3])  # heatmap

        sns.heatmap(
            s_list[i][order, :],
            ax=ax_heatmap,
            cbar_ax=ax_bar,
            yticklabels=False,
            xticklabels=False,
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
            # vmin=min(min_matr), vmax=max(max_matr)
        )
        # ax_heatmap.set_xticks(np.arange(0, nbins + 1, 5))
        # ax_heatmap.set_xticklabels(np.arange(-nbins/2, 1+nbins/2, 5).astype(int), rotation=90)
        # ax_col_marg.set_xticks(np.arange(0, nbins + 1, 5))
        # ax_col_marg.set_xticklabels(np.arange(-nbins/2, 1+nbins/2, 5).astype(int), rotation=90)

        # ax_row_marg.set_yticks([])
        # ax_row_marg.tick_params(labelrotation=90)
        if mode == "mean":
            upper_median = np.nanmean(s_list[i][order, :], axis=0)
            # tmp = sliding_window(seq=np.mean(s_list[i][order, :], axis=0), half_window_size=2)
            ax_col_marg.plot(upper_median)
            ax_col_marg.set_ylim([ymin, ymax])
            # err_style
            dyfit = np.nanstd(s_list[i][order, :], axis=0) / np.sqrt(
                np.size(s_list[i][order, :], axis=0)
            )
            if fill:
                ax_col_marg.fill_between(
                    range(nbins),
                    upper_median - dyfit,
                    upper_median + dyfit,
                    color="blue",
                    alpha=0.3,
                )
            # ax_col_marg.vlines(x=[nbins * 0.25, nbins * 0.75], ls='--', ymin=min(upper_median - dyfit),
            #                    ymax=max(upper_median + dyfit))  # [6.5, 22.5]

            # ax_col_marg.set_ylim([min(upper_median - dyfit), max(upper_median + dyfit)])
        else:
            upper_median = np.nanmedian(s_list[i][order, :], axis=0)
            # tmp = sliding_window(seq=np.mean(s_list[i][order, :], axis=0), half_window_size=2)
            ax_col_marg.plot(upper_median)
            ax_col_marg.set(xticklabels=[])
            # ax_col_marg.set_ylim([min(min_vec), max(max_vec)])
            # err_style
            to_bootstrap = (s_list[i][order, :],)
            rng = np.random.default_rng()
            res = bootstrap(
                to_bootstrap,
                np.nanstd,
                n_resamples=1000,
                confidence_level=0.9,  # vectorized=True,
                axis=0,
                random_state=rng,
                method="basic",
            )
            dyfit = res.standard_error
            # dyfit = np.std(s_list[i][order, :], axis=0) / np.sqrt(np.size(s_list[i][order, :], axis=0))
            if fill:
                ax_col_marg.fill_between(
                    range(nbins),
                    upper_median - dyfit,
                    upper_median + dyfit,
                    color="blue",
                    alpha=0.3,
                )
            # ax_col_marg.vlines(x=[nbins * 0.25, nbins * 0.75], ls='--', ymin=min(upper_median - dyfit),
            #                    ymax=max(upper_median + dyfit))  # [6.5, 22.5]
            ax_col_marg.set_ylim([ymin, ymax])
        ax_col_marg.vlines(
            x=[how_far_from_edge - 1, nbins - how_far_from_edge],
            ls="--",
            ymin=ymin,
            ymax=ymax,
        )  # [6.5, 22.5]

        # ax_col_marg.set_ylim([min(upper_median - dyfit), max(upper_median + dyfit)])
        # ax_row_marg.plot(for_order[order[::-1]], np.arange(s.shape[0]))
        # title = os.path.split(pathes_global[p])[1]
        # fig.suptitle("{}".format(title), x=0.5, y=0.92)

    fig.savefig("results/pics/paper/" + plot_name, dpi=100, bbox_inches="tight")
    plt.show()


def hic_zoo(
    resolution=2000,
    flank=80_000,
    chromsizes="data/genome/dicty.chrom.sizes",
    bedpe_file="data/loops_quantifyChromosight/0AB_chromosight_quantifyMarkedGood.bedpe",
    cool_file="../dicty_hic_distiller/subsampled/0AB.filtered.mcool",
    vmax=0.9,
    vmin=-0.35,
):
    """
    :param resolution: (int)
    :param flank: (int) how much to flank
    :param bedpe_file: path to BEDPE file
    :param cool_file: path to mcool file
    :param df_chromsizes: path to chromsizes (two columns)
    :return: IS table and plot
    """
    clr = cooler.Cooler(cool_file + "::/resolutions/" + str(resolution))
    df_chromsizes = pd.read_csv(
        chromsizes,
        sep="\t",
        header=None,
    )
    df_chromsizes.columns = ["chrom", "end"]
    df_chromsizes.loc[:, "start"] = 0
    df_chromsizes["name"] = (
        df_chromsizes["chrom"]
        + ":"
        + df_chromsizes["start"].astype(str)
        + "-"
        + df_chromsizes["end"].astype(str)
    )
    df_chromsizes = df_chromsizes.set_index("chrom").loc[clr.chromnames].reset_index()
    df_chromsizes["end"] = clr.chromsizes.tolist()
    df_chromsizes = df_chromsizes[["chrom", "start", "end", "name"]]
    plot_avLoop(
        bed_file=bedpe_file,
        clr=clr,
        resolution=resolution,
        df_chromsizes=df_chromsizes,
        flank=flank,
        vmax=vmax,
        vmin=vmin,
    )
    plt.colorbar(label="log2 median obs/exp")
    plt.grid(None)
    # plt.savefig("results/pics/paper/0AB_avLoop.pdf", dpi=100,
    #             bbox_inches='tight')

    plt.show()

    # windows = [3 * resolution, 5 * resolution, 10 * resolution, 25 * resolution]
    # insulation_table = insulation(clr, windows, verbose=True)
    # return insulation_table


def hic_zoo_coolpuppy(
    cooler_path="../dicty_hic_distiller/subsampled/0AB.filtered.mcool::/resolutions/",
    resolution=2000,
    nthreads=4,
    mode="bedpe",
    chromsizes_path="data/genome/dicty.chrom.sizes",
    bedpe_path="data/loops_quantifyChromosight/0AB_chromosight_quantifyMarkedGood.bedpe",
    rescale=False,
    local=False,
    score=True,
    coef_flank=3,
    ignore_diags=2,
    rescale_flank=1.0,
    vmax=2.5,
    vmin=0.5,
    organism="nonhuman",
    plot_name="",
):
    """
    Plot average chromatin feature using coolpuppy
    """

    clr = cooler.Cooler(cooler_path + str(resolution))

    # chromsizes
    tmp = {"chrom": clr.chromsizes.index.to_list(), "end": [3, 4]}

    df_chromsizes = clr.chromsizes.reset_index()  # pd.DataFrame(data=tmp)
    # df_chromsizes = pd.read_csv(chromsizes_path, sep='\t', header=None, )
    df_chromsizes.columns = ["chrom", "end"]
    df_chromsizes.loc[:, "start"] = 0
    df_chromsizes["name"] = (
        df_chromsizes["chrom"]
        + ":"
        + df_chromsizes["start"].astype(str)
        + "-"
        + df_chromsizes["end"].astype(str)
    )
    df_chromsizes = df_chromsizes.set_index("chrom").loc[clr.chromnames].reset_index()
    df_chromsizes["end"] = clr.chromsizes.tolist()
    df_chromsizes = df_chromsizes[["chrom", "start", "end", "name"]]

    df_bedpe = pd.read_table(bedpe_path, header=None)  # .iloc[0:20,:]
    if mode == "bed":
        if df_bedpe.shape[1] == 3:
            print("bed is loaded")
        elif df_bedpe.shape[1] == 6:
            df_bedpe.columns = ["chrom1", "start1", "end1", "chrom2", "start2", "end2"]
            df_bedpe = df_bedpe[["chrom1", "start1", "end2"]]
            print("bed is created")
        df_bedpe.columns = ["chrom", "start", "end"]
        paired_sites = df_bedpe[df_bedpe["end"] - df_bedpe["start"] > resolution * 4]
        median_size = np.median(paired_sites.end - paired_sites.start)
    elif mode == "bedpe":
        if df_bedpe.shape[1] == 3:
            df_bedpe.columns = ["chrom1", "start1", "end2"]
            df_bedpe["end1"] = df_bedpe["start1"] + 2000
            df_bedpe["start2"] = df_bedpe["end2"] - 2000
            df_bedpe = df_bedpe[
                ["chrom1", "start1", "end1", "chrom2", "start2", "end2"]
            ]
        elif df_bedpe.shape[1] == 6:
            df_bedpe.columns = ["chrom1", "start1", "end1", "chrom2", "start2", "end2"]
        paired_sites = df_bedpe[df_bedpe["start2"] - df_bedpe["end1"] > resolution * 2]
        median_size = np.median(paired_sites.start2 - paired_sites.end1)
    else:
        raise AssertionError("Not bedpe?")
    if organism == "human":
        expected = pd.read_csv(
            "GSE63525_GM12878_insitu_primary.5kb.expected.tsv", sep="\t"
        )
    else:
        expected = cooltools.expected_cis(
            clr,
            view_df=df_chromsizes,
            nproc=nthreads,
            chunksize=1_000_000,
            ignore_diags=0,
        )

    flank = int((median_size * coef_flank // resolution) * resolution)
    if rescale:
        cc = coolpup.CoordCreator(
            features=paired_sites,
            # flank=flank,
            mindist=resolution * 3,
            local=local,
            resolution=resolution,
            features_format=mode,
            rescale_flank=rescale_flank,
        )
        pu = coolpup.PileUpper(
            clr,
            cc,
            view_df=df_chromsizes,
            expected=expected,
            control=False,
            rescale=True,
            rescale_size=int(1 + flank * 2 // resolution),
            nproc=nthreads,
            ignore_diags=ignore_diags,
        )
    else:
        cc = coolpup.CoordCreator(
            features=paired_sites,
            local=local,
            flank=flank,
            mindist=resolution * 3,
            resolution=resolution,
            features_format=mode,
        )
        # , rescale_flank=1)
        pu = coolpup.PileUpper(
            clr,
            cc,
            view_df=df_chromsizes,
            expected=expected,
            control=False,
            # rescale=True, rescale_size=81,
            nproc=nthreads,
            ignore_diags=ignore_diags,
        )
    pup = pu.pileupsWithControl(nproc=nthreads)

    plotpup.make_heatmap_grid(
        pup,  # cols='separation',
        score=score,
        cmap="coolwarm",
        scale="log",
        sym=False,
        height=3,
        vmax=vmax,
        vmin=vmin,
    )
    plt.savefig(plot_name, dpi=100, bbox_inches="tight")
    plt.show()
    # return pup


def annotate(data, **kws):
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        data.dropna()["delta_is"],  #
        data.dropna()["delta_rna"],
    )
    ax = plt.gca()
    ax.text(
        0.05,
        0.8,
        "y={0:.1f}x+{1:.1f}, R-val={2:.2f}".format(slope, intercept, r_value),
        transform=ax.transAxes,
    )


def load_BedInMode(file_loops, time=None, mode="bedpe", resolution=2000):
    """
    Load a bed file of loops and convert it according to the mode
    """
    if time is not None:
        df_loops = bioframe.read_table(file_loops % (time), schema="bed3")
    else:
        df_loops = bioframe.read_table(file_loops, schema="bed3")

    if mode == "end":
        df_loops["start"] = df_loops["end"] - resolution
        df_loops["loop_id"] = df_loops.index
    elif mode == "start":
        df_loops["end"] = df_loops["start"] + resolution
        df_loops["loop_id"] = df_loops.index
    elif mode == "start_3bins":
        data = {
            "chrom": df_loops.chrom,
            "start": df_loops.start - resolution,
            "end": df_loops.start + resolution * 2,
        }
        df_loops = pd.DataFrame(data)
    elif mode == "end_3bins":
        data = {
            "chrom": df_loops.chrom,
            "start": df_loops.end - resolution * 2,
            "end": df_loops.end + resolution,
        }
        df_loops = pd.DataFrame(data)
    elif mode == "outside":
        resolution = 500
        cooler_path="../dicty_hic_distiller/subsampled/0AB.filtered.mcool::/resolutions/"
        clr = cooler.Cooler(cooler_path + str(resolution))
        df_chromsizes = clr.chromsizes.reset_index()
        df_chromsizes = clr.chromsizes.reset_index()
        df_chromsizes.columns = ['chrom', 'end']
        df_chromsizes.loc[:, 'start'] = 0
        df_chromsizes["name"] = df_chromsizes["chrom"] + ":" + df_chromsizes["start"].astype(str) + "-" + df_chromsizes["end"].astype(str)
        df_chromsizes = df_chromsizes.set_index("chrom").loc[clr.chromnames].reset_index()
        df_chromsizes = df_chromsizes[['chrom', 'start', 'end', 'name']]
        df_loops = bioframe.complement(df_loops, df_chromsizes)
    elif mode == "inside":
        df_loops["end"] = df_loops["end"] - resolution
        df_loops["start"] = df_loops["start"] + resolution
    elif mode == "inside-1bin":
        df_loops["end"] = df_loops["end"] - 2 * resolution
        df_loops["start"] = df_loops["start"] + 2 * resolution
    elif mode == "whole":
        df_loops["end"] = df_loops["end"]
        df_loops["start"] = df_loops["start"]
    elif mode == "anchors":
        # initialize data of lists.
        data = {
            "chrom": df_loops.chrom,
            "start": df_loops.start,
            "end": df_loops.start + resolution,
            "loop_id": df_loops.index
        }
        df_bed_left = pd.DataFrame(data)

        data = {
            "chrom": df_loops.chrom,
            "start": df_loops.end - resolution,
            "end": df_loops.end,
            "loop_id": df_loops.index
        }
        df_bed_right = pd.DataFrame(data)

        df_loops = pd.concat([df_bed_left, df_bed_right])
    elif mode == "anchors_3bins":
        # initialize data of lists.
        data = {
            "chrom": df_loops.chrom,
            "start": df_loops.start - resolution,
            "end": df_loops.start + resolution * 2,
            "loop_id": df_loops.index
        }
        df_bed_left = pd.DataFrame(data)

        data = {
            "chrom": df_loops.chrom,
            "start": df_loops.end - resolution * 2,
            "end": df_loops.end + resolution,
            "loop_id": df_loops.index
        }
        df_bed_right = pd.DataFrame(data)

        df_loops = pd.concat([df_bed_left, df_bed_right])
    elif mode == "anchors_5bins":
        # initialize data of lists.
        data = {
            "chrom": df_loops.chrom,
            "start": df_loops.start - 2 * resolution,
            "end": df_loops.start + resolution * 3,
            "loop_id": df_loops.index
        }
        df_bed_left = pd.DataFrame(data)

        data = {
            "chrom": df_loops.chrom,
            "start": df_loops.end - resolution * 3,
            "end": df_loops.end + resolution * 2,
            "loop_id": df_loops.index
        }
        df_bed_right = pd.DataFrame(data)
        df_loops = pd.concat([df_bed_left, df_bed_right])
    elif mode == "end+1":
        data = {
            "chrom": df_loops.chrom,
            "start": df_loops.end,
            "end": df_loops.end + resolution,
        }
        df_loops = pd.DataFrame(data)
    elif mode == "end-1":
        data = {
            "chrom": df_loops.chrom,
            "start": df_loops.end - 2 * resolution,
            "end": df_loops.end - resolution,
        }
        df_loops = pd.DataFrame(data)
    elif mode == "start+1":
        data = {
            "chrom": df_loops.chrom,
            "start": df_loops.start + resolution,
            "end": df_loops.start + 2 * resolution,
        }
        df_loops = pd.DataFrame(data)
    elif mode == "start-1":
        data = {
            "chrom": df_loops.chrom,
            "start": df_loops.start - resolution,
            "end": df_loops.start,
        }
        df_loops = pd.DataFrame(data)
    elif mode == "end+-1":
        data = {
            "chrom": df_loops.chrom,
            "start": df_loops.end - 2 * resolution,
            "end": df_loops.end + resolution,
        }
        df_loops = pd.DataFrame(data)
    elif mode == "start+-1":
        data = {
            "chrom": df_loops.chrom,
            "start": df_loops.start - resolution,
            "end": df_loops.start + 2 * resolution,
        }
        df_loops = pd.DataFrame(data)
    elif mode == "end+-2":
        data = {
            "chrom": df_loops.chrom,
            "start": df_loops.end - 3 * resolution,
            "end": df_loops.end + 2 * resolution,
        }
        df_loops = pd.DataFrame(data)
    elif mode == "start+-2":
        data = {
            "chrom": df_loops.chrom,
            "start": df_loops.start - 2 * resolution,
            "end": df_loops.start + 3 * resolution,
        }
        df_loops = pd.DataFrame(data)
    elif mode == "end+12":
        data = {
            "chrom": df_loops.chrom,
            "start": df_loops.end,
            "end": df_loops.end + 2 * resolution,
        }
        df_loops = pd.DataFrame(data)
    elif mode == "end-12":
        data = {
            "chrom": df_loops.chrom,
            "start": df_loops.end - 3 * resolution,
            "end": df_loops.end - 1 * resolution,
        }
        df_loops = pd.DataFrame(data)
    elif mode == "start+12":
        data = {
            "chrom": df_loops.chrom,
            "start": df_loops.start + 1 * resolution,
            "end": df_loops.start + 3 * resolution,
        }
        df_loops = pd.DataFrame(data)
    elif mode == "start-12":
        data = {
            "chrom": df_loops.chrom,
            "start": df_loops.start - 2 * resolution,
            "end": df_loops.start - 0 * resolution,
        }
        df_loops = pd.DataFrame(data)
    # start
    elif mode == "end+012":
        data = {
            "chrom": df_loops.chrom,
            "start": df_loops.end - resolution,
            "end": df_loops.end + 2 * resolution,
        }
        df_loops = pd.DataFrame(data)
    elif mode == "end-012":
        data = {
            "chrom": df_loops.chrom,
            "start": df_loops.end - 3 * resolution,
            "end": df_loops.end - 0 * resolution,
        }
        df_loops = pd.DataFrame(data)
    elif mode == "start+012":
        data = {
            "chrom": df_loops.chrom,
            "start": df_loops.start + 0 * resolution,
            "end": df_loops.start + 3 * resolution,
        }
        df_loops = pd.DataFrame(data)
    elif mode == "start-012":
        data = {
            "chrom": df_loops.chrom,
            "start": df_loops.start - 2 * resolution,
            "end": df_loops.start + 1 * resolution,
        }
        df_loops = pd.DataFrame(data)
    # end
    # start
    elif mode == "end+01":
        data = {
            "chrom": df_loops.chrom,
            "start": df_loops.end - resolution,
            "end": df_loops.end + 1 * resolution,
        }
        df_loops = pd.DataFrame(data)
    elif mode == "end-01":
        data = {
            "chrom": df_loops.chrom,
            "start": df_loops.end - 2 * resolution,
            "end": df_loops.end - 0 * resolution,
        }
        df_loops = pd.DataFrame(data)
    elif mode == "start+01":
        data = {
            "chrom": df_loops.chrom,
            "start": df_loops.start + 0 * resolution,
            "end": df_loops.start + 2 * resolution,
        }
        df_loops = pd.DataFrame(data)
    elif mode == "start-01":
        data = {
            "chrom": df_loops.chrom,
            "start": df_loops.start - 1 * resolution,
            "end": df_loops.start + 1 * resolution,
        }
        df_loops = pd.DataFrame(data)
    # end
    elif mode == "end-2":
        data = {
            "chrom": df_loops.chrom,
            "start": df_loops.end - 3 * resolution,
            "end": df_loops.end - 2 * resolution,
        }
        df_loops = pd.DataFrame(data)
    elif mode == "end+2":
        data = {
            "chrom": df_loops.chrom,
            "start": df_loops.end + 1 * resolution,
            "end": df_loops.end + 2 * resolution,
        }
        df_loops = pd.DataFrame(data)
    elif mode == "start+2":
        data = {
            "chrom": df_loops.chrom,
            "start": df_loops.start + 2 * resolution,
            "end": df_loops.start + 3 * resolution,
        }
        df_loops = pd.DataFrame(data)
    elif mode == "start-2":
        data = {
            "chrom": df_loops.chrom,
            "start": df_loops.start - 2 * resolution,
            "end": df_loops.start - 1 * resolution,
        }
        df_loops = pd.DataFrame(data)
    else:
        AssertionError("mode not found")
    return df_loops


def featureOccurenceInLoop(
    file_loops="results/long_loops/%sAB_regular_loops.bed",
    time="0",
    mode="start",
    file_features="results/V_H3K27ac_V3K4me1_atacseqWang_30k_noPromoters/V_H3K27ac_V3K4me1_atacseqWang_30k_noPromoters.bed3",
):
    """
    Compute occurences of features in part of loops defined by mode variable

    Parameters
    ----------
    file_loops : str
        Path to the bed file of loops
    time : str
    mode : str
        Which part of the loops to consider
    file_features : str
        File with features to count
    """
    df_loops = load_BedInMode(file_loops, time, mode)

    df_features = bioframe.read_table(file_features, schema="bed3")

    # df_loopsWithEnh = bioframe.overlap(df_loops, df_features, how='left', return_index=True, keep_order=True) #return_input=False, return_overlap=True,
    # df_loopsWithEnh = df_loopsWithEnh.dropna()
    # Percent_anchors_with_enh = df_loopsWithEnh['index'].unique().shape[0] * 100 / df_loops.shape[0]
    # easier way
    df_loopsWithFeature = count_overlaps(df_loops, df_features)
    Percent_anchors_withFeature = (
        sum(df_loopsWithFeature["count"] >= 1) * 100 / df_loops.shape[0]
    )
    return Percent_anchors_withFeature


def pValue_featureOccurenceInLoop(
    file_loops="results/long_loops/%sAB_regular_loops.bed",
    time="0",
    mode="start",
    N_shuffle=100,
    file_features="results/V_H3K27ac_V3K4me1_atacseqWang_30k_noPromoters/V_H3K27ac_V3K4me1_atacseqWang_30k_noPromoters.bed3",
    name="q00",
    genome_file="data/genome/dicty.chrom.sizes",
    pic_path="results/pics/paper",
    resolution=2000,
    fraction=0.1,
    Fraction=0.1,
    return_df=False
):
    """
    Plot feature occurence with permutation test p-value for occurences of features in part of loops defined by mode variable

    Parameters
    ----------
    file_loops : str
        Path to the bed file of loops
    time : str
    mode : str
        Which part of the loops to consider
    N_shuffle : int
        How many times to shuffle the loops
    file_features : str
        File with features to count
    name : str
        Name of bed file to consider (generally not needed)
    """
    # load las-loops
    df_loops = load_BedInMode(file_loops, time, mode, resolution=resolution)
    df_loops.loc[:, ["chrom", "start", "end"]].to_csv(
        "data/tmp.bed", sep="\t", index=False, header=False
    )
    df_loops = BedTool("data/tmp.bed")

    ncRNA_andFeature = BedTool(file_features)
    # create shuffled control
    inter_shuffle_vec = []
    for i in range(N_shuffle):
        las_loops_shuffled = df_loops.shuffle(g=genome_file, chrom=True, seed=i)
        inter_shuffle = (
            ncRNA_andFeature.intersect(las_loops_shuffled, wa=True, f=fraction, F=Fraction)
            .to_dataframe()
            .drop_duplicates()
        )
        inter_shuffle_vec.append(inter_shuffle.shape[0])
    inter_shuffle_df = pd.DataFrame({"Shuffle": inter_shuffle_vec})
    inter_lasLoops = (
        ncRNA_andFeature.intersect(df_loops, wa=True, f=fraction, F=Fraction).to_dataframe().drop_duplicates()
    )
    if return_df:
        return inter_shuffle_df, inter_lasLoops
    else:
        print(inter_lasLoops.shape[0])
        fig = sns.histplot(
            data=inter_shuffle_df, x="Shuffle", kde=True, stat="percent", binwidth=1
        )
        fig.axvline(inter_lasLoops.shape[0], color="red", lw=3)
        p_value = np.round(
            np.min(
                [
                    len(
                        inter_shuffle_df[
                            inter_shuffle_df["Shuffle"] > inter_lasLoops.shape[0]
                        ]
                    )
                    / N_shuffle,
                    len(
                        inter_shuffle_df[
                            inter_shuffle_df["Shuffle"] < inter_lasLoops.shape[0]
                        ]
                    )
                    / N_shuffle,
                ]
            ),
            3,
        )
        if p_value == 0:
            p_value = 1 / N_shuffle
            plt.annotate(
                "p-value < " + str(p_value),
                xy=(0.95, 0.95),
                xycoords="axes fraction",
                bbox=dict(facecolor="pink", alpha=0.5),
                horizontalalignment="right",
                fontsize=12,
            )
        else:
            plt.annotate(
                "p-value = " + str(p_value),
                xy=(0.95, 0.95),
                xycoords="axes fraction",
                bbox=dict(facecolor="pink", alpha=0.5),
                horizontalalignment="right",
                fontsize=12,
            )

        plt.savefig("%s/%s.pdf" % (pic_path, name), bbox_inches="tight")
        plt.show()
        # plt.clf()


def pValue_LoopAnchor_insideFeature(
    file_loops="results/long_loops/%sAB_regular_loops.bed",
    time="0",
    mode="anchors",
    N_shuffle=100,
    file_features="results/V_H3K27ac_V3K4me1_atacseqWang_30k_noPromoters/V_H3K27ac_V3K4me1_atacseqWang_30k_noPromoters.bed3",
    name="q00",
):
    # load las-loops
    df_loops = load_BedInMode(file_loops, time, mode)
    df_loops.loc[:, ["chrom", "start", "end"]].to_csv(
        "data/genome/tmp.bed", sep="\t", index=False, header=False
    )
    df_loops = BedTool("data/genome/tmp.bed")

    ncRNA_andFeature = BedTool(file_features)
    # create shuffled control
    inter_shuffle_vec = []
    for i in range(N_shuffle):
        las_loops_shuffled = df_loops.shuffle(
            g="data/genome/dicty.chrom.sizes", chrom=True, seed=i
        )
        inter_shuffle = ncRNA_andFeature.intersect(las_loops_shuffled, f=1.0)
        inter_shuffle_vec.append(inter_shuffle.count())
    inter_shuffle_df = pd.DataFrame({"Shuffle": inter_shuffle_vec})
    inter_lasLoops = ncRNA_andFeature.intersect(df_loops, f=1.0)
    print(inter_lasLoops.count())
    fig = sns.histplot(
        data=inter_shuffle_df, x="Shuffle", kde=True, stat="percent", binwidth=1
    )
    fig.axvline(inter_lasLoops.count(), color="red", lw=3)
    p_value = np.round(
        np.min(
            [
                len(
                    inter_shuffle_df[
                        inter_shuffle_df["Shuffle"] > inter_lasLoops.count()
                    ]
                )
                / N_shuffle,
                len(
                    inter_shuffle_df[
                        inter_shuffle_df["Shuffle"] < inter_lasLoops.count()
                    ]
                )
                / N_shuffle,
            ]
        ),
        3,
    )
    if p_value == 0:
        p_value = 1 / N_shuffle
        plt.annotate(
            "p-value < " + str(p_value),
            xy=(0.95, 0.95),
            xycoords="axes fraction",
            bbox=dict(facecolor="pink", alpha=0.5),
            horizontalalignment="right",
            fontsize=12,
        )
    else:
        plt.annotate(
            "p-value = " + str(p_value),
            xy=(0.95, 0.95),
            xycoords="axes fraction",
            bbox=dict(facecolor="pink", alpha=0.5),
            horizontalalignment="right",
            fontsize=12,
        )
    #         plt.show()

    # name_ncRNA = Path(file).stem
    # fig.set(xlim=(0, 40))

    plt.savefig("results/pics/paper/%s.pdf" % (name), bbox_inches="tight")
    plt.show()


def pValue_featureOccurenceInLoopByName(
    file_loops="results/long_loops/%sAB_regular_loops.bed",
    time="0",
    mode="start",
    N_shuffle=100,
    file_features="results/V_H3K27ac_V3K4me1_atacseqWang_30k_noPromoters/V_H3K27ac_V3K4me1_atacseqWang_30k_noPromoters.bed3",
    name="q00",
    genome_file="data/genome/dicty.chrom.sizes",
    pic_path="results/pics/paper",
    resolution=2000,
):
    """
    Plot feature occurence with permutation test p-value for occurences of features in part of loops defined by mode variable

    Parameters
    ----------
    file_loops : str
        Path to the bed file of loops
    time : str
    mode : str
        Which part of the loops to consider
    N_shuffle : int
        How many times to shuffle the loops
    file_features : str
        File with features to count
    name : str
        Name of bed file to consider (generally not needed)
    """
    # load all loops
    loops_path = "data/loops_quantifyChromosight/0AB_chromosight_quantifyMarkedGood.zeroLevel.filtered.bed"
    AllLoops_df = bioframe.read_table(loops_path, schema="bed3")
    AllLoops_df["name"] = (
        AllLoops_df.chrom
        + ":"
        + AllLoops_df.start.astype(str)
        + "-"
        + AllLoops_df.end.astype(str)
    )
    # load loops
    df_loops = load_BedInMode(file_loops, time, mode, resolution=resolution)
    df_loops["name"] = (
        df_loops.chrom
        + ":"
        + df_loops.start.astype(str)
        + "-"
        + df_loops.end.astype(str)
    )

    ncRNA_andFeature = bioframe.read_table(file_features, schema="bed3")
    ncRNA_andFeature["name"] = (
        ncRNA_andFeature.chrom
        + ":"
        + ncRNA_andFeature.start.astype(str)
        + "-"
        + ncRNA_andFeature.end.astype(str)
    )

    inter_Loops = pd.merge(df_loops, ncRNA_andFeature, on="name", how="inner").shape[0]
    # print(tmp.shape[0])
    # print(np.round(tmp.shape[0]*100/loops_thresBMA.shape[0], 3))
    # create shuffled control
    import random

    thelist = range(AllLoops_df.shape[0])
    inter_shuffle_vec = []
    for i in range(N_shuffle):
        index_shuffled = random.sample(thelist, df_loops.shape[0])
        index_shuffled.sort()

        loops_shuffled = AllLoops_df.iloc[index_shuffled]
        inter_Loops_shuffled = pd.merge(
            loops_shuffled, ncRNA_andFeature, on="name", how="inner"
        ).shape[0]
        inter_shuffle_vec.append(inter_Loops_shuffled)
    inter_shuffle_df = pd.DataFrame({"Shuffle": inter_shuffle_vec})
    print(inter_Loops)
    fig = sns.histplot(
        data=inter_shuffle_df, x="Shuffle", kde=True, stat="percent", binwidth=1
    )
    fig.axvline(inter_Loops, color="red", lw=3)
    p_value = np.round(
        np.min(
            [
                len(inter_shuffle_df[inter_shuffle_df["Shuffle"] > inter_Loops])
                / N_shuffle,
                len(inter_shuffle_df[inter_shuffle_df["Shuffle"] < inter_Loops])
                / N_shuffle,
            ]
        ),
        3,
    )
    if p_value == 0:
        p_value = 1 / N_shuffle
        plt.annotate(
            "p-value < " + str(p_value),
            xy=(0.95, 0.95),
            xycoords="axes fraction",
            bbox=dict(facecolor="pink", alpha=0.5),
            horizontalalignment="right",
            fontsize=12,
        )
    else:
        plt.annotate(
            "p-value = " + str(p_value),
            xy=(0.95, 0.95),
            xycoords="axes fraction",
            bbox=dict(facecolor="pink", alpha=0.5),
            horizontalalignment="right",
            fontsize=12,
        )
    #         plt.show()

    # name_ncRNA = Path(file).stem
    # fig.set(xlim=(0, 40))

    plt.savefig("%s/%s.pdf" % (pic_path, name), bbox_inches="tight")
    plt.show()
    # plt.clf()


def create_rnaseq_df(
    rnaseq_dir="data/all_gene_markdup_htseq", need_coord=False, need_TSS=False
):
    feature = BedTool("data/genome/genes_noContigs.bed")

    tpm_df = pd.read_table(rnaseq_dir, header=None)
    tpm_df.columns = ["name", "0A", "0B", "2A", "2B", "5A", "5B", "8A", "8B"]
    tpm_df_withLength = pd.merge(
        tpm_df, feature.to_dataframe().set_index("name"), on="name"
    )
    tpm_df_withLength["length"] = tpm_df_withLength["end"] - tpm_df_withLength["start"]
    tpm_df_withLength = tpm_df_withLength.reset_index()[
        ["name", "0A", "0B", "2A", "2B", "5A", "5B", "8A", "8B", "length"]
    ].set_index("name")

    from bioinfokit.analys import norm

    nm = norm()
    nm.tpm(df=tpm_df_withLength, gl="length")
    # get TPM normalized dataframe
    tpm_df = nm.tpm_norm
    tpm_df["0AB"] = tpm_df[["0A", "0B"]].mean(axis=1)  # np.log2()
    tpm_df["2AB"] = tpm_df[["2A", "2B"]].mean(axis=1)  # np.log2()
    tpm_df["5AB"] = tpm_df[["5A", "5B"]].mean(axis=1)  # np.log2()
    tpm_df["8AB"] = tpm_df[["8A", "8B"]].mean(axis=1)  # np.log2()
    tpm_df = tpm_df.reset_index()[["name", "0AB", "2AB", "5AB", "8AB"]]
    if need_coord:
        tpm_df_withLength = tpm_df.set_index("name").join(
            feature.to_dataframe().set_index("name"), on="name", how="inner"
        )
        tpm_df = tpm_df_withLength.reset_index()[
            ["chrom", "start", "end", "name", "0AB", "strand", "2AB", "5AB", "8AB"]
        ]
    if need_TSS:
        tpm_df_withLength = tpm_df.set_index("name").join(
            feature.to_dataframe().set_index("name"), on="name", how="inner"
        )
        tpm_df = tpm_df_withLength.reset_index()[
            ["chrom", "start", "end", "name", "0AB", "strand", "2AB", "5AB", "8AB"]
        ]
        tpm_df["TSS_start"] = tpm_df["start"].tolist()
        tpm_df.loc[tpm_df.strand == "-", "TSS_start"] = tpm_df["end"] - 1
        tpm_df["TSS_end"] = tpm_df["TSS_start"] + 1
        tpm_df = tpm_df.reset_index()[
            [
                "chrom",
                "TSS_start",
                "TSS_end",
                "name",
                "0AB",
                "strand",
                "2AB",
                "5AB",
                "8AB",
            ]
        ]
    if need_TSS and need_coord:
        ("current mode in not supported")

    return tpm_df


def create_rnaseq_df_rosengarten():
    from pathlib import Path
    import glob

    file_path = "."
    # load expression files
    files = glob.glob("data/rnaseq_rosengarten/GSM*raw.txt")
    files.sort()
    rosengartnePolyA_expr = pd.read_table(files[0], header=0)
    # rosengartnePolyA_expr.columns = ['name', Path(files[0]).stem.split('_', 2)[2]]
    times = []
    for file in files[1:]:
        tmp = pd.read_table(file, header=0)
        # tmp.columns = ['name', Path(file).stem.split('_', 2)[2]]
        rosengartnePolyA_expr = rosengartnePolyA_expr.merge(tmp, on="ddb_g")
        times.append(Path(file).stem.split("_", 4)[3])
    rosengartnePolyA_expr["name"] = rosengartnePolyA_expr["ddb_g"]
    feature = BedTool("data/genome/genes_noContigs.bed").to_dataframe()
    tpm_df_withLength = rosengartnePolyA_expr.set_index("name").join(
        feature.set_index("name"), on="name"
    )

    tpmRosengarten_df_withLength = tpm_df_withLength.filter(like="FDrep", axis=1)
    tpmRosengarten_df_withLength["length"] = (
        tpm_df_withLength["end"] - tpm_df_withLength["start"]
    )
    tpmRosengarten_df_withLength = tpmRosengarten_df_withLength.query("length > 0")
    # tpm_df_withLength = tpm_df_withLength.reset_index()[['name', '0A', '0B', '2A', '2B', '5A', '5B', '8A', '8B', 'length']].set_index('name')

    from bioinfokit.analys import norm
    import re

    nm = norm()
    nm.tpm(df=tpmRosengarten_df_withLength, gl="length")
    # get TPM normalized dataframe
    tpm_rosengarten_df = nm.tpm_norm
    tpm_rosengarten_df.columns = [
        re.sub("_raw", "", x)
        for x in tpm_rosengarten_df.filter(like="FDrep", axis=1).columns
    ]
    rosengartenHours_list = np.unique(
        [
            re.sub("FDrep[12]_", "", x)
            for x in tpm_rosengarten_df.filter(like="FDrep", axis=1).columns
        ]
    ).tolist()
    for name in rosengartenHours_list:
        tpm_rosengarten_df[name] = np.nanmean(
            tpm_rosengarten_df.filter(like=name, axis=1), axis=1
        ).tolist()

    tpm_rosengarten_df_withLength = tpm_rosengarten_df.join(
        feature.set_index("name"), on="name"
    ).reset_index()
    return tpm_rosengarten_df_withLength


def compute_rescaleSize_zoo(
    cooler_path="../dicty_hic_distiller/subsampled/0AB.filtered.mcool::/resolutions/",
    resolution=2000,
    nthreads=4,
    mode="bedpe",
    chromsizes_path="data/genome/dicty.chrom.sizes",
    bedpe_path="data/loops_quantifyChromosight/0AB_chromosight_quantifyMarkedGood.bedpe",
    rescale=False,
    local=False,
    score=True,
    coef_flank=3,
    ignore_diags=2,
    rescale_flank=1.0,
    vmax=2.5,
    vmin=0.5,
    organism="nonhuman",
    plot_name="",
):
    clr = cooler.Cooler(cooler_path + str(resolution))

    # chromsizes
    df_chromsizes = clr.chromsizes.reset_index()
    df_chromsizes.columns = ["chrom", "end"]
    df_chromsizes.loc[:, "start"] = 0
    df_chromsizes["name"] = (
        df_chromsizes["chrom"]
        + ":"
        + df_chromsizes["start"].astype(str)
        + "-"
        + df_chromsizes["end"].astype(str)
    )
    df_chromsizes = df_chromsizes.set_index("chrom").loc[clr.chromnames].reset_index()
    df_chromsizes["end"] = clr.chromsizes.tolist()
    df_chromsizes = df_chromsizes[["chrom", "start", "end", "name"]]

    df_bedpe = pd.read_table(bedpe_path, header=None)  # .iloc[0:20,:]
    if df_bedpe.shape[1] == 3:
        df_bedpe.columns = ["chrom", "start", "end"]
        paired_sites = df_bedpe[df_bedpe["end"] - df_bedpe["start"] > resolution * 4]
        median_size = np.median(paired_sites.end - paired_sites.start)
    elif df_bedpe.shape[1] == 6:
        df_bedpe.columns = ["chrom1", "start1", "end1", "chrom2", "start2", "end2"]
        paired_sites = df_bedpe[df_bedpe["start2"] - df_bedpe["end1"] > resolution * 2]
        median_size = np.median(paired_sites.start2 - paired_sites.end1)
    else:
        raise AssertionError("Not bedpe?")
    if organism == "human":
        expected = pd.read_csv(
            "GSE63525_GM12878_insitu_primary.5kb.expected.tsv", sep="\t"
        )
    else:
        expected = cooltools.expected_cis(
            clr,
            view_df=df_chromsizes,
            nproc=nthreads,
            chunksize=1_000_000,
            ignore_diags=0,
        )

    flank = int((median_size * coef_flank // resolution) * resolution)
    return median_size  # (int(1+flank * 2// resolution))


def dataframe_difference(df1: None, df2: None, which=None):
    """
    Find rows which are different between two DataFrames.

    Parameters
    ----------
    df1 : pandas.DataFrame
    df2 : pandas.DataFrame
    which : str
        Provide a value 'both', 'left', or 'right' to specify which rows to select.
    Returns
    -------
    diff_df : pandas.DataFrame

    """
    comparison_df = df1.merge(df2, indicator=True, how="outer")
    if which is None:
        diff_df = comparison_df[comparison_df["_merge"] != "both"]
    else:
        diff_df = comparison_df[comparison_df["_merge"] == which]
    #     diff_df.to_csv('data/diff.csv')
    return diff_df


def compute_avMiddlePixels(
    resolution=2000,
    flank_bins=2,
    cell_cycle=False,
    Timing="0",
    use_bed="one_for_all",
    bed_file=None,
):
    """
    Compute the average number of pixels in the middle of each bin.

    Parameters
    ----------
    resolution : int
        Resolution of the cooler.
        Default is 2000.
    flank_bins : int
        Number of flanking bins.
    cell_cycle : bool
        Whether to use data for dicty cell cycle.
    Timing : str
        Which timing to use.
    use_bed : str
        There are some predetermined bed files.
        Default is 'one_for_all'.
        Options are 'one_for_all' and 'one_for_one'.
    bed_file : str
        Provide the path to the bed file.
    Returns
    -------
    avMiddlePixels : float
        Average number of pixels in the middle of each bin.
    """
    # compute size of flanking in bp
    flank = flank_bins * resolution
    if use_bed == "one_for_all":
        bed_file = (
            "data/loops_quantifyChromosight/0AB_chromosight_quantifyMarkedGood.bedpe"
        )
        # bed_file = "data/loops_quantifyChromosight/0AB_chromosight_quantifyMarkedGood.zeroLevel.filtered.bedpe"
        # bed_file = "data/loops_quantifyChromosight/0AB_consecutive_0binsAdded.filtered.bedpe"
        clr = cooler.Cooler(
            "../dicty_hic_distiller/subsampled/%sAB.filtered.mcool::/resolutions/"
            % (Timing)
            + str(resolution)
        )
    elif cell_cycle:
        bed_file = (
            "data/loops_quantifyChromosight/%sAB_CC_chromosight_quantifyMarkedGood.bedpe"
            % (Timing)
        )
        clr = cooler.Cooler(
            "data/hic_dicty_mitosis/merged/subsampled/%sAB.filtered.mcool::/resolutions/"
            % (Timing)
            + str(resolution)
        )
    elif use_bed == "provided_bed":
        clr = cooler.Cooler(
            "../dicty_hic_distiller/subsampled/%sAB.filtered.mcool::/resolutions/"
            % (Timing)
            + str(resolution)
        )
    else:
        bed_file = (
            "data/loops_quantifyChromosight/%sAB_chromosight_quantifyMarkedGood.bedpe"
            % (Timing)
        )
        clr = cooler.Cooler(
            "../dicty_hic_distiller/subsampled/%sAB.filtered.mcool::/resolutions/"
            % (Timing)
            + str(resolution)
        )

    df_bedpe = bioframe.read_table(bed_file, schema='bedpe')
    # df_bedpe = df_bedpe[['chrom1', 'start1', 'end2']]
    print("bedpe is created")
    # df_bedpe.columns = ['chrom', 'start', 'end']
    # select only the regions that are more than 2 diagonals away from main diagonal
    paired_sites = df_bedpe[df_bedpe["end2"] - df_bedpe["start1"] > resolution * 2]

    # extract chromsizes from cooler
    df_chromsizes = clr.chromsizes.reset_index()
    df_chromsizes.columns = ["chrom", "end"]
    df_chromsizes.loc[:, "start"] = 0
    df_chromsizes["name"] = (
        df_chromsizes["chrom"]
        + ":"
        + df_chromsizes["start"].astype(str)
        + "-"
        + df_chromsizes["end"].astype(str)
    )
    df_chromsizes = df_chromsizes.set_index("chrom").loc[clr.chromnames].reset_index()
    df_chromsizes["end"] = clr.chromsizes.tolist()
    df_chromsizes = df_chromsizes[["chrom", "start", "end", "name"]]

    # compute expected values for each diagonal
    expected = cooltools.expected_cis(
        clr, view_df=df_chromsizes, nproc=4, chunksize=1_000_000
    )
    # compute average number of pixels in the middle of each bin
    oe_stack = cooltools.pileup(
        clr, paired_sites, view_df=df_chromsizes, expected_df=expected, flank=flank
    )
    av_middle = np.nansum(np.nansum(oe_stack, axis=0), axis=0)
    # select only loops without white stripes
    # av_middle[np.sum(np.isnan(oe_stack[:,:,:]).any(1), axis=0) > 0] = np.nan
    return av_middle


import textwrap


def wrap_labels(ax, width, break_long_words=False):
    labels = []
    for label in ax.get_xticklabels():
        text = label.get_text()
        labels.append(
            textwrap.fill(text, width=width, break_long_words=break_long_words)
        )
    ax.set_xticklabels(labels, rotation=0)


def get_expression_sum_interesectingFeature(
    feature=None, tpm_df=None, start_col="start", group=None
):
    tpm_bed = BedTool.from_dataframe(
        tpm_df.iloc[:, [0, 1, 2, 3]].sort_values(by=["chrom", start_col])
    )
    genes_withEnh = tpm_bed.intersect(feature, wb=True).to_dataframe()
    # create column with enhancers coordinates
    genes_withEnh["coords"] = (
        genes_withEnh.score
        + ":"
        + genes_withEnh.strand.astype("str")
        + "-"
        + genes_withEnh.thickStart.astype("str")
    )
    # creat dic
    genesInWindowsEnh_dic = {}
    enhInWindowsGenes_dic = {}
    for i in range(genes_withEnh.shape[0]):
        enh_name = genes_withEnh.loc[i, "coords"]
        gene_name = genes_withEnh.loc[i, "name"]
        # if tpm_df.query('name in @gene_name').shape[0] > 0:
        # append term dict
        # prevent duplicates (??)
        if enh_name in genesInWindowsEnh_dic.keys():
            del genesInWindowsEnh_dic[enh_name]
        if gene_name in enhInWindowsGenes_dic.keys():
            del enhInWindowsGenes_dic[gene_name]
        genesInWindowsEnh_dic[enh_name] = gene_name
        enhInWindowsGenes_dic[gene_name] = enh_name
    df = (
        tpm_df.set_index("name")
        .loc[:, ["delta_8_0"]]
        .groupby(by=enhInWindowsGenes_dic, axis=0)
        .mean()
        .reset_index()
    )
    df.loc[:, "group"] = group
    return df


def get_expression_sum_aroundFeature(
    feature_window=None, feaure=None, tpm_df=None, group=None
):
    tpm_bed = BedTool.from_dataframe(
        tpm_df.iloc[:, [0, 1, 2, 3]].sort_values(by=["chrom", "start"])
    )
    genes_withEnh = tpm_bed.intersect(feature_window, wb=True).to_dataframe()
    genes_withEnh["coords"] = (
        genes_withEnh.score
        + ":"
        + genes_withEnh.strand.astype("str")
        + "-"
        + genes_withEnh.thickStart.astype("str")
    )
    # select genes intersecting enhancers with fraction
    genes_containingEnh = tpm_bed.intersect(feature, wb=True, f=0.01).to_dataframe()
    genes_containingEnh["coords"] = (
        genes_containingEnh.score
        + ":"
        + genes_containingEnh.strand.astype("str")
        + "-"
        + genes_containingEnh.thickStart.astype("str")
    )

    genesInWIndowsEnh_dic = {}
    enhInWIndowsGenes_dic = {}
    for i in range(genes_withEnh.shape[0]):
        enh_name = genes_withEnh.loc[i, "coords"]
        gene_name = genes_withEnh.loc[i, "name"]
        # if tpm_df.query('name in @gene_name').shape[0] > 0:
        # append term dict
        if enh_name in genesInWIndowsEnh_dic.keys():
            del genesInWIndowsEnh_dic[enh_name]
        if gene_name in enhInWIndowsGenes_dic.keys():  # TODO what is going on here?
            del enhInWIndowsGenes_dic[gene_name]
        genesInWIndowsEnh_dic[enh_name] = gene_name
        enhInWIndowsGenes_dic[gene_name] = enh_name
    df = (
        tpm_df.set_index("name")
        .groupby(by=enhInWIndowsGenes_dic, axis=0)
        .mean()
        .reset_index()
    )
    df.loc[:, "group"] = group
    return df


def const_line(*args, **kwargs):
    x = np.arange(-3, 15, 0.5)
    y = x
    plt.plot(y, x, c="k", linestyle="dashed", linewidth=2)


def table_to_BigWig(long_name, df_chromsizes, chroms, start, end, values):
    bw = pyBigWig.open("bw/" + long_name + ".bw", "w")
    bw.addHeader(list(df_chromsizes.itertuples(index=False, name=None)), maxZooms=0)
    bw.addEntries(chroms, starts, ends=ends, values=values0)
    bw.close()

# Define a function to calculate promoter coordinates
def define_promoter(row):
    if row[6] == "-" and row[4] > 100:
        return [row[0], row[4] - 100, row[4] + 300]
    elif row[6] == "+" and row[3] > 300:
        return [row[0], row[3] - 300, row[3] + 100]
    else:
        return None
# Read the GFF file into a DataFrame
# gff_file = "data/genome/dicty_fixedSourceName.gff"
# gff_df = pd.read_csv(gff_file, sep='\t', header=None)

# # Filter for gene annotations
# genes_df = gff_df[gff_df[2] == "gene"]
# # Apply the function to each row and drop None values
# promoters_df = pd.DataFrame(genes_df.apply(define_promoter, axis=1).tolist()).dropna()

# save to a BED file
# promoters_df.to_csv("data/genome/promoters.bed", sep='\t', index=False, header=False)

# %%
from seaborn.distributions import _DistributionPlotter

def kdeplot_uniColor(
    data=None, *, x=None, y=None, hue=None, weights=None,
    palette=None, hue_order=None, hue_norm=None, color=None, fill=None,
    multiple="layer", common_norm=True, common_grid=False, cumulative=False,
    bw_method="scott", bw_adjust=1, warn_singular=True, log_scale=None,
    levels=10, thresh=.05, gridsize=200, cut=3, clip=None,
    legend=True, cbar=False, cbar_ax=None, cbar_kws=None, ax=None, draw_levels=None,
    **kwargs,
):

    # --- Start with backwards compatability for versions < 0.11.0 ----------------

    # Handle (past) deprecation of `data2`
    if "data2" in kwargs:
        msg = "`data2` has been removed (replaced by `y`); please update your code."
        raise TypeError(msg)

    # Handle deprecation of `vertical`
    vertical = kwargs.pop("vertical", None)
    if vertical is not None:
        if vertical:
            action_taken = "assigning data to `y`."
            if x is None:
                data, y = y, data
            else:
                x, y = y, x
        else:
            action_taken = "assigning data to `x`."
        msg = textwrap.dedent(f"""\n
        The `vertical` parameter is deprecated; {action_taken}
        This will become an error in seaborn v0.14.0; please update your code.
        """)
        warnings.warn(msg, UserWarning, stacklevel=2)

    # Handle deprecation of `bw`
    bw = kwargs.pop("bw", None)
    if bw is not None:
        msg = textwrap.dedent(f"""\n
        The `bw` parameter is deprecated in favor of `bw_method` and `bw_adjust`.
        Setting `bw_method={bw}`, but please see the docs for the new parameters
        and update your code. This will become an error in seaborn v0.14.0.
        """)
        warnings.warn(msg, UserWarning, stacklevel=2)
        bw_method = bw

    # Handle deprecation of `kernel`
    if kwargs.pop("kernel", None) is not None:
        msg = textwrap.dedent("""\n
        Support for alternate kernels has been removed; using Gaussian kernel.
        This will become an error in seaborn v0.14.0; please update your code.
        """)
        warnings.warn(msg, UserWarning, stacklevel=2)

    # Handle deprecation of shade_lowest
    shade_lowest = kwargs.pop("shade_lowest", None)
    if shade_lowest is not None:
        if shade_lowest:
            thresh = 0
        msg = textwrap.dedent(f"""\n
        `shade_lowest` has been replaced by `thresh`; setting `thresh={thresh}.
        This will become an error in seaborn v0.14.0; please update your code.
        """)
        warnings.warn(msg, UserWarning, stacklevel=2)

    # Handle "soft" deprecation of shade `shade` is not really the right
    # terminology here, but unlike some of the other deprecated parameters it
    # is probably very commonly used and much hard to remove. This is therefore
    # going to be a longer process where, first, `fill` will be introduced and
    # be used throughout the documentation. In 0.12, when kwarg-only
    # enforcement hits, we can remove the shade/shade_lowest out of the
    # function signature all together and pull them out of the kwargs. Then we
    # can actually fire a FutureWarning, and eventually remove.
    shade = kwargs.pop("shade", None)
    if shade is not None:
        fill = shade
        msg = textwrap.dedent(f"""\n
        `shade` is now deprecated in favor of `fill`; setting `fill={shade}`.
        This will become an error in seaborn v0.14.0; please update your code.
        """)
        warnings.warn(msg, FutureWarning, stacklevel=2)

    # Handle `n_levels`
    # This was never in the formal API but it was processed, and appeared in an
    # example. We can treat as an alias for `levels` now and deprecate later.
    levels = kwargs.pop("n_levels", levels)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    from seaborn.distributions import _DistributionPlotter
    # _DistributionPlotter.plot_bivariate_density_UniColor = MethodType(plot_bivariate_density_UniColor, _DistributionPlotter)
    p = _DistributionPlotter_UniColor(
        data=data,
        variables=dict(x=x, y=y, hue=hue, weights=weights),
    )

    p.map_hue(palette=palette, order=hue_order, norm=hue_norm)

    if ax is None:
        ax = plt.gca()

    p._attach(ax, allowed_types=["numeric", "datetime"], log_scale=log_scale)

    method = ax.fill_between if fill else ax.plot
    from seaborn.utils import (
    _default_color,
)
    color = _default_color(method, hue, color, kwargs)

    if not p.has_xy_data:
        return ax

    # Pack the kwargs for statistics.KDE
    estimate_kws = dict(
        bw_method=bw_method,
        bw_adjust=bw_adjust,
        gridsize=gridsize,
        cut=cut,
        clip=clip,
        cumulative=cumulative,
    )

    if p.univariate:

        plot_kws = kwargs.copy()

        p.plot_univariate_density(
            multiple=multiple,
            common_norm=common_norm,
            common_grid=common_grid,
            fill=fill,
            color=color,
            legend=legend,
            warn_singular=warn_singular,
            estimate_kws=estimate_kws,
            **plot_kws,
        )

    else:

        p.plot_bivariate_density_uniColor(
            common_norm=common_norm,
            fill=fill,
            levels=levels,
            thresh=thresh,
            legend=legend,
            color=color,
            warn_singular=warn_singular,
            draw_levels=draw_levels,
            cbar=cbar,
            cbar_ax=cbar_ax,
            cbar_kws=cbar_kws,
            estimate_kws=estimate_kws,
            **kwargs,
        )

    return ax

# create inreritent class
from seaborn.distributions import _DistributionPlotter
import warnings
from numbers import Number
# from seaborn.utils import (
#     remove_na,
#     _get_transform_functions,
#     _kde_support,
#     _check_argument,
#     _assign_default_kwargs,
#     _default_color,
# )
class _DistributionPlotter_UniColor(_DistributionPlotter):
    def plot_bivariate_density_uniColor(
        self,
        common_norm,
        fill,
        levels,
        thresh,
        color,
        legend,
        cbar,
        draw_levels,
        warn_singular,
        cbar_ax,
        cbar_kws,
        estimate_kws,
        **contour_kws,
    ):

        contour_kws = contour_kws.copy()

        estimator = KDE(**estimate_kws)

        if not set(self.variables) - {"x", "y"}:
            common_norm = False

        all_data = self.plot_data.dropna()

        # Loop through the subsets and estimate the KDEs
        densities, supports = {}, {}

        for sub_vars, sub_data in self.iter_data("hue", from_comp_data=True):

            # Extract the data points from this sub set
            observations = sub_data[["x", "y"]]
            min_variance = observations.var().fillna(0).min()
            observations = observations["x"], observations["y"]

            # Extract the weights for this subset of observations
            if "weights" in self.variables:
                weights = sub_data["weights"]
            else:
                weights = None

            # Estimate the density of observations at this level
            singular = math.isclose(min_variance, 0)
            try:
                if not singular:
                    density, support = estimator(*observations, weights=weights)
            except np.linalg.LinAlgError:
                # Testing for 0 variance doesn't catch all cases where scipy raises,
                # but we can also get a ValueError, so we need this convoluted approach
                singular = True

            if singular:
                msg = (
                    "KDE cannot be estimated (0 variance or perfect covariance). "
                    "Pass `warn_singular=False` to disable this warning."
                )
                if warn_singular:
                    warnings.warn(msg, UserWarning, stacklevel=3)
                continue

            # Transform the support grid back to the original scale
            ax = self._get_axes(sub_vars)
            _, inv_x = _get_transform_functions(ax, "x")
            _, inv_y = _get_transform_functions(ax, "y")
            support = inv_x(support[0]), inv_y(support[1])

            # Apply a scaling factor so that the integral over all subsets is 1
            if common_norm:
                density *= len(sub_data) / len(all_data)

            key = tuple(sub_vars.items())
            densities[key] = density
            supports[key] = support

        # Define a grid of iso-proportion levels
        if thresh is None:
            thresh = 0
        if isinstance(levels, Number):
            levels = np.linspace(thresh, 1, levels)
        else:
            if min(levels) < 0 or max(levels) > 1:
                raise ValueError("levels must be in [0, 1]")

        # Transform from iso-proportions to iso-densities
        if common_norm:
            common_levels = self._quantile_to_level(
                list(densities.values()), levels,
            )
            if draw_levels == None:
                draw_levels = {k: common_levels for k in densities}
        else:
            if draw_levels == None:
                draw_levels = {
                    k: self._quantile_to_level(d, levels)
                    for k, d in densities.items()
                }

        # Define the coloring of the contours
        if "hue" in self.variables:
            for param in ["cmap", "colors"]:
                if param in contour_kws:
                    msg = f"{param} parameter ignored when using hue mapping."
                    warnings.warn(msg, UserWarning)
                    contour_kws.pop(param)
        else:

            # Work out a default coloring of the contours
            coloring_given = set(contour_kws) & {"cmap", "colors"}
            if fill and not coloring_given:
                cmap = self._cmap_from_color(color)
                contour_kws["cmap"] = cmap
            if not fill and not coloring_given:
                contour_kws["colors"] = [color]

            # Use our internal colormap lookup
            cmap = contour_kws.pop("cmap", None)
            if isinstance(cmap, str):
                cmap = color_palette(cmap, as_cmap=True)
            if cmap is not None:
                contour_kws["cmap"] = cmap

        # Loop through the subsets again and plot the data
        for sub_vars, _ in self.iter_data("hue"):

            if "hue" in sub_vars:
                color = self._hue_map(sub_vars["hue"])
                if fill:
                    contour_kws["cmap"] = self._cmap_from_color(color)
                else:
                    contour_kws["colors"] = [color]

            ax = self._get_axes(sub_vars)

            # Choose the function to plot with
            # TODO could add a pcolormesh based option as well
            # Which would look something like element="raster"
            if fill:
                contour_func = ax.contourf
            else:
                contour_func = ax.contour

            key = tuple(sub_vars.items())
            if key not in densities:
                continue
            density = densities[key]
            xx, yy = supports[key]

            # Pop the label kwarg which is unused by contour_func (but warns)
            contour_kws.pop("label", None)

            cset = contour_func(
                xx, yy, density,
                levels=draw_levels[key],
                **contour_kws,
            )

            # Add a color bar representing the contour heights
            # Note: this shows iso densities, not iso proportions
            # See more notes in histplot about how this could be improved
            if cbar:
                cbar_kws = {} if cbar_kws is None else cbar_kws
                ax.figure.colorbar(cset, cbar_ax, ax, **cbar_kws)

        # --- Finalize the plot
        ax = self.ax if self.ax is not None else self.facets.axes.flat[0]
        self._add_axis_labels(ax)

        if "hue" in self.variables and legend:

            # TODO if possible, I would like to move the contour
            # intensity information into the legend too and label the
            # iso proportions rather than the raw density values

            artist_kws = {}
            if fill:
                artist = partial(mpl.patches.Patch)
            else:
                artist = partial(mpl.lines.Line2D, [], [])

            ax_obj = self.ax if self.ax is not None else self.facets
            self._add_legend(
                ax_obj, artist, fill, False, "layer", 1, artist_kws, {},
            )

    def plot_univariate_ecdf(self, estimate_kws, legend, **plot_kws):

        estimator = ECDF(**estimate_kws)

        # Set the draw style to step the right way for the data variable
        drawstyles = dict(x="steps-post", y="steps-pre")
        plot_kws["drawstyle"] = drawstyles[self.data_variable]

        # Loop through the subsets, transform and plot the data
        for sub_vars, sub_data in self.iter_data(
            "hue", reverse=True, from_comp_data=True,
        ):

            # Compute the ECDF
            if sub_data.empty:
                continue

            observations = sub_data[self.data_variable]
            weights = sub_data.get("weights", None)
            stat, vals = estimator(observations, weights=weights)

            # Assign attributes based on semantic mapping
            artist_kws = plot_kws.copy()
            if "hue" in self.variables:
                artist_kws["color"] = self._hue_map(sub_vars["hue"])

            # Return the data variable to the linear domain
            ax = self._get_axes(sub_vars)
            _, inv = _get_transform_functions(ax, self.data_variable)
            vals = inv(vals)

            # Manually set the minimum value on a "log" scale
            if isinstance(inv.__self__, mpl.scale.LogTransform):
                vals[0] = -np.inf

            # Work out the orientation of the plot
            if self.data_variable == "x":
                plot_args = vals, stat
                stat_variable = "y"
            else:
                plot_args = stat, vals
                stat_variable = "x"

            if estimator.stat == "count":
                top_edge = len(observations)
            else:
                top_edge = 1

            # Draw the line for this subset
            artist, = ax.plot(*plot_args, **artist_kws)
            sticky_edges = getattr(artist.sticky_edges, stat_variable)
            sticky_edges[:] = 0, top_edge

        # --- Finalize the plot ----
        ax = self.ax if self.ax is not None else self.facets.axes.flat[0]
        stat = estimator.stat.capitalize()
        default_x = default_y = ""
        if self.data_variable == "x":
            default_y = stat
        if self.data_variable == "y":
            default_x = stat
        self._add_axis_labels(ax, default_x, default_y)

        if "hue" in self.variables and legend:
            artist = partial(mpl.lines.Line2D, [], [])
            alpha = plot_kws.get("alpha", 1)
            ax_obj = self.ax if self.ax is not None else self.facets
            self._add_legend(
                ax_obj, artist, False, False, None, alpha, plot_kws, {},
            )

# %%
