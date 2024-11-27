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



import textwrap


def wrap_labels(ax, width, break_long_words=False):
    labels = []
    for label in ax.get_xticklabels():
        text = label.get_text()
        labels.append(
            textwrap.fill(text, width=width, break_long_words=break_long_words)
        )
    ax.set_xticklabels(labels, rotation=0)

