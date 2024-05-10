# %%
# !grep '"y":' data/pins.txt | cut -f2 -d":" | sed "s/ //g" | sed "s/,//g" > data/y.txt
# !grep '"x":' data/pins.txt | cut -f2 -d":" | sed "s/ //g" | sed "s/,//g" > data/x.txt
# !paste data/x.txt data/y.txt -d "," > data/pins.csv
import pandas as pd
import cooler
import cooltools
import bioframe as bf
import os
os.chdir('/home/fox/projects/ecoli_microc')

# %% hairpins
df = pd.read_excel('data/hairpins_Gavrilov.xlsx')
hairpins_df = pd.DataFrame({
    'chrom1': 'NC_000913.3',
    'start1': df.wt_middle//25*25,
    'end1': (df.wt_middle//25+1)*25,
    'chrom2': 'NC_000913.3',
    'start2': df.wt_middle//25*25,
    'end2': (df.wt_middle//25+1)*25,
})
# hairpins_df.to_csv('data/hairpins_25.bedpe', sep='\t', index=False, header=False)
hairpins_df.head()
# %% chromosight
# chromosight quantify --pattern hairpins --threads=4 data/hairpins_25.bedpe data/wt.combined.MG1655.mapq30.25.mcool::/resolutions/25 hairping.AGav
# !chromosight detect --pattern hairpins --threads=4 --min-separation=200 data/wt.combined.MG1655.mapq30.25.mcool::/resolutions/25 data/hairpins_25 
# %% download Regulon db
# !bash scripts/chipseq_mine.sh

# %% create bins
resolution = 500
clr = cooler.Cooler('data/wt.combined.MG1655.mapq30.25.mcool::/resolutions/' + str(resolution))
df_chromsizes = clr.chromsizes
bins = cooler.binnify(df_chromsizes, 25)
bins.chrom = bins.chrom.astype(str)
fasta_records = bf.load_fasta('data/genome.fasta')
# %%
chipseq_df = pd.read_csv('data/chipseq/RHTECOLIBSD00243.csv', index_col=0)
chipseq_df['chrom'] = 'NC_000913.3'
chipseq_df['start'] = chipseq_df.Peak_start
chipseq_df['end'] = chipseq_df.Peak_end

# %%
chipseq_df_cov = bf.coverage(bins, chipseq_df)
# chipseq_df_cov['frac'] = chipseq_df_cov.coverage / 25

bf.to_bigwig(df=chipseq_df_cov, chromsizes=df_chromsizes,
                        outpath='data/%s.25.bw' % chipseq_df['TF_name*'][0],
                        path_to_binary='/home/fox/micromamba/envs/omics_env/bin/bedGraphToBigWig')
# %%
tmp = pyBigWig.open('data/%s.25.bw' % chipseq_df['TF_name*'][0])
tmp.stats("1", 0, 3)
# %% deeptools
import sys
from sys import exit
from deeptools.parserCommon import writableFile, numberOfProcessors
from deeptools import parserCommon
from deeptools import heatmapper
import deeptools.computeMatrixOperations as cmo

parameters = {'upstream': 1000,
            'downstream': 1000,
            'body': 25,
            'bin size': 25,
            'ref point': None,
            'verbose': False,
            'bin avg type': 'median',
            'missing data as zero': False,
            'min threshold': 0,
            'max threshold': 25,
            'scale': 1,
            'skip zeros': False,
            'nan after end': False,
            'proc number': 4,
            'sort regions': "descend",
            'sort using': "median",
            'unscaled 5 prime': 0,
            'unscaled 3 prime': 0
            }

hm = heatmapper.heatmapper()

scores_file_list = ['data/%s.25.bw' % chipseq_df['TF_name*'][0]]
hm.computeMatrix(scores_file_list, ["data/hairpins_25.bed"], parameters, blackListFileName=None, verbose=False)

# %%
if args.sortRegions not in ['no', 'keep']:
    sortUsingSamples = []
    if args.sortUsingSamples is not None:
        for i in args.sortUsingSamples:
            if (i > 0 and i <= hm.matrix.get_num_samples()):
                sortUsingSamples.append(i - 1)
            else:
                exit("The value {0} for --sortUsingSamples is not valid. Only values from 1 to {1} are allowed.".format(args.sortUsingSamples, hm.matrix.get_num_samples()))
        print('Samples used for ordering within each group: ', sortUsingSamples)

    hm.matrix.sort_groups(sort_using=args.sortUsing, sort_method=args.sortRegions, sample_list=sortUsingSamples)
elif args.sortRegions == 'keep':
    hm.parameters['group_labels'] = hm.matrix.group_labels
    hm.parameters["group_boundaries"] = hm.matrix.group_boundaries
    cmo.sortMatrix(hm, args.regionsFileName, args.transcriptID, args.transcript_id_designator, verbose=not args.quiet)

hm.save_matrix(args.outFileName)

# %%
def plot_around_loop(
    path_bw,
    plot_name,
    nbins=30,
    resolution=2000,
    chrom_file=df_chromsizes,
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
    flip_negative_strand=False,
    return_matrix=False,
    pic_path="results/pics/paper/"
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
        1, len(bed_list)
    )  # gridspec with two adjacent horizontal cells
    fig = plt.figure(figsize=[3 * len(bed_list), 9])

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

    fig.savefig(pic_path + plot_name, dpi=100, bbox_inches="tight")
    plt.show()

def wrapper_stackup(
    i,
    pathes,
    nbins,
    window=10000,
    resolution=2000,
    chrom_file=df_chromsizes,
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
        neg_index = df_bed.loc[df_bed.strand == "-", :].index.tolist()
        s[neg_index,] = np.flip(s[neg_index,], axis=1)
    # s[np.isnan(s)] = 0

    return s

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

#from pybedtools import BedTool

import cooler
import cooltools.api.snipping as clsnip
from cooltools.lib.numutils import LazyToeplitz
import cooltools.lib.plotting
#from coolpuppy import coolpup
# from plotpuppy import plotpup

import bioframe
from bioframe import count_overlaps

import bbi

import pyBigWig
plot_around_loop(
    path_bw='data/CpxR.25.bw',
    plot_name='CpxR',
    nbins=81,
    resolution=25,
    chrom_file=df_chromsizes,
    window=1000,
    mode="median",
    ymin=0,
    ymax=25,
    vmin=0,
    vmax=25,
    norm=False,
    fill=True,
    how_far_from_edge=40,
    bed_list=[
        "data/hairpins_25.bedpe",
    ],
    pic_path='results/'
)
# %%
