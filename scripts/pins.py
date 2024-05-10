# %%
# !grep '"y":' data/pins.txt | cut -f2 -d":" | sed "s/ //g" | sed "s/,//g" > data/y.txt
# !grep '"x":' data/pins.txt | cut -f2 -d":" | sed "s/ //g" | sed "s/,//g" > data/x.txt
# !paste data/x.txt data/y.txt -d "," > data/pins.csv
import pandas as pd
import cooler
import cooltools
import bioframe as bf
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import bootstrap
import seaborn as sns
os.chdir('/home/fox/projects/ecoli_microc')
from textwrap import wrap
from matplotlib import font_manager

font_dirs = ["/usr/share/fonts/arial/"]  # The path to the custom font file.
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)
    

# colors
# Create an array with the colors you want to use
susielu = ["#b84873", "#6dbc5f","#5a3789","#bdab3d","#6d80d8","#bd7635","#bf68b8","#46c19a","#ba4b41","#71883b"] # Set your custom color palette
# susielu_pal = sns.set_palette(sns.color_palette(susielu))
susielu_greyscale = ["#636363", "#a4a4a4","#444444", "#a7a7a7", "#828282",] # Set your custom color palette
# susielu_greyscale_pal = sns.set_palette(sns.color_palette(susielu_greyscale))
susielu_accent = ["#636363", "#b84873", "#a4a4a4","#444444", "#a7a7a7", "#828282",] 
sns.set_theme(context="paper", style='whitegrid', palette=susielu, font="Arial")
cm = 1/2.54  # centimeters in inches


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

# %% 337 - 350, 360-364 bad (no coordinates)
for id in range(363, 364):
    chipseq_df = pd.read_csv('data/chipseq/RHTECOLIBSD00%s.csv' % id, index_col=0)
    chipseq_df['chrom'] = 'NC_000913.3'
    chipseq_df['start'] = chipseq_df.Peak_start
    chipseq_df['end'] = chipseq_df.Peak_end
    chipseq_df_cov = bf.coverage(bins, chipseq_df)
    # chipseq_df_cov['frac'] = chipseq_df_cov.coverage / 25

    bf.to_bigwig(df=chipseq_df_cov, chromsizes=df_chromsizes,
                            outpath='data/bw/%s.25.bw' % chipseq_df['TF_name*'][0],
                            path_to_binary='/home/fox/micromamba/envs/omics_env/bin/bedGraphToBigWig')

# %% tried to use deeptools api
# !python scripts/deeptools.py
# %%
import sys
sys.path.append('/home/fox/projects/dicty/hic_loop_study/scripts/functions/modules/')
from functions import plot_around_loop, wrapper_stackup
# %%
nbins=81
i=0
j=0
fig, axs = plt.subplots(10, 10,figsize=(20,20), 
sharex='all', sharey='all')
for id in range(242, 337): #337
    chipseq_df = pd.read_csv('data/chipseq/RHTECOLIBSD00%s.csv' % id, index_col=0)
    s_list, order = plot_around_loop(
        path_bw='data/bw/%s.25.bw' % chipseq_df['TF_name*'][0],
        plot_name=chipseq_df['TF_name*'][0],
        nbins=81,
        resolution=25,
        chrom_file=df_chromsizes,
        window=1000,
        mode="mean",
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
        pic_path='results/',
        return_matrix=True
    )
    upper_median = np.nanmedian(s_list[0][order, :], axis=0)
    # tmp = sliding_window(seq=np.mean(s_list[i][order, :], axis=0), half_window_size=2)
    axs[i,j].set_title(chipseq_df['TF_name*'][0])
    axs[i,j].plot(upper_median)
    axs[i,j].set(xticklabels=[])
    axs[i,j].set_ylim([-1, 10])
    # err_style
    to_bootstrap = (s_list[0][order, :],)
    rng = np.random.default_rng()
    res = bootstrap(
    to_bootstrap,
    np.nanstd,
    n_resamples=100,
    confidence_level=0.9,  # vectorized=True,
    axis=0,
    random_state=rng,
    method="basic",
    )
    dyfit = res.standard_error

    axs[i,j].fill_between(
        range(nbins),
        upper_median - dyfit,
        upper_median + dyfit,
        color="blue",
        alpha=0.3,
    )
    i += 1
    if i == 10:
        i = 0
        j += 1
fig.savefig('results/chipseq.map_mean.pdf', dpi=100, bbox_inches='tight')
fig.show()
fig.clf()
# %%
