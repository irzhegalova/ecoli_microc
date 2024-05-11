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
hairpins_df_Pseudobedpe = pd.DataFrame({
    'chrom1': 'NC_000913.3',
    'start1': df.wt_middle//25*25,
    'end1': (df.wt_middle//25+1)*25,
    'chrom2': 'NC_000913.3',
    'start2': df.wt_middle//25*25,
    'end2': (df.wt_middle//25+1)*25,
})

hairpins_df = pd.DataFrame({
    'chrom': 'NC_000913.3',
    'start': df.wt_middle//25*25,
    'end': (df.wt_middle//25+1)*25,
})

hairpins_df_bedpe = pd.DataFrame({
    'chrom1': 'NC_000913.3',
    'start1': df.wt_left//25*25,
    'end1': (df.wt_left//25+1)*25,
    'chrom2': 'NC_000913.3',
    'start2': df.wt_right//25*25,
    'end2': (df.wt_right//25+1)*25,
})
hairpins_df_bedpe.to_csv('data/hairpins_25.bedpe', sep='\t', index=False, header=False)


hairpins_df_Pseudobedpe.to_csv('data/hairpins_25.pseudo.bedpe', sep='\t', index=False, header=False)
# hairpins_df.to_csv('data/hairpins_25.bed', sep='\t', index=False, header=False)
hairpins_df.head()

# %% create bins
resolution = 100
clr = cooler.Cooler('data/wt.combined.MG1655.mapq30.25.mcool::/resolutions/' + str(resolution))
df_chromsizes = clr.chromsizes
bins = cooler.binnify(df_chromsizes, 25)
bins.chrom = bins.chrom.astype(str)
fasta_records = bf.load_fasta('data/genome.fasta')
view_df_chromsizes = clr.chromsizes.reset_index()
view_df_chromsizes.columns = ['chrom', 'end']
view_df_chromsizes['start'] = 0
view_df_chromsizes['name'] = 'NC_000913.3'
view_df_chromsizes = view_df_chromsizes[['chrom', 'start', 'end', 'name']]

# %%
resolution = 100
flank = 100
stack = cooltools.pileup(clr, hairpins_df, view_df=view_df_chromsizes, 
flank=flank, nproc=2)
# Mirror reflect snippets when the feature is on the opposite strand
# mask = np.array(sites.strand == '-', dtype=bool)
# stack[:, :, mask] = stack[::-1, ::-1, mask]

# %%
# Aggregate. Note that some pixels might be converted to NaNs after IC, thus we aggregate by nanmean: 
mtx = np.nanmean(stack, axis=2)

# Load colormap with large number of distinguishable intermediary tones,
# The "fall" colormap in cooltools is exactly for this purpose.
# After this step, you can use "fall" as cmap parameter in matplotlib:
import cooltools.lib.plotting

plt.imshow(
    np.log10(mtx),
    vmin = -3,
    vmax = -1,
    cmap='fall',
    interpolation='none')

plt.colorbar(label = 'log10 mean ICed Hi-C')
ticks_pixels = np.linspace(0, flank*2//resolution,5)
ticks_kbp = ((ticks_pixels-ticks_pixels[-1]/2)*resolution//1000).astype(int)
plt.xticks(ticks_pixels, ticks_kbp)
plt.yticks(ticks_pixels, ticks_kbp)
plt.xlabel('relative position, kbp')
plt.ylabel('relative position, kbp')

plt.show()


# %% chromosight
# chromosight quantify --pattern hairpins --threads=4 data/hairpins_25.bedpe data/wt.combined.MG1655.mapq30.25.mcool::/resolutions/25 hairping.AGav
# !chromosight detect --pattern hairpins --threads=4 --min-separation=200 data/wt.combined.MG1655.mapq30.25.mcool::/resolutions/25 data/hairpins_25 

# %% download Regulon db
# !bash scripts/chipseq_mine.sh

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
from scipy.stats import zscore
redC_nascent_df = bf.read_table('data/wt_merged.rna1.10k.bg', schema='bedGraph')
redC_nascent_df['chrom'] = 'NC_000913.3'
redC_nascent_df['z_value'] = zscore(redC_nascent_df.value.tolist())
bf.to_bigwig(df=redC_nascent_df.loc[:,['chrom', 'start', 'end', 'value']], chromsizes=df_chromsizes,
            outpath='data/wt_merged.rna1.10k.chromName.bw',
            path_to_binary='/home/fox/micromamba/envs/omics_env/bin/bedGraphToBigWig')

bf.to_bigwig(df=redC_nascent_df.loc[:,['chrom', 'start', 'end', 'z_value']], chromsizes=df_chromsizes,
            outpath='data/wt_merged.rna1.10k.zscore.bw',
            path_to_binary='/home/fox/micromamba/envs/omics_env/bin/bedGraphToBigWig')
# %%
redC_total_df = bf.read_table('data/wt_merged.rna1.total.bg', schema='bedGraph')
redC_total_df['chrom'] = 'NC_000913.3'
redC_total_df['z_value'] = zscore(redC_total_df.value.tolist())
bf.to_bigwig(df=redC_total_df.loc[:,['chrom', 'start', 'end', 'value']], chromsizes=df_chromsizes,
            outpath='data/wt_merged.rna1.total.chromName.bw',
            path_to_binary='/home/fox/micromamba/envs/omics_env/bin/bedGraphToBigWig')

bf.to_bigwig(df=redC_total_df.loc[:,['chrom', 'start', 'end', 'z_value']], chromsizes=df_chromsizes,
            outpath='data/wt_merged.rna1.total.zscore.bw',
            path_to_binary='/home/fox/micromamba/envs/omics_env/bin/bedGraphToBigWig')
# %%
nbins=81
plot_around_loop(
        path_bw='data/wt_merged.rna1.10k.zscore.bw',
        plot_name="RedC.10k.zscore.pdf",
        nbins=61,
        resolution=200,
        chrom_file=df_chromsizes,
        window=5000,
        mode="median",
        ymin=-2,
        ymax=2,
        vmin=-3,
        vmax=3,
        norm=False,
        fill=True,
        how_far_from_edge=25,
        bed_list=[
            "data/hairpins_25.bedpe",
        ],
        pic_path='results/',
        return_matrix=False
    )