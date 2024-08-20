import os, sys
import polars as pl, numpy as np, pandas as pd
sys.path.append('/home/karbabi/projects/def-wainberg/karbabi/utils')
from single_cell import SingleCell as SingleCell
from single_cell_old import SingleCell as SingleCellOld
from utils import debug, savefig, Timer, use_font
debug()

# Load p400 for a 10% random subsample of inhibitory neurons
# biorxiv.org/content/10.1101/2023.03.07.531493v1.full

with Timer('Loading p400'):
    p400_file = \
        os.path.expanduser('~wainberg/p400_Inhibitory_subsampled.h5ad')
    p400 = SingleCell(p400_file)
    p400_old = SingleCellOld(p400_file)

# Load atlas: mini SEA-AD, subset to inhibitory
# portal.brain-map.org/atlases-and-data/rnaseq/human-mtg-10x_sea-ad

with Timer('Loading mini SEA-AD'):
    mini_SEAAD_file = \
        os.path.expanduser('~wainberg/mini_SEA_AD_Inhibitory.h5ad')
    mini_SEAAD = SingleCell(mini_SEAAD_file)
    mini_SEAAD_old = SingleCellOld(mini_SEAAD_file)

# Quality-control (QC) both datasets; subset to confident inhibitory cells for
# p400 but confident inhibitory SUBTYPE cells for mini SEA-AD

with Timer('p400 QC'):
    p400 = p400.qc(
        custom_filter= pl.col('cell.type.prob').ge(0.9) & 
        pl.col('is.doublet.df').not_() &
        pl.col.projid.is_not_null(),
        MALAT1_filter=False)

    p400_old = p400_old.qc(
        cell_type_confidence_column='cell.type.prob',
        doublet_column='is.doublet.df',
        custom_filter=pl.col.projid.is_not_null()) 

with Timer('Mini SEA-AD QC'):
    mini_SEAAD = mini_SEAAD.qc(
        custom_filter=pl.col('subclass_confidence').ge(0.9),
        allow_float=True) 
    
    mini_SEAAD_old = mini_SEAAD_old.qc(
        cell_type_confidence_column='subclass_confidence',
        doublet_column=None,
        allow_float=True) 

assert p400.obs.equals(p400_old.obs)
assert mini_SEAAD.obs.equals(mini_SEAAD_old.obs)

# Find highly variable genes

with Timer('hvg'):
    p400, mini_SEAAD = p400.hvg(mini_SEAAD, allow_float=True)
    p400_old, mini_SEAAD_old = p400_old.hvg(mini_SEAAD_old, allow_float=True)

f'Overlapping hvg genes: {
    p400.var.filter(pl.col.highly_variable)['_index'].is_in(
        p400_old.var.filter(pl.col.highly_variable)['_index']).sum()} / 2000'
# Overlapping hvg genes: 2000 / 2000

p400.var.join(p400_old.var, on='_index', how='inner')\
    .filter(pl.col.highly_variable)\
    .select(pl.corr('highly_variable_rank', 'highly_variable_rank_right',
                    method='spearman')).item()
# 1.0

# Normalize

with Timer('normalize'):
    p400 = p400.normalize()
    mini_SEAAD = mini_SEAAD.normalize(allow_float=True)

    p400_old = p400_old.normalize()
    mini_SEAAD_old = mini_SEAAD_old.normalize(allow_float=True)

assert (p400.X != p400_old.X).sum() == 0

# Run PCA

with Timer('PCA'):
    p400, mini_SEAAD = p400.PCA(mini_SEAAD)
    p400_old, mini_SEAAD_old = p400_old.PCA(mini_SEAAD_old)

assert np.allclose(p400.obsm['PCs'], p400_old.obsm['PCs'], equal_nan=True)

# Harmonize the principal components between the two datasets with Harmony

with Timer('Harmony'):
    p400, mini_SEAAD = p400.harmonize(mini_SEAAD)
    p400_old, mini_SEAAD_old = p400_old.harmonize(mini_SEAAD_old)

assert np.allclose(p400.obsm['Harmony_PCs'], p400_old.obsm['Harmony_PCs'], 
                   equal_nan=True)
# AssertionError

corrs = [pd.Series(p400.obsm['Harmony_PCs'][:,i]).corr(
         pd.Series(p400_old.obsm['Harmony_PCs'][:,i])).round(4)
         for i in range(p400.obsm['Harmony_PCs'].shape[1])]
print(f"PC correlations: {corrs}")
# PC correlations: [
# 1.0, 0.9999, 1.0, 0.9999, 0.9997, 0.9999, 0.9998, 0.9999, 0.9999, 0.9998,
# 0.9999, 0.9999, 0.9996, 0.9999, 0.9998, 0.9996, 0.9998, 0.9996, 0.9996, 
# 0.9998, 0.9996, 0.9998, 0.9997, 0.9997, 0.9997, 0.9998, 0.9995, 0.9996, 
# 0.9996, 0.9998, 0.9997, 0.9996, 0.9998, 0.9998, 0.9995, 0.9982, 0.9997, 
# 0.9996, 0.9994, 0.9997, 0.9998, 0.9998, 0.9996, 0.9996, 0.9999, 0.9998, 
# 0.9998, 0.9997, 0.9998, 0.9997]

# Transfer cell-type labels from mini SEA-AD to the p400

with Timer('Label transfer'):
    p400 = p400.label_transfer_from(mini_SEAAD, 'subclass_label')  

    p400_old = p400_old\
        .label_transfer_from(mini_SEAAD_old, 'subclass_label')\
        .rename_obs({'subclass_label': 'cell_type',
                     'subclass_label_confidence': 'cell_type_confidence'})

# Create a PaCMAP plot of the transferred labels

with Timer('p400 label transfer PaCMAP plot'):
    p400.plot_embedding('cell_type', 'p400_transferred.pdf', label=True)

# Create a confusion matrix plot of the transferred labels

import matplotlib.pyplot as plt 
import seaborn as sns
use_font('Helvetica Neue')

def confusion_matrix_plot(sc, original_labels_column,
                          transferred_labels_column, save_to):
    confusion_matrix = sc.obs\
        .select(original_labels_column, transferred_labels_column)\
        .to_pandas()\
        .groupby([original_labels_column, transferred_labels_column],
                 observed=True)\
        .size()\
        .unstack(fill_value=0)\
        .sort_index(axis=1)\
        .assign(broad_cell_type=lambda df: df.index.str.split('.').str[0],
                cell_type_cluster=lambda df: df.index.str.split('.').str[1]
                .astype('Int64').fillna(0))\
        .sort_values(['broad_cell_type', 'cell_type_cluster'])\
        .drop(['broad_cell_type', 'cell_type_cluster'], axis=1)
    print(confusion_matrix)
    ax = sns.heatmap(confusion_matrix.T.div(confusion_matrix.T.sum()),
                     xticklabels=1, yticklabels=1, rasterized=True,
                     square=True, linewidths=0.5, cmap='rocket_r',
                     cbar_kws=dict(pad=0.01), vmin=0, vmax=1)
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    cbar.set_ticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
    w, h = plt.gcf().get_size_inches()
    plt.gcf().set_size_inches(3.5 * w, h)
    savefig(save_to)


confusion_matrix_plot(p400, 'state', 'cell_type', 'confusion_matrix.pdf')
