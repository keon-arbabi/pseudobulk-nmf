import polars as pl 
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import os 
from utils import savefig

def plot_volcano(df, significance_column='FDR', threshold=0.05, 
                 label_top_genes=True, num_top_genes=30, min_distance=0.1,
                 colors=('blue', 'grey', 'red'), alpha=1, size=6,
                 plot_as_pdf=False, plot_directory=None):
    
    import matplotlib.patches as mpatches
    from adjustText import adjust_text  
    from scipy.spatial.distance import pdist, squareform
    
    df = df.to_pandas()
    df['logP'] = -np.log10(df['P'])
    df['rank'] = abs(df['logFC']) * df['logP']
    df['color'] = colors[1]
    sig = df[significance_column] < threshold
    df.loc[sig & (df['logFC'] > 0), 'color'] = colors[2]
    df.loc[sig & (df['logFC'] < 0), 'color'] = colors[0]

    plt.figure(figsize=(6, 8))
    plt.scatter(df['logFC'], df['logP'], c=df['color'], alpha=alpha, s=size)
    if label_top_genes:
        for direction in ['up', 'down']:
            top_genes = \
                df[(sig & ((df['logFC'] > 0) if direction == 'up' 
                    else (df['logFC'] < 0)))].nlargest(num_top_genes, 'rank')
            if not top_genes.empty:
                distances = squareform(pdist(top_genes[['logFC', 'logP']]))
                np.fill_diagonal(distances, np.inf)
                filter_idx = [np.min(distances[i]) > min_distance 
                              for i in range(len(distances))]
                filtered_genes = top_genes.iloc[filter_idx]
                for _, gene in filtered_genes.iterrows():
                    plt.text(gene['logFC'], gene['logP'], gene['gene'],
                             fontweight='semibold', fontsize=8)
    plt.title(f"{df['cell_type'].unique()[0]}")
    plt.xlabel(r'$\log_2(Fold Change)$')
    plt.ylabel(r'$-\log_{10}(P-value)$')
    legend_patches = [
        mpatches.Patch(color=color, label=label) 
        for color, label in zip(colors, ['down', '', 'up'])]
    plt.legend(handles=legend_patches, 
               title=f"{significance_column} < {threshold}", loc='best')
    if plot_directory is not None:
        filename = (
            f"{df['cell_type'].unique()[0].replace('/', '-')}"
            f".{'pdf' if plot_as_pdf else 'png'}"
        )
        plot_file = os.path.join(plot_directory, filename)
        plt.savefig(plot_file)
        

def normalize_matrix(mat, norm_method):
    
    from scipy.stats import rankdata
    from utils import inverse_normal_transform
    
    if not np.isfinite(mat).all():
        raise ValueError('Matrix contains NaN, infinity, or missing values')
    if np.any(np.ptp(mat, axis=1) == 0):
        raise ValueError("Matrix contains rows with constant values")
    if norm_method == 'median' or norm_method == 'mean':
        shift = abs(np.min(mat))
        mat += shift
        if norm_method == 'median':
            norm_factor = np.median(mat, axis=1)[:, None]
        else:  
            norm_factor = np.mean(mat, axis=1)[:, None]
        mat /= norm_factor
    elif norm_method == 'minmax':
        mat -= np.min(mat, axis=1)[:, None]
        mat /= np.max(mat, axis=1)[:, None]
    elif norm_method == 'quantile':
        mat = np.apply_along_axis(rankdata, 1, mat) - 1
        mat /= (mat.shape[1] - 1)  
    elif norm_method == 'rint':
        mat = np.apply_along_axis(inverse_normal_transform, 1, mat)
        mat += abs(np.min(mat))
    else:
        raise ValueError(f"Unknown method: {norm_method}") 
    return mat
