import os
import polars as pl, pandas as pd, numpy as np
import matplotlib.pylab as plt
from utils import savefig

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

# Plotting #####################################################################

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
        savefig(plot_file)
        
def plot_k_MSE(axes, idx, cell_type, MSE_trial,
               k_1se_trial, MSE_final, best_beta):
    row, col = divmod(idx, 2)
    ax = axes[row, col]
    for (current_cell_type, current_beta), MSE in MSE_trial.items():
        if current_cell_type == cell_type:
            mean_MSE = MSE.groupby('k').mean()
            k_1se = k_1se_trial[current_cell_type, current_beta]
            #alpha = 0.4 + ((best_beta - 1e-6) * (0.6 - 0.2) / (1e-1 - 1e-6))
            ax.plot(mean_MSE.index, mean_MSE.values, color='black', alpha=0.4)
            ax.scatter(k_1se, mean_MSE[k_1se], color='black', s=16, alpha=0.4)
    mean_MSE = MSE_final.groupby('k').mean()
    k_final = k_1se_trial[cell_type, best_beta]
    ax.plot(mean_MSE.index, mean_MSE.values, color='red')
    ax.scatter(k_final, mean_MSE[k_final], color='red', s=50)
    ax.set_xticks(ticks=mean_MSE.index)
    ax.set_yscale('log')
    ax.set_title(rf'$\mathbf{{{cell_type}}}$'
                + "\nMSE across Optuna trials\n"
                + f"Best k: {k_final}, "
                + f"Best beta: {best_beta:.2g}")
    ax.set_xlabel('k')
    ax.set_ylabel('Mean MSE')
    
def plot_k_stats(snmf, kmax, cell_type, 
                 plot_as_pdf=False, plot_directory=None):
    
    from matplotlib.ticker import MaxNLocator
    
    ks = range(1, kmax)
    summary = snmf.estimate_rank(
        rank_range=ks, n_run=1, 
        what=['sparseness','rss','evar', 'dispersion'])

    spar = [summary[rank]['sparseness'] for rank in ks]
    rss = [summary[rank]['rss'] for rank in ks]
    evar = [summary[rank]['evar'] for rank in ks]    
    disp = [summary[rank]['dispersion'] for rank in ks]
    spar_w, spar_h = zip(*spar)
    # coph = [summary[rank]['cophenetic'] for rank in ks]   
            
    fig, axs = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    axs[0, 0].plot(ks, rss, 'o-', color='k', label='RSS', linewidth=2)
    axs[0, 0].set_title('RSS')
    axs[0, 1].plot(ks, disp, 'o-', color='k', label='Dispersion', linewidth=2)
    axs[0, 1].set_title('Dispersion')
    axs[0, 2].plot(ks, evar, 'o-', color='k', label='Explained variance', 
                   linewidth=2)
    axs[0, 2].set_title('Explained variance')
    axs[1, 0].plot(ks, spar_w, 'o-', color='k', label='Sparsity (Basis)', 
                   linewidth=2)
    axs[1, 0].set_title('Sparsity (Basis)')
    axs[1, 1].plot(ks, spar_h, 'o-', color='k', label='Sparsity (Mixture)', 
                   linewidth=2)
    axs[1, 1].set_title('Sparsity (Mixture)')
    axs[1, 2].axis('off')
    # axs[2, 2].plot(ks, coph, 'o-', label='Cophenetic correlation', linewidth=2)
    # axs[2, 2].set_title('Cophenetic correlation')
    for ax in axs.flat:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if plot_directory is not None:
        filename = (
            f"{cell_type.replace('/', '-')}"
            f".{'pdf' if plot_as_pdf else 'png'}"
        )
        plot_file = os.path.join(plot_directory, filename)
        savefig(plot_file)

