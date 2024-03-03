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

    if norm_method == 'median':
        mat += abs(np.min(mat))
        mat /= np.median(mat, axis=1)[:, None]
    elif norm_method == 'mean':
        mat /= np.mean(mat, axis=1)[:, None]
    elif norm_method == 'min-max':
        mat -= np.min(mat, axis=1)[:, None]
        mat /= np.max(mat, axis=1)[:, None]
    elif norm_method == 'quantile':
        mat = np.apply_along_axis(rankdata, 1, mat)
    elif norm_method == 'rint':
        mat = np.apply_along_axis(inverse_normal_transform, 1, mat)
        mat += abs(np.min(mat) + 1)
    else:
        raise ValueError(f"Unknown method: {norm_method}")
    return mat

def plot_A_heatmap(mat, obs, var, cell_type, rownames, filename):

    from ryp import r, to_r    

    to_r(mat, 'mat', format='df',
        rownames=rownames,
        colnames=obs['ID'])
    to_r(cell_type, 'cell_type')
    meta = obs.select(sorted([
        'num_cells', 'sex', 'Cdx', 'braaksc', 'ceradsc', 'pmi', 'niareagansc',
        'apoe_genotype', 'tomm40_hap','age_death', 'age_first_ad_dx', 'cogdx',
        'ad_reagan', 'gpath', 'amyloid', 'hspath_typ', 'dlbdx', 'tangles',
        'tdp_st4', 'arteriol_scler', 'caa_4gp', 'cvda_4gp2','ci_num2_gct',
        'ci_num2_mct', 'tot_cog_res']))
    to_r(meta, 'meta', format='df', rownames=obs['ID'])
    
    r('''
    suppressPackageStartupMessages({
            library(ComplexHeatmap)
            library(circlize)
            library(seriation)
        })
        row_order = get_order(seriate(dist(mat), method = "OLO"))
        col_order = get_order(seriate(dist(t(mat)), method = "OLO"))
        ht = HeatmapAnnotation(
            df = meta,
            simple_anno_size = unit(0.15, "cm"),
            annotation_name_gp = gpar(fontsize = 5),
            show_legend = FALSE)     
        # hb = HeatmapAnnotation(
        #     df = t(H),
        #     simple_anno_size = unit(0.3, "cm"),
        #     annotation_name_gp = gpar(fontsize = 8),
        #     show_legend = FALSE)         
        # hr = rowAnnotation(
        #     df = W,
        #     simple_anno_size = unit(0.3, "cm"),
        #     annotation_name_gp = gpar(fontsize = 8),
        #     show_legend = FALSE)
        col_fun = colorRamp2(quantile(mat, probs = c(0.00, 1.00)), 
            hcl_palette = "Batlow", reverse = TRUE)
        file_name = paste0("figures/explore/p400/", cell_type,
                            "_rint.png")
        png(file = file_name, width=7, height=7, units="in", res=1200)
        h = Heatmap(
            mat,
            row_order = row_order,
            column_order = col_order,
            cluster_rows = F,
            cluster_columns = F,
            show_row_names = FALSE,
            show_column_names = FALSE,
            top_annotation = ht,
            # bottom_annotation = hb,
            # left_annotation = hr,
            col = col_fun,
            name = paste0('p400', "\n", cell_type),
            show_heatmap_legend = TRUE
        )
        draw(h)
        dev.off()
    ''')