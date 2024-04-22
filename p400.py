import os
import polars as pl, numpy as np

os.chdir('projects/def-wainberg/karbabi/single-cell-nmf')
from utils import Timer, print_df, get_coding_genes, debug, \
    SingleCell, Pseudobulk

# debug(third_party=True)

# Columbia p400 ################################################################

with Timer('Load single-cell data'):
    sc = SingleCell('data/single-cell/p400/p400_qced_shareable.h5ad')
    
rosmap_pheno = \
    pl.read_csv(
        'data/single-cell/p400/dataset_978_basic_04-21-2023_with_pmAD.csv',
        dtypes={'projid': pl.String})\
    .unique(subset='projid')\
    .drop([col for col in sc.obs.columns if col != 'projid'])

with Timer('QC single-cell data and pseudobulk'):
    pb = sc\
        .qc(cell_type_confidence_column='cell.type.prob',
            doublet_confidence_column='doublet.score',
            custom_filter=pl.col.projid.is_not_null())\
        .with_columns_obs(
            subset=pl.col.subset.replace({'CUX2+': 'Excitatory'}),
            projid=pl.col.projid.cast(pl.String))\
        .pseudobulk(ID_column='projid', 
                    cell_type_column='subset',
                    additional_obs=rosmap_pheno)
    pb.save('data/pseudobulk/p400')

with Timer('QC pseudobulk data'):
    pb = Pseudobulk('data/pseudobulk/p400')
    pb = pb\
        .filter_var(pl.col._index.is_in(get_coding_genes()['gene']))\
        .with_columns_obs(
            apoe4_dosage=pl.col.apoe_genotype.cast(pl.String)\
                .str.count_matches('4').fill_null(strategy='mean'),
            pmi=pl.col.pmi.fill_null(strategy='mean'))\
        .qc(case_control_column='pmAD', 
            max_standard_deviations=2, 
            custom_filter=pl.col.pmAD.is_not_null())
    pb.save('data/pseudobulk/p400_qcd')
        
with Timer('Differential expression'):
    pb = Pseudobulk('data/pseudobulk/p400_qcd')
    de = pb\
        .DE(DE_label_column='pmAD', 
            covariate_columns=['age_death', 'sex', 'pmi', 'apoe4_dosage'],
            include_library_size_as_covariate=True,
            voom_plot_directory='figures/DE/voom/p400')
    de.write_csv('results/DE/p400_broad.csv')     

print(Pseudobulk.get_num_DE_hits(de, threshold=0.05).sort('cell_type'))
print_df(Pseudobulk.get_DE_hits(de, threshold=1, num_top_hits=20)\
         .filter(cell_type='Inhibitory'))


# MIT p400 #####################################################################

with Timer('Load single-cell data'):
    import gc
    from scipy.sparse import vstack
    cell_types = ['Excitatory_neurons_set1', 'Excitatory_neurons_set2', 
                  'Excitatory_neurons_set3', 'Inhibitory_neurons',
                  'Oligodendrocytes', 'OPCs', 'Astrocytes',
                  'Immune_cells', 'Vasculature_cells'] 
    sc_per_cell_type = {cell_type: SingleCell(
        f'data/single-cell/Mathys/processed/{cell_type}.rds')
                        for cell_type in cell_types}
    gc.collect()
    
    print('Merging...')
    X = vstack([sc_per_cell_type[cell_type].X 
                for cell_type in cell_types])
    obs = pl.concat([sc_per_cell_type[cell_type].obs 
                     for cell_type in cell_types])
    var = sc_per_cell_type[cell_types[0]].var
    assert all(var.equals(sc_per_cell_type[cell_type].var)
               for cell_type in cell_types[1:])
    sc = SingleCell(X=X, obs=obs, var=var)
    
    
    
sc_full = SingleCell('data/single-cell/Mathys/PFC427_raw_data.h5ad')

sc = SingleCell('data/single-cell/Mathys/processed/Immune_cells.rds')
sc.obs['cell'][0]
sc['AACTTTCAGGATGGTC-1-0']
sc[0]

sc_full['SM_Last16_AAACCCATCCGTAGTA-1']
sc_full[1]

with Timer('Find doublets'):
    import scrublet as scr
    scrub = scr.Scrublet(sc.X, expected_doublet_rate=0.045)
    sc.obs['doublet_scores'], sc.obs['predicted_doublets'] = \
        scrub.scrub_doublets(n_prin_comps=20)

rosmap_pheno = \
    pl.read_csv(
        'data/single-cell/p400/dataset_978_basic_04-21-2023_with_pmAD.csv',
        dtypes={'projid': pl.String})\
    .unique(subset='projid')
    
    
#    .drop([col for col in sc.obs.columns if col != 'projid'])







################################################################################

import os, optuna, pickle
import polars as pl, pandas as pd, numpy as np
import seaborn as sns
import matplotlib.pylab as plt

os.chdir('projects/def-wainberg/karbabi/single-cell-nmf')
from utils import Pseudobulk, debug, savefig
from sklearn.decomposition._nmf import _initialize_nmf
from sparseNMF import sparse_nmf
from ryp import r, to_r    

#debug(third_party=True)

def sparseness_hoyer(x):
    """
    The sparseness of array x is a real number in [0, 1], where sparser array
    has value closer to 1. Sparseness is 1 iff the vector contains a single
    nonzero component and is equal to 0 iff all components of the vector are 
    the same
        
    modified from Hoyer 2004: [sqrt(n)-L1/L2]/[sqrt(n)-1]
    adapted from nimfa package: https://nimfa.biolab.si/
    """
    from math import sqrt 
    eps = np.finfo(x.dtype).eps if 'int' not in str(x.dtype) else 1e-9
    n = x.size
    if np.min(x) < 0:
        x -= np.min(x)
    if np.allclose(x, np.zeros(x.shape), atol=1e-6):
        return 0.0
    L1 = abs(x).sum()
    L2 = sqrt(np.multiply(x, x).sum())
    sparseness_num = sqrt(n) - (L1 + eps) / (L2 + eps)
    sparseness_den = sqrt(n) - 1
    return sparseness_num / sparseness_den  

# rank_max=10
# spar=0.2
# reps=3
# n=0.05
# verbose=True

def cross_validate(A, rank_max, spar, reps, n, verbose=False):
    res = []
    for rep in range(1, reps + 1):
        np.random.seed(rep)
        mask = np.zeros(A.shape, dtype=bool)
        zero_indices = np.random.choice(
            A.size, int(round(n * A.size)), replace=False)
        mask.flat[zero_indices] = True
        A_masked = A.copy()
        A_masked.flat[zero_indices] = 0

        for rank in range(1, rank_max + 1):
            W, H = _initialize_nmf(A_masked, rank, init='nndsvd')
            W, H = sparse_nmf(A_masked, rank=rank, spar=spar, W=W, H=H,
                              tol=1e-4, maxiter=np.iinfo('int32').max,
                              verbose=False)
            A_r = W @ H
            MSE = np.mean((A[mask] - A_r[mask]) ** 2)
            sparseness = sparseness_hoyer(H)
            if verbose:      
                print(f'{rep=}, {rank=}, {MSE=:0.4f}, {sparseness=:0.4f}')
            res.append((rep, rank, MSE, sparseness))
            
    results = pd.DataFrame(
        res, columns=['rep', 'rank', 'MSE', 'sparseness'])\
        .set_index(['rank', 'rep'])   
    return results

def objective(trial, MSE_trial, rank_1se_trial, A, rank_max,
              reps=3, n=0.05, verbose=False):
    
    from scipy.stats import sem 
    m = A.shape[0]
    spar_l = (np.sqrt(m) - np.sqrt(m - 1)) / (np.sqrt(m) - 1) + 1e-10
    spar_u = (np.sqrt(m) + np.sqrt(m - 1)) / (np.sqrt(m) - 1) - 1e-10
    spar = trial.suggest_float('spar', spar_l, spar_u, log=True)
    
    results = cross_validate(A, rank_max, spar, reps, n, verbose)    
    mean_MSE = results.groupby('rank').mean()['MSE']
    rank_best = int(mean_MSE.idxmin())
    rank_1se = int(mean_MSE.index[mean_MSE <= mean_MSE[rank_best] + \
        sem(results.loc[rank_best, 'MSE'])][0])
    
    # mean_sparseness = results.groupby('rank').mean()['sparseness']
    # sparseness = mean_sparseness[rank_1se]
    
    MSE_trial[spar] = mean_MSE
    rank_1se_trial[spar] = rank_1se
    print(f'{rank_1se=}')
    MSE = mean_MSE[rank_1se]
    return MSE

################################################################################

save_name = '_fdr05_cpm_norm'

de = pl.read_csv('results/DE/p400_broad.csv')
pb = Pseudobulk('data/pseudobulk/p400_qcd').filter_obs(pmAD=1)
shared_ids = sorted(set.intersection(*(set(obs['ID'])
                                       for obs in pb.iter_obs())))
cpm = pb.filter_obs(pl.col.ID.is_in(shared_ids)).CPM()

cell_types = list(cpm.keys())
matrices, cell_types, genes = [], [], []
for cell_type, (X, obs, var) in cpm.items():
    gene_mask = var['_index'].is_in(
        de.filter((pl.col.cell_type == cell_type) & \
            (pl.col.FDR < 0.05))['gene']) 
    matrices.append(X.T[gene_mask]) 
    gene_select = var['_index'].filter(gene_mask).to_list()    
    genes.extend(gene_select)    
    cell_types.extend([cell_type] * len(gene_select))

A = np.vstack(matrices)
A = A / np.mean(A, axis=1)[:, None]

MSE_trial, rank_1se_trial = {}, {}
study = optuna.create_study(
    sampler=optuna.samplers.TPESampler(multivariate=True),
    direction='minimize')
study.optimize(lambda trial: objective(
    trial, MSE_trial, rank_1se_trial, A, rank_max=15, verbose=True), 
    n_trials=20)

with open(f'results/NMF/trial/p400_trial{save_name}.pkl', 'wb') as file:
    pickle.dump((study, MSE_trial, rank_1se_trial), file)
with open(f'results/NMF/trial/p400_trial{save_name}.pkl', 'rb') as file:
    study, MSE_trial, rank_1se_trial = pickle.load(file)

spar_select = study.best_trial.params.get('spar')
rank_select = rank_1se_trial[spar_select]

W, H = _initialize_nmf(A, n_components=rank_select, init='nndsvd')
W, H = sparse_nmf(A, rank=rank_select, spar=spar_select, W=W, H=H,
                  tol=1e-6, maxiter=np.iinfo('int32').max,
                  verbose=True)
A_r = W @ H

W = pl.DataFrame(W)\
    .rename(lambda col: col.replace('column_', 'S'))\
    .insert_column(0, pl.Series(genes).alias('gene'))
H = pl.DataFrame(H.T)\
    .rename(lambda col: col.replace('column_', 'S'))\
    .insert_column(0, pl.Series(shared_ids).alias('ID'))
W.write_csv(f'results/NMF/factors/p400_W{save_name}.tsv', separator='\t')
H.write_csv(f'results/NMF/factors/p400_H{save_name}.tsv', separator='\t')

pl.DataFrame(A, shared_ids)\
    .insert_column(0, pl.Series(genes).alias('gene'))\
    .write_csv(f'results/NMF/A/p400_A{save_name}.tsv', separator='\t')
pl.DataFrame(A_r, shared_ids)\
    .insert_column(0, pl.Series(genes).alias('gene'))\
    .write_csv(f'results/NMF/A/p400_Ar{save_name}.tsv', separator='\t')


pb = Pseudobulk('data/pseudobulk/p400_qcd')\
    .filter_obs(pl.col.ID.is_in(shared_ids))
for cell_type, (_, obs, _) in cpm.items():
    pb.obs[cell_type] = obs.join(H, on='ID', how='left') 

de = pb\
    .DE(DE_label_column='S0', 
        case_control=False, 
        covariate_columns=['age_death', 'sex', 'pmi', 'apoe4_dosage'],
        include_library_size_as_covariate=True,
        voom_plot_directory='figures/DE/voom/p400_H')
de.write_csv('results/DE/p400_H.csv')     




fig, ax = plt.subplots(figsize=(8, 7)) 
MSE_values = []
spar_min = min([spar for spar in MSE_trial.keys()])
for (current_spar), mean_MSE in MSE_trial.items():
    MSE_values.extend(mean_MSE.values) 
    rank_1se = rank_1se_trial[current_spar]
    ax.plot(mean_MSE.index, mean_MSE.values, color='black', alpha=0.1)
    ax.scatter(rank_1se, mean_MSE[rank_1se], color='black', s=16, alpha=0.1)
mean_MSE_select = MSE_trial[spar_select]
rank_select = rank_1se_trial[spar_select]
lower, upper = np.quantile(MSE_values, [0, 0.9])
ax.set_ylim(bottom=lower-0.05, top=upper)
ax.plot(mean_MSE_select.index, mean_MSE_select.values, 
        linewidth = 3, color='red')
ax.scatter(rank_select, mean_MSE_select[rank_select], color='red', s=80)
#ax.set_xticks(ticks=mean_MSE_select.index)
ax.set_yscale('log')
ax.set_title(rf'$\mathbf{{All\ cell\ types}}$'
            + "\nMSE across Optuna trials\n"
            + f"Best rank: {rank_select}, "
            + f"Best spar: {spar_select:.2g}, Min spar: {spar_min:.2g}")
ax.set_xlabel('Rank', fontsize=12, fontweight='bold')
ax.set_ylabel('Mean MSE', fontsize=12, fontweight='bold')
fig.subplots_adjust(bottom=0.1, top=0.9, hspace=0.3, wspace=0.25)
savefig(f"figures/NMF/MSE/p400_MSE{save_name}.png", dpi=300)


meta = cpm.obs[next(cpm.keys())]
meta.write_csv('results/NMF/A/p400_metadata.tsv', separator='\t')
pd.DataFrame(cell_types).to_csv('results/NMF/A/p400_celltypes.tsv', 
                                sep='\t', index=False)

to_r(np.array(cell_types), 'cell_types')
to_r(A, 'A', format='matrix', rownames=genes)
to_r(A_r, 'A_r', format='matrix', rownames=genes)

to_r(W.drop('gene'), "W", rownames=W['gene'])
to_r(H.drop('ID'), "H", rownames=shared_ids)
to_r(save_name, 'save_name')

r('''
suppressPackageStartupMessages({
        library(ComplexHeatmap)
        library(circlize)
        library(seriation)
        library(scico)
        library(ggsci)
    })
    mat = A
    row_order = get_order(seriate(dist(mat), method = "OLO"))
    col_order = get_order(seriate(dist(t(mat)), method = "OLO"))
    
    # create_color_list = function(data, palette) {
    #     n = ncol(data)
    #     cols = scico(n+1, palette = palette)[1:n]
    #     col_funs = mapply(function(col, max_val) {
    #         colorRamp2(c(0, max_val), c("white", col))
    #         }, cols, apply(data, 2, max), SIMPLIFY = FALSE)
    #     setNames(col_funs, colnames(data))
    # }
    # col_list_H = create_color_list(H, "batlow")
    # col_list_W = create_color_list(W, "batlow")
    colors = pal_frontiers()(length(unique(cell_types)))
    names(colors) = unique(cell_types)
    
    hr1 = rowAnnotation(
        cell_types = factor(cell_types),
        simple_anno_size = unit(0.3, "cm"),
        col = list(cell_types = colors),
        name = "cell type",
        show_annotation_name = FALSE,
        show_legend = TRUE,
        annotation_legend_param  = list(
            title_gp = gpar(fontsize = 7, fontface = "bold"),
            labels_gp = gpar(fontsize = 5)))
    
    # hr2 = rowAnnotation( 
    #     df = W, col = col_list_W,
    #     simple_anno_size = unit(0.3, "cm"),
    #     annotation_name_gp = gpar(fontsize = 8),
    #     annotation_name_side = "bottom",
    #     show_legend = FALSE)
     
    # hb = HeatmapAnnotation(
    #     df = H, col = col_list_H,
    #     simple_anno_size = unit(0.3, "cm"),
    #     annotation_name_gp = gpar(fontsize = 8),
    #     annotation_name_side = "left",    
    #     show_legend = FALSE)         

    col_fun = colorRamp2(quantile(mat, probs = c(0.00, 1.00)), 
        hcl_palette = "Batlow", reverse = TRUE)
    h = Heatmap(
        mat,
        row_order = row_order,
        column_order = col_order,
        cluster_rows = F,
        cluster_columns = F,
        show_row_names = FALSE,
        show_column_names = FALSE,
        # bottom_annotation = hb,
        left_annotation = hr1,
        # right_annotation = hr2,
        col = col_fun,
        name = "normalized\nexpression",
        heatmap_legend_param = list(
            title_gp = gpar(fontsize = 7, fontface = "bold"),
            labels_gp = gpar(fontsize = 5)),
        show_heatmap_legend = TRUE,
        use_raster = FALSE
    )
    file_name = paste0("figures/NMF/A/p400_A", save_name, ".png")
    png(file = file_name, width=7, height=7, units="in", res=1200)
    draw(h)
    dev.off()
''')

################################################################################
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, leaves_list

df = H.to_pandas().set_index('ID')
df = (df-df.min())/(df.max()-df.min())
row_order = leaves_list(linkage(pdist(df.to_numpy()),
                               method='complete', optimal_ordering=True))
df = df.iloc[row_order]

fig, ax = plt.subplots(figsize=(5,10))    
sns.heatmap(df, cmap='rocket_r', cbar=False, 
            xticklabels=True, yticklabels=False, rasterized=True)
ax.set_ylabel('Samples', fontsize=12, fontweight='bold')
savefig('tmp.png')





















def norm(X, p="fro"):
    assert 1 in X.shape or p != 2, "Computing entry-wise norms only."
    return np.linalg.norm(np.mat(X), p)

def random_c(V, rank, options, seed=0):
    from operator import itemgetter
    rank = rank
    p_c = options.get('p_c', int(np.ceil(1. / 5 * V.shape[1])))
    p_r = options.get('p_r', int(np.ceil(1. / 5 * V.shape[0])))
    l_c = options.get('l_c', int(np.ceil(1. / 2 * V.shape[1])))
    l_r = options.get('l_r', int(np.ceil(1. / 2 * V.shape[0])))
    
    prng = np.random.RandomState(seed=seed)
    W = np.mat(np.zeros((V.shape[0], rank)))
    H = np.mat(np.zeros((rank, V.shape[1])))
    top_c = sorted(enumerate([norm(V[:, i], 2)
                    for i in range(
                        V.shape[1])]), key=itemgetter(1), reverse=True)[:l_c]
    top_r = sorted(
        enumerate([norm(V[i, :], 2) for i in range(V.shape[0])]),
        key=itemgetter(1), reverse=True)[:l_r]
    
    top_c = np.mat(list(zip(*top_c))[0])
    top_r = np.mat(list(zip(*top_r))[0])
    for i in range(rank):
        W[:, i] = V[
            :, top_c[0, prng.randint(low=0, high=l_c, size=p_c)]\
                .tolist()[0]].mean(axis=1)
        H[i, :] = V[
            top_r[0, prng.randint(low=0, high=l_r, size=p_r)]\
                .tolist()[0], :].mean(axis=0)
    return np.array(W), np.array(H)

def random_vcol(V, rank, options, seed=0):
    rank = rank
    p_c = options.get('p_c', int(np.ceil(1. / 5 * V.shape[1])))
    p_r = options.get('p_r', int(np.ceil(1. / 5 * V.shape[0])))
    prng = np.random.RandomState(seed=seed)
    W = np.mat(np.zeros((V.shape[0], rank)))
    H = np.mat(np.zeros((rank, V.shape[1])))
    for i in range(rank):
        W[:, i] = V[:, prng.randint(
            low=0, high=V.shape[1], size=p_c)].mean(axis=1)
        H[i, :] = V[
            prng.randint(low=0, high=V.shape[0], size=p_r), :].mean(axis=0)
    return np.array(W), np.array(H)

n_runs = 100
corr_list = []
cluster_agreement  = np.zeros((A.shape[1], A.shape[1]))

for run in range(n_runs):
    #W, H = random_vcol(np.matrix(A), rank=9, options=dict(), seed=run)
    #W, H = random_c(np.matrix(A), rank=9, options=dict(), seed=run)
    W, H = sparse_nmf(A, rank=9, maxiter=50, spar=spar_select, 
                      seed=run, verbose=True)
    
    corr_list.append(np.corrcoef(H, rowvar=False))
    
    cluster_membership = np.argmax(H, axis=0)
    overlap_matrix = (cluster_membership[:, None] == \
        cluster_membership[None, :]).astype(int)
    cluster_agreement  += overlap_matrix
    
corr_sd_matrix = np.std(corr_list, axis=0)
cluster_agreement  /= n_runs

sns.clustermap(corr_sd_matrix, method='average', cmap='rocket_r',
               xticklabels=False, yticklabels=False, figsize=(10, 10))
plt.savefig('corr_sd_matrix3.png')
sns.clustermap(cluster_agreement , method='average', 
               xticklabels=False, yticklabels=False, figsize=(10, 10))
plt.savefig('cluster_agreement3.png')













mat = A[0:500, 0:100]

m = A.shape[0]
spar_l = (np.sqrt(m) - np.sqrt(m - 1)) / (np.sqrt(m) - 1) + 1e-10
spar_u = (np.sqrt(m) + np.sqrt(m - 1)) / (np.sqrt(m) - 1) - 1e-10
spar_l
spar_u

W, H = sparse_nmf(mat, rank=5, maxiter=300, spar=0.5)
W = pl.DataFrame(W)\
    .rename(lambda col: col.replace('column_', 'S'))\
    .insert_column(0, pl.Series(genes[0:500]).alias('gene'))
H = pl.DataFrame(H.T)\
    .rename(lambda col: col.replace('column_', 'S'))\
    .insert_column(0, pl.Series(shared_ids[0:100]).alias('ID'))
    
to_r(mat, 'mat', format='matrix', rownames=genes[0:500])
to_r(W.select(pl.exclude('gene')), "W", 
        format='data.frame', rownames=W['gene'])
to_r(H.select(pl.exclude('ID')), "H",
        format='data.frame', rownames=H['ID'].cast(str))

r('''
    suppressPackageStartupMessages({
        library(ComplexHeatmap)
        library(circlize)
        library(seriation)
        library(scico)
        library(viridis)
    })
    row_order = get_order(seriate(dist(W), method = "OLO"))
    col_order = get_order(seriate(dist(H), method = "OLO"))
    
    create_color_list = function(data, palette) {
        cols = viridis(n = ncol(data)+2, option = "rocket")[1:ncol(data)]
        col_funs = mapply(function(col, max_val) {
            colorRamp2(c(0, max_val), c("white", col))
        }, cols, apply(data, 2, max), SIMPLIFY = FALSE)
        setNames(col_funs, colnames(data))
    }
    col_list_H = create_color_list(H)
    col_list_W = create_color_list(W)

    hr = rowAnnotation( 
        df = W, col = col_list_W,
        simple_anno_size = unit(0.4, "cm"),
        annotation_name_gp = gpar(fontsize = 8),
        show_annotation_name = FALSE,    
        show_legend = FALSE)
     
    hb = HeatmapAnnotation(
        df = H, col = col_list_H,
        simple_anno_size = unit(0.4, "cm"),
        annotation_name_gp = gpar(fontsize = 8),
        show_annotation_name = FALSE,    
        show_legend = FALSE)         

    col_breaks = seq(min(mat), max(mat), 0.1)
    col_fun = colorRamp2(col_breaks, 
                        viridis(length(col_breaks), option = "rocket"))
    h = Heatmap(
            mat,
            row_order = row_order,
            column_order = col_order,
            cluster_rows = F,
            cluster_columns = F,
            show_row_names = FALSE,
            show_column_names = FALSE,
            bottom_annotation = hb,
            left_annotation = hr,
            col = col_fun,
            show_heatmap_legend = FALSE,
            use_raster = FALSE
        )
        file_name = "tmp4.png"
        png(file = file_name, width=5, height=5, units="in", res=1200)
        draw(h)
    dev.off()
''')

























