import os
import polars as pl, numpy as np

os.chdir('projects/def-wainberg/karbabi/single-cell-nmf')
from utils import Timer, print_df, get_coding_genes, debug, \
    SingleCell, Pseudobulk

debug(third_party=True)

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
        .with_columns_obs(projid=pl.col.projid.cast(pl.String))\
        .pseudobulk(ID_column='projid', 
                    cell_type_column='subset',
                    additional_obs=rosmap_pheno)
    #pb.save('data/pseudobulk/p400')

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
    #pb.save('data/pseudobulk/p400_qcd')
        
with Timer('Differential expression'):
    pb = Pseudobulk('data/pseudobulk/p400_qcd')
    de = pb\
        .DE(DE_label_column='pmAD', 
            covariate_columns=['age_death', 'sex', 'pmi', 'apoe4_dosage'],
            include_library_size_as_covariate=True,
            voom_plot_directory='figures/DE/voom/p400')
    #de.write_csv('results/DE/p400_broad.csv')     

print(Pseudobulk.get_num_DE_hits(de, threshold=0.1).sort('cell_type'))
print_df(Pseudobulk.get_DE_hits(de, threshold=1, num_top_hits=20)\
         .filter(cell_type='Inhibitory'))

'''
cell_type         num_hits 
 Astrocytes        568      
 CUX2+             1305     
 Endothelial       2        
 Inhibitory        601      
 Microglia         21       
 OPCs              4        
 Oligodendrocytes  289      
shape: (7, 2)
 
'''
################################################################################

import os, optuna, pickle
import polars as pl, pandas as pd, numpy as np
import matplotlib.pylab as plt

os.chdir('projects/def-wainberg/karbabi/single-cell-nmf')
from sparseNMF import sparse_nmf
from utils import Pseudobulk, debug, savefig
from ryp import r, to_r    
from project_utils import normalize_matrix

debug(third_party=True)

def cross_validate(A, r_max, spar, reps, n, verbose=False):
    res = []
    for rep in range(1, reps + 1):
        np.random.seed(rep)
        mask = np.zeros(A.shape, dtype=bool)
        zero_indices = np.random.choice(
            A.size, int(round(n * A.size)), replace=False)
        mask.flat[zero_indices] = True
        A_masked = A.copy()
        A_masked.flat[zero_indices] = 0

        for r in range(1, r_max + 1):
            W, H = sparse_nmf(A_masked, r, maxiter=50, spar=spar)
            A_r = W @ H
            MSE = np.mean((A[mask] - A_r[mask]) ** 2)
            if verbose:      
                print(f'rep {rep}, rank: {r}, MSE: {MSE}')
            res.append((rep, r, MSE))
            
    MSE = pd.DataFrame(res, columns=['rep', 'r', 'MSE'])\
        .astype({'r': int, 'rep': int})\
        .set_index(['r', 'rep']).squeeze()
    return(MSE)    

def objective(trial, MSE_trial, r_1se_trial, cell_type, 
              A, r_max, reps=3, n=0.05, verbose=False):
    
    from scipy.stats import sem 
    m = A.shape[0]
    spar_l = (np.sqrt(m) - np.sqrt(m - 1)) / (np.sqrt(m) - 1) + 1e-10
    spar_u = (np.sqrt(m) + np.sqrt(m - 1)) / (np.sqrt(m) - 1) - 1e-10
    spar = trial.suggest_float('spar', spar_l, spar_u, log=True)
    
    MSE = cross_validate(A, r_max, spar, reps, n, verbose)
    mean_MSE = MSE.groupby('r').mean()
    r_best = int(mean_MSE.idxmin())
    r_1se = int(mean_MSE.index[mean_MSE <= mean_MSE[r_best] + \
        sem(MSE[r_best])][0])
    
    MSE_trial[cell_type, spar] = MSE
    r_1se_trial[cell_type, spar] = r_1se
    print(f'[{cell_type}]: {r_1se=}')
    error = mean_MSE[r_1se]
    return(error)

################################################################################

de = pl.read_csv('results/DE/p400_broad.csv')
lcpm = Pseudobulk('data/pseudobulk/p400_qcd')\
    .drop(Pseudobulk.get_num_DE_hits(de, threshold=0.1)\
            .filter(pl.col.num_hits <= 100)['cell_type'])\
    .filter_obs(pmAD=1)\
    .log_CPM(prior_count=2)

r_max = 30
n_trials = 30
save_name = '_rint'
to_r(save_name, 'save_name')

MSE_trial, r_1se_trial, spar_select_trial = {}, {}, {}
for cell_type, (X, obs, var) in lcpm.items():
    
    gene_mask = var['_index'].is_in(de.filter(
        (pl.col.cell_type == cell_type) & (pl.col.FDR < 0.1))['gene'])
    A = X.T[gene_mask] 
    A = normalize_matrix(A, 'rint')
    
    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(seed=0, multivariate=True),
        #sampler=optuna.samplers.RandomSampler(seed=0),
        direction='minimize')
    study.optimize(lambda trial: objective(
        trial, MSE_trial, r_1se_trial, cell_type, 
        A, r_max, reps=3, n=0.05, verbose=True), 
        n_trials=n_trials)
    spar_select = study.best_trial.params.get('spar', 0)
    r_select = r_1se_trial[cell_type, spar_select]
    spar_select_trial[cell_type] = spar_select
    
    W, H = sparse_nmf(A, r_select, maxiter=300, spar=spar_select)
    A_r = W @ H
    
    os.makedirs(f'results/NMF/factors/p400', exist_ok=True)    
    pl.DataFrame(W)\
        .rename(lambda col: col.replace('column_', 'S'))\
        .insert_column(0, var.filter(gene_mask)['_index'].alias('gene'))\
        .write_csv(f'results/NMF/factors/p400/{cell_type}_W{save_name}.tsv',
                   separator='\t')
    pl.DataFrame(H.T)\
        .rename(lambda col: col.replace('column_', 'S'))\
        .insert_column(0, obs['ID'].alias('ID'))\
        .write_csv(f'results/NMF/factors/p400/{cell_type}_H{save_name}.tsv', 
                   separator='\t')
    A_r = pl.DataFrame(A_r)\
        .insert_column(0, var.filter(gene_mask)['_index'].alias('gene'))
    A_r.columns = ['gene'] + obs['ID'].to_list()
    A_r.write_csv(f'results/NMF/factors/p400/{cell_type}_A_r{save_name}.tsv', 
            separator='\t')
    
fig, axes = plt.subplots(2, 2, figsize=(12, 11))
for idx, cell_type in enumerate(lcpm.keys()):    
    from matplotlib import cm
    row, col = divmod(idx, 2)
    ax = axes[row, col]
    MSE_values = []
    spar_min = min([spar for _, spar in MSE_trial.keys() if _ == cell_type])
    for (current_cell_type, current_spar), MSE in MSE_trial.items():
        if current_cell_type == cell_type:
            mean_MSE = MSE.groupby('r').mean()
            MSE_values.extend(mean_MSE.values) 
            r_1se = r_1se_trial[current_cell_type, current_spar]
            ax.plot(mean_MSE.index, mean_MSE.values, color='black', alpha=0.1)
            ax.scatter(r_1se, mean_MSE[r_1se], color='black', s=16, alpha=0.1)
    spar_select = spar_select_trial[cell_type]
    MSE_select = MSE_trial[cell_type, spar_select]
    mean_MSE = MSE_select.groupby('r').mean()
    r_final = r_1se_trial[cell_type, spar_select]
    
    lower, upper = np.quantile(MSE_values, [0, 0.9])
    ax.set_ylim(bottom=lower-0.05, top=upper)
    ax.plot(mean_MSE.index, mean_MSE.values, linewidth = 3, color='red')
    ax.scatter(r_final, mean_MSE[r_final], color='red', s=80)
    #ax.set_xticks(ticks=mean_MSE.index)
    ax.set_yscale('log')
    ax.set_title(rf'$\mathbf{{{cell_type}}}$'
                + "\nMSE across Optuna trials\n"
                + f"Best r: {r_final}, "
                + f"Best spar: {spar_select:.2g}, Min spar: {spar_min:.2g}")
    ax.set_xlabel('r')
    ax.set_ylabel('Mean MSE')
fig.subplots_adjust(bottom=0.1, top=0.9, hspace=0.3, wspace=0.25)
savefig(f"figures/NMF/MSE/p400/MSE{save_name}.png", dpi=300)

for cell_type, (X, obs, var) in lcpm.items():   
    gene_mask = var['_index'].is_in(
        de.filter((pl.col.cell_type == cell_type) & (pl.col.FDR < 0.1))['gene']) 
    A = X.T[gene_mask] 
    A = normalize_matrix(A, 'rint')
    to_r(A, 'A', format='matrix',
        rownames=var['_index'].filter(gene_mask),
        colnames=obs['ID'])
    to_r(cell_type, 'cell_type')
    
    H = pl.read_csv(f'results/NMF/factors/p400/{cell_type}_H{save_name}.tsv',
        separator='\t')
    W = pl.read_csv(f'results/NMF/factors/p400/{cell_type}_W{save_name}.tsv',
        separator='\t')
    A_r = pl.read_csv(f'results/NMF/factors/p400/{cell_type}_A_r{save_name}.tsv',
        separator='\t')
    to_r(H.select(pl.exclude('ID')), "H",
         format='data.frame', rownames=H['ID'].cast(str))
    to_r(W.select(pl.exclude('gene')), "W", 
         format='data.frame', rownames=W['gene'])
    to_r(A_r.select(pl.exclude('gene')), 'A_r',
         format='matrix', rownames=W['gene'])
    
    os.makedirs(f'figures/NMF/A/p400', exist_ok=True)
    r('source("project_utils.R")')
    r('plot_3M_heatmap(A, H, W, celltype, cluster="factors", save_name)')
    r('plot_3M_heatmap(A_r, H, W, celltype, cluster="factors", "_rint_r")')

################################################################################

def objective(trial, MSE_trial, r_1se_trial, A, r_max,
              reps=3, n=0.05, verbose=False):
    
    from scipy.stats import sem 
    m = A.shape[0]
    spar_l = (np.sqrt(m) - np.sqrt(m - 1)) / (np.sqrt(m) - 1) + 1e-10
    spar_u = (np.sqrt(m) + np.sqrt(m - 1)) / (np.sqrt(m) - 1) - 1e-10
    spar = trial.suggest_float('spar', spar_l, spar_u, log=True)
    
    MSE = cross_validate(A, r_max, spar, reps, n, verbose)
    mean_MSE = MSE.groupby('r').mean()
    r_best = int(mean_MSE.idxmin())
    r_1se = int(mean_MSE.index[mean_MSE <= mean_MSE[r_best] + \
        sem(MSE[r_best])][0])
    
    MSE_trial[spar] = MSE
    r_1se_trial[spar] = r_1se
    print(f'[{cell_type}]: {r_1se=}')
    error = mean_MSE[r_1se]
    return(error)

de = pl.read_csv('results/DE/p400_broad.csv')
pb = Pseudobulk('data/pseudobulk/p400_qcd')\
    .drop(Pseudobulk.get_num_DE_hits(de, threshold=0.1)\
            .filter(pl.col.num_hits <= 100)['cell_type'])\
    .filter_obs(pmAD=1)
shared_IDs = list(set.intersection(*[set(obs['ID'].to_list())
                                     for _, (_, obs, _) in pb.items()]))
lcpm = pb\
    .filter_obs(pl.col.ID.is_in(shared_IDs))\
    .log_CPM(prior_count=2)
    
cell_types = list(lcpm.keys())
matrices, gene_cell_type = [], []

MSE_trial, r_1se_trial = {}, {}

for cell_type, (X, _, var) in lcpm.items():
    gene_mask = var['_index'].is_in(
        de.filter((pl.col.cell_type == cell_type) & (pl.col.FDR < 0.1))['gene']) 
    matrices.append(normalize_matrix(X.T[gene_mask], 'rint'))
    gene_cell_type += [(gene, cell_type) 
                   for gene in var['_index'].filter(gene_mask).to_list()]
    
A = np.vstack(matrices)
A = normalize_matrix(A, 'rint')
  
study = optuna.create_study(
    sampler=optuna.samplers.TPESampler(seed=0, multivariate=True),
    #sampler=optuna.samplers.RandomSampler(seed=0),
    direction='minimize')
study.optimize(lambda trial: objective(
    trial, MSE_trial, r_1se_trial, A, r_max, reps=3, n=0.05, verbose=True), 
    n_trials=n_trials)
spar_select = study.best_trial.params.get('spar', 0)
r_select = r_1se_trial[spar_select]

W, H = sparse_nmf(A, r_select, maxiter=300, spar=spar_select)
A_r = W @ H

os.makedirs(f'results/NMF/factors/p400', exist_ok=True)    
pl.DataFrame(W)\
    .rename(lambda col: col.replace('column_', 'S'))\
    .insert_column(0, var.filter(gene_mask)['_index'].alias('gene'))\
    .write_csv(f'results/NMF/factors/p400/combined_W{save_name}.tsv',
                separator='\t')
pl.DataFrame(H.T)\
    .rename(lambda col: col.replace('column_', 'S'))\
    .insert_column(0, obs['ID'].alias('ID'))\
    .write_csv(f'results/NMF/factors/p400/combined_H{save_name}.tsv', 
                separator='\t')
A_r = pl.DataFrame(A_r)\
    .insert_column(0, var.filter(gene_mask)['_index'].alias('gene'))
A_r.columns = ['gene'] + obs['ID'].to_list()
A_r.write_csv(f'results/NMF/factors/p400/combined_A_r{save_name}.tsv', 
        separator='\t')



















de = pl.read_csv('results/DE/p400_broad.csv')
lcpm = Pseudobulk('data/pseudobulk/p400_qcd')\
    .drop(Pseudobulk.get_num_DE_hits(de, threshold=0.1)\
            .filter(pl.col.num_hits <= 100)['cell_type'])\
    .filter_obs(pmAD=1)\
    .log_CPM(prior_count=2)

cell_type = 'Inhibitory'
X = lcpm.X[cell_type]
obs = lcpm.obs[cell_type]
var = lcpm.var[cell_type]

gene_mask = var['_index'].is_in(
    de.filter((pl.col.cell_type == cell_type) & (pl.col.FDR < 0.1))['gene']) 
mat = X.T[gene_mask] 
mat = normalize_matrix(mat, 'rint')
np.min(mat)
np.max(mat)
m, n = mat.shape


spar_min = (np.sqrt(m) - np.sqrt(m - 1)) / (np.sqrt(m) - 1) + 1e-10
spar_max = (np.sqrt(m) + np.sqrt(m - 1)) / (np.sqrt(m) - 1) - 1e-10

W, H = sparse_nmf(mat, 8, maxiter=10, spar=spar_max)
A = W @ H








































































