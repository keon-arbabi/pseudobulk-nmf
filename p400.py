import sys, os, pickle, optuna
import polars as pl, pandas as pd, numpy as np
import matplotlib.pylab as plt
import seaborn as sns

os.chdir('projects/def-wainberg/karbabi/single-cell-nmf')
from utils import Timer, print_df, get_coding_genes, debug, savefig, \
    SingleCell, Pseudobulk
from ryp import r, to_py, to_r    
from project_utils import plot_volcano, normalize_matrix

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
            max_standard_deviations=3, 
            min_nonzero_percent=80,
            min_num_cells=10,
            custom_filter=pl.col.pmAD.is_not_null())
    #pb.save('data/pseudobulk/p400_qcd')
        
with Timer('Differential expression'):
    pb = Pseudobulk('data/pseudobulk/p400_qcd')
    de = pb\
        .DE(DE_label_column='pmAD', 
            covariate_columns=['age_death', 'sex', 'pmi', 'apoe4_dosage'],
            include_library_size_as_covariate=True,
            voom_plot_directory=None)
    de.write_csv('results/differential-expression/p400_broad.csv')     

print(Pseudobulk.get_num_DE_hits(de, threshold=0.1).sort('cell_type'))
print_df(Pseudobulk.get_DE_hits(de, threshold=1, num_top_hits=20)\
         .filter(cell_type='Inhibitory'))

'''
cell_type         num_hits 
 Astrocytes        608      
 CUX2+             1097     
 Endothelial       2        
 Inhibitory        690      
 Microglia         28       
 OPCs              7        
 Oligodendrocytes  361       
 
'''
################################################################################

def nmf_objective(trial, mat, MSE_trial, k_1se_trial, study_name,
                  cell_type, k_max, L1=True, L2=False):
    
    from scipy.stats import sem
    r('library(RcppML, quietly=TRUE)')
    r('options(RcppML.verbose=TRUE)')

    L1_w = trial.suggest_float('L1_w', 0.001, 0.999, log=True) if L1 else 0
    L1_h = trial.suggest_float('L1_h', 0.001, 0.999, log=True) if L1 else 0
    L2_w = trial.suggest_float('L2_w', 0.001, 0.999, log=True) if L2 else 0
    L2_h = trial.suggest_float('L2_h', 0.001, 0.999, log=True) if L2 else 0
    to_r(L1_w, "L1_w"); to_r(L1_h, "L1_h")
    to_r(L2_w, "L2_w"); to_r(L2_h, "L2_h")
    
    #r('seed = sample(1:1000, 1, replace=FALSE)')
    r('MSE = RcppML::crossValidate(mat, k = seq(1, k_max, 1), '
        'L1 = c(L1_w, L1_h), L2 = c(L2_w, L2_h), '
        'seed = 1, reps = 3, tol = 1e-2, '
        'maxit = .Machine$integer.max)')
    
    MSE = to_py('MSE', format='pandas')\
        .astype({'k': int, 'rep': int})\
        .set_index(['k', 'rep'])\
        .squeeze()\
        .rename('MSE')
    mean_MSE = MSE.groupby('k').mean()
    k_best = int(mean_MSE.idxmin())
    k_1se = int(mean_MSE.index[mean_MSE <= mean_MSE[k_best] +\
        sem(MSE[k_best])][0])
    MSE_trial[study_name, cell_type, L1_w, L1_h, L2_w, L2_h] = MSE
    k_1se_trial[study_name, cell_type, L1_w, L1_h, L2_w, L2_h] = k_1se
    print(f'[{study_name} {cell_type}]: {k_1se=}')
    error = mean_MSE[k_1se]
    return error

def plot_MSE(axes, idx, study_name, cell_type, MSE_trial, 
             k_1se_trial, MSE_final, best_params):
    row, col = divmod(idx, 2)
    ax = axes[row, col]
    for (current_study, current_cell_type, L1_w, L1_h, L2_w, L2_h), \
        MSE in MSE_trial.items():
        if current_study == study_name and current_cell_type == cell_type:
            mean_MSE = MSE.groupby('k').mean()
            k_1se = k_1se_trial[study_name, cell_type, L1_w, L1_h, L2_w, L2_h]
            
            ax.plot(mean_MSE.index, mean_MSE.values, color='black', 
                    alpha=0.08)
            ax.scatter(k_1se, mean_MSE[k_1se], color='black', s=16, 
                       alpha=0.08)
    mean_MSE = MSE_final.groupby('k').mean()
    k_final = k_1se_trial[study_name, cell_type, *best_params.values()]
    ax.plot(mean_MSE.index, mean_MSE.values, color='red')
    ax.scatter(k_final, mean_MSE[k_final], color='red', s=50)
    ax.set_xticks(ticks=mean_MSE.index)
    ax.set_yscale('log')
    ax.set_title(rf"$\bf{{{study_name}\;{cell_type}}}$"
                + "\nMSE across Optuna trials\n"
                + f"Selected L1_w: {best_params['L1_w']:.3f}, "
                + f"L1_h: {best_params['L1_h']:.3f}, "
                + f"L2_w: {best_params['L2_w']:.3f}, "
                + f"L2_h: {best_params['L2_h']:.3f}")
    ax.set_xlabel('k')
    ax.set_ylabel('Mean MSE')

de = pl.read_csv('results/differential-expression/p400_broad.csv')
lcpm = Pseudobulk('data/pseudobulk/p400_qcd')\
    .drop(Pseudobulk.get_num_DE_hits(de, threshold=0.1)\
            .filter(pl.col.num_hits < 100)['cell_type'])\
    .filter_obs(pmAD=1)\
    .log_CPM(prior_count=2)

study_name = 'p400'
save_name = ''
n_trials = 10
k_max = 20; to_r(k_max, 'k_max')

MSE_trial, k_1se_trial = {}, {}
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for idx, (cell_type, (X, obs, var)) in enumerate(lcpm.items()):
    gene_mask = var['_index'].is_in(de.filter(
            (pl.col.cell_type == cell_type) & (pl.col.FDR < 0.1))['gene'])
    mat = X.T[gene_mask] 
    mat = normalize_matrix(mat, 'rint')
    to_r(mat, 'mat', format='matrix')
    
    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(multivariate=True),
        #sampler=optuna.samplers.RandomSampler(),
        direction='minimize')
    study.optimize(lambda trial: nmf_objective(
        trial, mat, MSE_trial, k_1se_trial,\
        study_name, cell_type, k_max, L1=True, L2=False), 
        n_trials=n_trials)

    best_params = {param: study.best_trial.params.get(param, 0)
                    for param in ['L1_w', 'L1_h', 'L2_w', 'L2_h']}
    L1_w, L1_h, L2_w, L2_h = best_params.values()
    MSE_final = MSE_trial[study_name, cell_type, *best_params.values()]
    k_select = k_1se_trial[study_name, cell_type, *best_params.values()]
    to_r(k_select, 'k_select')
    to_r(best_params, 'bp')
    r('''
        library(RcppML, quietly=True)
        options(RcppML.verbose = TRUE)
        res = nmf(mat, k = k_select, 
                L1=c(bp$L1_w, bp$L1_h), L2=c(bp$L2_w, bp$L2_h), 
                seed=1:10, tol=1e-5, maxit=.Machine$integer.max)
        H = res@h
        W = res@w
    ''')
    W = to_py('W', format='pandas')\
        .set_axis(var.filter(gene_mask)['_index'].to_list())\
        .rename(columns=lambda col: col.replace('nmf', 'S'))
    H = to_py('H', format='pandas').T\
        .set_axis(obs['ID'].to_list())\
        .rename(columns=lambda col: col.replace('nmf', 'S'))

    os.makedirs(f'results/MSE/{study_name}', exist_ok=True)
    os.makedirs(f'results/NMF/{study_name}', exist_ok=True)
    MSE_final.to_csv(f'results/MSE/{study_name}/{cell_type}_MSE{save_name}.tsv', sep='\t')
    W.to_csv(f'results/NMF/{study_name}/{cell_type}_W{save_name}.tsv', sep='\t')
    H.to_csv(f'results/NMF/{study_name}/{cell_type}_H{save_name}.tsv', sep='\t')

    plot_MSE(axes, idx, study_name, cell_type,
             MSE_trial, k_1se_trial, MSE_final, best_params)
fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.4, hspace=0.4)
savefig(f"results/MSE/{study_name}/MSE_plots{save_name}.png", dpi=300)




de = pl.read_csv('results/differential-expression/p400_broad.csv')
lcpm = Pseudobulk('data/pseudobulk/p400_qcd')\
    .drop(Pseudobulk.get_num_DE_hits(de, threshold=0.1)\
            .filter(pl.col.num_hits < 100)['cell_type'])\
    .filter_obs(pmAD=1)\
    .log_CPM(prior_count=2)

cell_type = 'Inhibitory'
X = lcpm.X[cell_type]
obs = lcpm.obs[cell_type]
var = lcpm.var[cell_type]

gene_mask = var['_index'].is_in(de.filter(
        (pl.col.cell_type == cell_type) & (pl.col.FDR < 0.1))['gene'])
mat = X.T[gene_mask] 
mat = normalize_matrix(mat, 'rint')

import nimfa

mat_m = 

snmf = nimfa.Snmf(mat, seed='random_vcol', version='l', 
                  max_iter=10, track_factor=True, track_error=True)

snmf.select_features()

ranks = range(1, 20, 1)
summary = snmf.estimate_rank(rank_range=ranks, n_run=10, what='all')
summary[1].keys()

spar = [summary[rank]['sparseness'] for rank in ranks]
rss = [summary[rank]['rss'] for rank in ranks]
evar = [summary[rank]['evar'] for rank in ranks]    
coph = [summary[rank]['cophenetic'] for rank in ranks]
disp = [summary[rank]['dispersion'] for rank in ranks]
spar_w, spar_h = zip(*spar)
        
fig, axs = plt.subplots(2, 3, figsize=(15, 10)) 

axs[0, 0].plot(ranks, rss, 'o-', label='RSS', linewidth=2)
axs[0, 0].set_title('RSS')
axs[0, 1].plot(ranks, coph, 'o-', label='Cophenetic correlation', linewidth=2)
axs[0, 1].set_title('Cophenetic correlation')
axs[0, 2].plot(ranks, disp, 'o-', label='Dispersion', linewidth=2)
axs[0, 2].set_title('Dispersion')
axs[1, 0].plot(ranks, spar_w, 'o-', label='Sparsity (Basis)', linewidth=2)
axs[1, 0].set_title('Sparsity (Basis)')
axs[1, 1].plot(ranks, spar_h, 'o-', label='Sparsity (Mixture)', linewidth=2)
axs[1, 1].set_title('Sparsity (Mixture)')
axs[1, 2].plot(ranks, evar, 'o-', label='Explained variance', linewidth=2)
axs[1, 2].set_title('Explained variance')

from matplotlib.ticker import MaxNLocator
for ax in axs.flat:
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
savefig("p3.png", dpi=300)

snmf = nimfa.Snmf(V, seed="random_c", rank=8, max_iter=30, version='r')
snmf_fit = snmf()
 
W = pl.DataFrame(snmf_fit.basis()).explode(pl.all())\
    .rename(lambda col: col.replace('column_', 'S'))
to_r(W, 'W', format='df', rownames=var['_index'].filter(gene_mask))

H = pl.DataFrame(snmf_fit.coef()).explode(pl.all())
H.columns = obs['ID']
to_r(H, 'H', format='df', rownames=W.columns)

















































def peek_metagenes(study_name, cell_type):
    W = pd.read_table(f'results/NMF/{study_name}/{cell_type}_W{save_name}.tsv',
                      index_col=0)
    print(pd.DataFrame({col: W.nlargest(20, col).index.tolist()
                        for col in W.columns}))

peek_metagenes('p400','Inhibitory')



with Timer('Volcano plots'):
    for cell_type in de['cell_type'].unique():
        df = de.filter(cell_type=cell_type)
        plot_volcano(df, threshold=0.1, num_top_genes=30,
                     min_distance=0.08,
                     plot_directory='figures/volcano/p400')     



for cell_type, (X, obs, var) in lcpm.items():
    gene_mask = var['_index'].is_in(
        de.filter((pl.col.cell_type == cell_type) & (pl.col.FDR < 0.1))['gene']) 
    mat = X.T[gene_mask] 
    mat = normalize_matrix(mat, 'rint')
    
    to_r(mat, 'mat', format='matrix',
        rownames=var['_index'].filter(gene_mask),
        colnames=obs['ID'])
    to_r(cell_type, 'cell_type')
    meta = obs.select(sorted([
        'num_cells', 'sex', 'Cdx', 'braaksc', 'ceradsc', 'pmi', 'niareagansc',
        'apoe_genotype', 'tomm40_hap','age_death', 'age_first_ad_dx', 'cogdx',
        'ad_reagan', 'gpath', 'amyloid', 'hspath_typ', 'dlbdx', 'tangles',
        'tdp_st4', 'arteriol_scler', 'caa_4gp', 'cvda_4gp2','ci_num2_gct',
        'ci_num2_mct', 'tot_cog_res']))
    to_r(meta, 'meta', format='data.frame', rownames=obs['ID'])
    
    W = pd.read_table(
        f'results/NMF/{study_name}/{cell_type}_W{save_name}.tsv', index_col=0)
    H = pd.read_table(
        f'results/NMF/{study_name}/{cell_type}_H{save_name}.tsv', index_col=0)
    H.index = H.index.astype(str)
    to_r(W, "W", format='data.frame')
    to_r(H, "H", format='data.frame')
    
    r('''
    suppressPackageStartupMessages({
            library(ComplexHeatmap)
            library(circlize)
            library(seriation)
            library(scico)
        })
        row_order = get_order(seriate(dist(W), method = "OLO"))
        col_order = get_order(seriate(dist(H), method = "OLO"))
        ht = HeatmapAnnotation(
            df = meta, 
            simple_anno_size = unit(0.15, "cm"),
            annotation_name_gp = gpar(fontsize = 5),
            show_legend = FALSE)     
        hb = HeatmapAnnotation(
            df = H, 
            simple_anno_size = unit(0.3, "cm"),
            annotation_name_gp = gpar(fontsize = 8),
            show_legend = FALSE)         
        hr = rowAnnotation(
            df = W,
            simple_anno_size = unit(0.3, "cm"),
            annotation_name_gp = gpar(fontsize = 8),
            show_legend = FALSE)
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
            bottom_annotation = hb,
            left_annotation = hr,
            col = col_fun,
            name = paste0('p400', "\n", cell_type),
            show_heatmap_legend = TRUE
        )
        draw(h)
        dev.off()
    ''')





