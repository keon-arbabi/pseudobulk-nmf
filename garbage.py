

meta = obs.select(sorted([
    'num_cells', 'sex', 'Cdx', 'braaksc', 'ceradsc', 'pmi', 'niareagansc',
    'apoe_genotype', 'tomm40_hap', 'age_death', 'age_first_ad_dx', 'cogdx',
    'ad_reagan', 'gpath', 'amyloid', 'hspath_typ', 'dlbdx', 'tangles',
    'tdp_st4', 'arteriol_scler', 'caa_4gp', 'cvda_4gp2', 'ci_num2_gct',
    'ci_num2_mct', 'tot_cog_res'
    ]))
to_r(meta, 'meta', format='data.frame', rownames=obs['ID'])
    
















import os, nimfa, optuna
import polars as pl, pandas as pd, numpy as np
import matplotlib.pylab as plt, seaborn as sns

os.chdir('projects/def-wainberg/karbabi/single-cell-nmf')
from sparseNMF import sparse_nmf
from utils import Timer, print_df, debug, savefig, Pseudobulk
from ryp import r, to_py, to_r    
from project_utils import normalize_matrix, \
    plot_volcano, plot_k_MSE, plot_k_stats

debug(third_party=True)

def cross_validate(mat, kmax, beta, reps, n, verbose=False):
    import nimfa    
    res = []
    for rep in range(1, reps + 1):
        np.random.seed(rep)
        mask = np.zeros(mat.shape, dtype=bool)
        zero_indices = np.random.choice(
            mat.size, round(n * mat.size), replace=False)
        mask.flat[zero_indices] = True
        mat_masked = mat.copy()
        mat_masked.flat[zero_indices] = 0
        for k in range(1, kmax + 1):
            snmf = nimfa.Snmf(mat_masked, rank=k, version='r',
                              beta=beta, max_iter=30, n_run=1)
            mat_est = snmf().fitted()
            mat_est = np.asarray(mat_est)
            if verbose:      
                print(f'rep {rep}, rank: {k}, MSE: {MSE}')
            res.append((rep, k, MSE))
            
    MSE = pd.DataFrame(res, columns=['rep', 'k', 'MSE'])\
        .astype({'k': int, 'rep': int})\
        .set_index(['k', 'rep']).squeeze()
    return(MSE)    

def objective(trial, MSE_trial, k_1se_trial, cell_type, 
              mat, kmax, reps=3, n=0.05, verbose=False):
    
    from scipy.stats import sem 
    beta = trial.suggest_float('beta', 1e-6, 1e-1, log=True)
    
    MSE = cross_validate(mat, kmax, beta, reps, n, verbose)
    mean_MSE = MSE.groupby('k').mean()
    k_best = int(mean_MSE.idxmin())
    k_1se = int(mean_MSE.index[mean_MSE <= mean_MSE[k_best] + \
        sem(MSE[k_best])][0])
    
    MSE_trial[cell_type, beta] = MSE
    k_1se_trial[cell_type, beta] = k_1se
    print(f'[{cell_type}]: {k_1se=}')
    error = mean_MSE[k_1se]
    return(error)

################################################################################


de = pl.read_csv('results/DE/p400_broad.csv')
lcpm = Pseudobulk('data/pseudobulk/p400_qcd')\
    .drop(Pseudobulk.get_num_DE_hits(de, threshold=0.1)\
            .filter(pl.col.num_hits <= 100)['cell_type'])\
    .filter_obs(pmAD=1)\
    .log_CPM(prior_count=2)

kmax = 20
n_trials = 1
save_name = '_test'; to_r(save_name, 'save_name')
MSE_trial, k_1se_trial = {}, {}
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for idx, (cell_type, (X, obs, var)) in enumerate(lcpm.items()):
        
    gene_mask = var['_index'].is_in(de.filter(
        (pl.col.cell_type == cell_type) & (pl.col.FDR < 0.1))['gene'])
    mat = X.T[gene_mask] 
    mat = normalize_matrix(mat, 'rint')
    
    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(multivariate=True),
        direction='minimize')
    study.optimize(lambda trial: objective(
        trial, MSE_trial, k_1se_trial, cell_type, 
        mat, kmax, reps=3, n=0.05, verbose=True), 
        n_trials=n_trials)

    best_beta = study.best_trial.params.get('beta', 0)
    MSE_final = MSE_trial[cell_type, best_beta]
    k_final = k_1se_trial[cell_type, best_beta]
    
    snmf = nimfa.Snmf(mat, seed="random_vcol",
                    rank=k_final, version='r', beta=best_beta,
                    max_iter=50, n_run=20)
    snmf_fit = snmf()
    
    W = pl.DataFrame(snmf_fit.basis()).explode(pl.all())\
        .rename(lambda col: col.replace('column_', 'S'))\
        .insert_column(0, var.filter(gene_mask)['_index'].alias('gene'))
    H = pl.DataFrame(snmf_fit.coef().T).explode(pl.all())\
        .rename(lambda col: col.replace('column_', 'S'))\
        .insert_column(0, obs['ID'].alias('ID'))
    MSE_final.to_csv(
        f'results/NMF/MSE/p400/{cell_type}_MSE{save_name}.tsv', sep='\t')
    W.write_csv(
        f'results/NMF/factors/p400/{cell_type}_W{save_name}.tsv', 
        separator='\t')
    H.write_csv(
        f'results/NMF/factors/p400/{cell_type}_H{save_name}.tsv', 
        separator='\t')
    
    plot_k_stats(snmf, kmax, cell_type,
                plot_directory='figures/NMF/stats/p400')
    plot_k_MSE(axes, idx, cell_type, 
            MSE_trial, k_1se_trial, MSE_final, best_beta)
    
fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1,
                    top=0.9, wspace=0.4, hspace=0.4)
savefig(f"figures/NMF/MSE/p400/MSE{save_name}.png", dpi=300)

# os.makedirs(f'results/NMF/MSE/p400', exist_ok=True)
# os.makedirs(f'results/NMF/factors/p400', exist_ok=True)
# os.makedirs(f'figures/NMF/stats/p400', exist_ok=True)
# os.makedirs(f'figures/NMF/MSE/p400', exist_ok=True)
# os.makedirs(f'figures/NMF/A/p400', exist_ok=True)

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
        'apoe_genotype', 'tomm40_hap', 'age_death', 'age_first_ad_dx', 'cogdx',
        'ad_reagan', 'gpath', 'amyloid', 'hspath_typ', 'dlbdx', 'tangles',
        'tdp_st4', 'arteriol_scler', 'caa_4gp', 'cvda_4gp2', 'ci_num2_gct',
        'ci_num2_mct', 'tot_cog_res']))
    to_r(meta, 'meta', format='data.frame', rownames=obs['ID'])
    
    W = pl.read_csv(f'results/NMF/factors/p400/{cell_type}_W{save_name}.tsv',
                    separator='\t')
    H = pl.read_csv(f'results/NMF/factors/p400/{cell_type}_H{save_name}.tsv',
                    separator='\t')
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
        })
        # row_order = get_order(seriate(dist(mat), method = "OLO"))
        # col_order = get_order(seriate(dist(t(mat)), method = "OLO"))
        row_order = get_order(seriate(dist(W), method = "OLO"))
        col_order = get_order(seriate(dist(H), method = "OLO"))
        
        create_color_list = function(data, palette) {
            cols = scico(ncol(data), palette = palette)
            col_funs = mapply(function(col, max_val) {
                colorRamp2(c(0, max_val), c("white", col))
            }, cols, apply(data, 2, max), SIMPLIFY = FALSE)
            setNames(col_funs, colnames(data))
        }
        col_list_H = create_color_list(H, "batlow")
        col_list_W = create_color_list(W, "batlow")

        ht = HeatmapAnnotation(
            df = meta,
            simple_anno_size = unit(0.15, "cm"),
            annotation_name_gp = gpar(fontsize = 5),
            show_legend = FALSE)     
        hb = HeatmapAnnotation(
            df = H, col = col_list_H,
            simple_anno_size = unit(0.3, "cm"),
            annotation_name_gp = gpar(fontsize = 8),
            show_legend = FALSE)         
        hr = rowAnnotation(
            df = W, col = col_list_W,
            simple_anno_size = unit(0.3, "cm"),
            annotation_name_gp = gpar(fontsize = 8),
            show_legend = FALSE)
        col_fun = colorRamp2(quantile(mat, probs = c(0.00, 1.00)), 
            hcl_palette = "Batlow", reverse = TRUE)
        file_name = paste0("figures/NMF/A/p400/", cell_type, 
                    save_name, ".png")
        png(file = file_name, width=7, height=7, units="in", res=1200)
        h = Heatmap(
            mat,
            row_order = row_order,
            column_order = col_order,
            cluster_rows = F,
            cluster_columns = F,
            show_row_names = FALSE,
            show_column_names = FALSE,
            #top_annotation = ht,
            bottom_annotation = hb,
            left_annotation = hr,
            col = col_fun,
            name = paste0('p400', "\n", cell_type),
            heatmap_legend_param = list(
                title_gp = gpar(fontsize = 7, fontface = "bold")),
            show_heatmap_legend = TRUE
        )
        draw(h)
        dev.off()
    ''')


with Timer('Volcano plots'):
    for cell_type in de['cell_type'].unique():
        df = pl.read_csv('results/DE/p400_broad.csv')\
            .filter(cell_type=cell_type)
        plot_volcano(df, threshold=0.1, num_top_genes=30,
                     min_distance=0.08,
                     plot_directory='figures/DE/volcano/p400')     




































def nmf_objective(trial, mat, MSE_trial, k_1se_trial, study_name,
                  cell_type, k_max, L1=True, L2=False):
    
    from scipy.stats import sem
    r('''library(RcppML, quietly = TRUE)
    if (is.null(getOption("RcppML.verbose"))) options(RcppML.verbose = TRUE)
    if (is.null(getOption("RcppML.threads"))) options(RcppML.threads = 0)''')

    L1_w = trial.suggest_float('L1_w', 0.001, 0.999, log=False) if L1 else 0
    L1_h = trial.suggest_float('L1_h', 0.001, 0.999, log=False) if L1 else 0
    L2_w = trial.suggest_float('L2_w', 1e-5, 1e-1, log=True) if L2 else 0
    L2_h = trial.suggest_float('L2_h', 1e-5, 1e-1, log=True) if L2 else 0
    to_r(L1_w, "L1_w"); to_r(L1_h, "L1_h")
    to_r(L2_w, "L2_w"); to_r(L2_h, "L2_h")
    
    r('''MSE = RcppML::crossValidate(
        mat, k = seq(1, k_max, 1), 
        L1 = c(L1_w, L1_h), L2 = c(L2_w, L2_h), 
        seed = 0, reps = 3, tol = 1e-2, 
        maxit = .Machine$integer.max)''')
    
    MSE = to_py('MSE', format='pandas').astype({'k': int, 'rep': int})\
        .set_index(['k', 'rep']).squeeze().rename('MSE')
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
            alpha = min(1, sum(best_params.values()) / 4) ** (1/2) 
            ax.plot(mean_MSE.index, mean_MSE.values, color='black', alpha=alpha)
            ax.scatter(k_1se, mean_MSE[k_1se], color='black', s=16, alpha=alpha)
    mean_MSE = MSE_final.groupby('k').mean()
    k_final = k_1se_trial[study_name, cell_type, *best_params.values()]
    ax.plot(mean_MSE.index, mean_MSE.values, color='red')
    ax.scatter(k_final, mean_MSE[k_final], color='red', s=50)
    #ax.set_xticks(ticks=mean_MSE.index)
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


norm = 'rint'
L1 = True
L2 = False
n_trials = 30
k_max = 50; to_r(k_max, 'k_max')
study_name = 'p400'
save_name = '_rint_L1'; to_r(save_name, 'save_name')

MSE_trial, k_1se_trial = {}, {}
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for idx, (cell_type, (X, obs, var)) in enumerate(lcpm.items()):
    gene_mask = var['_index'].is_in(de.filter(
            (pl.col.cell_type == cell_type) & (pl.col.FDR < 0.1))['gene'])
    mat = X.T[gene_mask] 
    mat = normalize_matrix(mat, norm)
    to_r(mat, 'mat', format='matrix')
    
    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(multivariate=True),
        #sampler=optuna.samplers.RandomSampler(),
        direction='minimize')
    study.optimize(lambda trial: nmf_objective(
        trial, mat, MSE_trial, k_1se_trial,\
        study_name, cell_type, k_max, L1=L1, L2=L2), 
        n_trials=n_trials)

    best_params = {param: study.best_trial.params.get(param, 0)
                    for param in ['L1_w', 'L1_h', 'L2_w', 'L2_h']}
    L1_w, L1_h, L2_w, L2_h = best_params.values()
    MSE_final = MSE_trial[study_name, cell_type, *best_params.values()]
    k_select = k_1se_trial[study_name, cell_type, *best_params.values()]
    to_r(k_select, 'k_select')
    to_r(best_params, 'bp')
    r('''
      res = nmf(mat, k = k_select, 
                L1=c(bp$L1_w, bp$L1_h), L2=c(bp$L2_w, bp$L2_h), 
                seed=1:50, tol=1e-5, maxit=.Machine$integer.max)
        H = res@h; W = res@w
    ''')
    W = to_py('W', format='pandas')\
        .set_axis(var.filter(gene_mask)['_index'].to_list())\
        .rename(columns=lambda col: col.replace('nmf', 'S'))
    H = to_py('H', format='pandas').T\
        .set_axis(obs['ID'].to_list())\
        .rename(columns=lambda col: col.replace('nmf', 'S'))

    os.makedirs(f'results/MSE/{study_name}', exist_ok=True)
    os.makedirs(f'results/NMF/{study_name}', exist_ok=True)
    MSE_final.to_csv(
        f'results/MSE/{study_name}/{cell_type}_MSE{save_name}.tsv', sep='\t')
    W.to_csv(f'results/NMF/{study_name}/{cell_type}_W{save_name}.tsv', sep='\t')
    H.to_csv(f'results/NMF/{study_name}/{cell_type}_H{save_name}.tsv', sep='\t')

    plot_MSE(axes, idx, study_name, cell_type,
             MSE_trial, k_1se_trial, MSE_final, best_params)
fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1,
                    top=0.9, wspace=0.4, hspace=0.4)
savefig(f"results/MSE/{study_name}/MSE_plots{save_name}.png", dpi=300)





















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

gene_mask = var['_index'].is_in(de.filter(
    (pl.col.cell_type == cell_type) & (pl.col.FDR < 0.1))['gene'])
mat = X.T[gene_mask] 
mat = normalize_matrix(mat, 'rint')

mask = np.zeros(mat.shape, dtype=bool)
zero_indices = np.random.choice(
    mat.size, round(0.05 * mat.size), replace=False)
mask.flat[zero_indices] = True
mat_masked = mat.copy()
mat_masked.flat[zero_indices] = 0

snmf = nimfa.Snmf(mat_masked, rank=5, version='l',
                    beta=1e-4, max_iter=5, n_run=1)
mat_est = snmf().fitted()
mat_est = np.asarray(mat_est)

from matplotlib.colors import ListedColormap

data1 = mat[0:50, 0:50] 
data2 = mask[0:50, 0:50]
fig, ax = plt.subplots(figsize=(5,5))    
sns.heatmap(data1, cbar=False, xticklabels=False, yticklabels=False,
            rasterized=True)
sns.heatmap(data2, cmap = ListedColormap(['white']), 
            cbar=False, xticklabels=False, yticklabels=False,
            rasterized=True, square=True, mask=~data2)
savefig('tmp2.png')

fig, ax = plt.subplots(figsize=(5,5))    
sns.heatmap(data1, cbar=False, xticklabels=False, yticklabels=False,
            square=True, rasterized=True)
savefig('tmp1.png')

data3 = mat_est[0:50, 0:50]
fig, ax = plt.subplots(figsize=(5,5))    
sns.heatmap(data3, cbar=False, xticklabels=False, yticklabels=False,
            square=True, rasterized=True)
savefig('tmp3.png')

fig, ax = plt.subplots(figsize=(5,5))    
sns.heatmap(data1, cbar=False, xticklabels=False, yticklabels=False,
            rasterized=True)
sns.heatmap(data2, cmap = ListedColormap(['white']), 
            cbar=False, xticklabels=False, yticklabels=False,
            rasterized=True, square=True, mask=data2)
savefig('tmp4.png')

fig, ax = plt.subplots(figsize=(5,5))    
sns.heatmap(data3, cbar=False, xticklabels=False, yticklabels=False,
            rasterized=True)
sns.heatmap(data2, cmap = ListedColormap(['white']), 
            cbar=False, xticklabels=False, yticklabels=False,
            rasterized=True, square=True, mask=data2)
savefig('tmp5.png')


def peek_metagenes(study_name, cell_type):
    W = pd.read_table(f'results/NMF/{study_name}/{cell_type}_W{save_name}.tsv',
                      index_col=0)
    print(pd.DataFrame({col: W.nlargest(20, col).index.tolist()
                        for col in W.columns}))

peek_metagenes('p400','Inhibitory')


 




from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage
col_linkage = linkage(pdist(mat_norm.T), optimal_ordering=True)
row_linkage = linkage(pdist(mat_norm), optimal_ordering=True)

percentiles = np.percentile(mat_norm, [1, 99])
import cmcrameri
sys.setrecursionlimit(100000)
cm = sns.clustermap(mat_norm,
                    cmap='cmc.buda',
                    square=True,
                    col_cluster=False, row_cluster=False,
                    yticklabels=False, xticklabels=False, 
                    #row_linkage=row_linkage, col_linkage=col_linkage,
                    vmin=percentiles[0], vmax=percentiles[1])
#cm.ax_col_dendrogram.set_visible(False)
cm.ax_row_dendrogram .set_visible(False)
savefig("p1.png", dpi=300)
plt.clf()


de_tain = pl.read_csv('limma_voom.tsv.gz', separator='\t')\
    .rename({'p': 'P', 'fdr': 'FDR'})\
    .filter(trait='p400')
Pseudobulk.get_num_DE_hits(de_tain, threshold=0.1)




harmony_labels = pl.concat([
        pl.read_csv(
            'data/single-cell/cell-type-labels/Glut_cell_type_labels.tsv.gz', 
            separator='\t').rename({'': '_index'}),
        pl.read_csv(
            'data/single-cell/cell-type-labels/GABA_cell_type_labels.tsv.gz', 
            separator='\t').rename({'0': '_index'})
        ])\
    .filter(study_name = 'p400')

rosmap_pheno = \
    pl.read_csv('data/single-cell/p400/dataset_978_basic_04-21-2023.csv',
        infer_schema_length=5000)\
    .unique(subset='projid')\
    .drop([col for col in sc.obs.columns if col != 'projid'])

assert sc.obs['projid'].is_in(rosmap_pheno['projid'].unique()).all()
assert harmony_labels['_index'].is_in(sc.obs['_index'].unique()).all()

# temporary filter because harmony labels are missing for ~15k neurons 
sc = sc.filter_obs(
    pl.col('_index').is_in(
        harmony_labels.select('_index').vstack(
            sc.obs.filter(~pl.col('subset').is_in(['CUX2+', 'Inhibitory']))
            .select('_index')).unique()))
sc.obs = sc.obs\
    .with_columns(
        num_cells_total=pl.col.projid.count().over('projid'),
        broad_cell_type=pl.col.subset.replace({
                'OPCs': 'OPC', 
                'CUX2+': 'Excitatory', 
                'Astrocytes': 'Astrocyte', 
                'Microglia': 'Microglia-PVM', 
                'Oligodendrocytes': 'Oligodendrocyte'}))\
    .join(harmony_labels.select(['_index', 'cell_type']), 
          on='_index', how='left')\
    .with_columns(
        fine_cell_type=pl.coalesce(pl.col.cell_type, pl.col.broad_cell_type))\
    .drop('cell_type')\
    .with_columns(pl.col.projid.cast(pl.Int64))\
    .join(rosmap_pheno, on='projid', how='left')

with Timer():    
    pseudobulk = (
        Pseudobulk(
            sc.filter_obs(
                pl.col.broad_cell_type.is_in(['Excitatory','Inhibitory'])),
            ID_column='projid', cell_type_column='broad_cell_type')
        .rename({'broad_cell_type': 'cell_type'}).vstack(
            Pseudobulk(
                sc, ID_column='projid', cell_type_column='fine_cell_type')
            .rename({'fine_cell_type': 'cell_type'}))
        .sort('projid')
    )
    pseudobulk = pseudobulk.filter(pl.col('projid').is_not_null())
    pseudobulk.write_csv(file='data/pseudobulk/p400/p400_pseudobulk.csv')







r('source("project_utils.R")')
de_ruv = {}
for cell_type, (X, obs, var) in pb.items():
    library_size = X.sum(axis=1) * Pseudobulk.calc_norm_factors(X.T)
    obs = obs.with_columns(library_size=np.log2(library_size))
    to_r(X.T, 'X', format='matrix', rownames=var['_index'], colnames=obs['ID'])
    to_r(obs, 'obs', format='df', rownames=obs['ID'])
    to_r(cell_type, "cell_type")
    r('''
        res = diagnose_ruv(
            input = X, meta = obs, contrast = "pmAD", cell_type = cell_type, 
            model = ~ pmAD + age_death + sex + pmi + apoe4_dosage + library_size, 
            min_k = 1, max_k = 10, 
            plot_voom_dir = "figures/voom/p400_ruvseq",
            verbose = 3)
      ''')
    de_ruv.update(to_py('res'))
with open('results/differential-expression/p400_ruvseq.pkl', 'wb') as file:
    pickle.dump(de_ruv, file)

