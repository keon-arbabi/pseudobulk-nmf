import sys, optuna, pickle
import polars as pl, polars.selectors as cs
import numpy as np, pandas as pd

sys.path.append('/home/karbabi/projects/def-wainberg/karbabi/utils')
from single_cell import SingleCell, Pseudobulk, DE
from utils import inverse_normal_transform, debug, print_df, savefig
from ryp import r, to_r

sys.path.append('/home/karbabi/projects/def-wainberg/karbabi/pseudobulk-nmf')
from sklearn.decomposition._nmf import _initialize_nmf
from sparse_nmf import sparse_nmf

debug(third_party=True)

data_dir = 'projects/def-wainberg/single-cell'
working_dir = 'projects/def-wainberg/karbabi/pseudobulk-nmf' 

def initialize_flat(A, rank):
    W = np.full((A.shape[0], rank), 1e-8)
    H = np.full((rank, A.shape[1]), 1e-8)
    return W, H

def cross_validate(A, rank_max, spar, reps, n):
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
            W, H = _initialize_nmf(
                A_masked, rank, init='nndsvd', 
                random_state=rep*rank_max+rank-1)
            # W, H = initialize_flat(A_masked, rank)
            
            W, H = sparse_nmf(
                A_masked, rank=rank, spar=spar, W=W, H=H,
                tol=1e-2, maxiter=np.iinfo(np.int32).max, verbose=1)
            A_r = W @ H
            MSE = np.mean((A[mask] - A_r[mask]) ** 2)
            res.append((rep, rank, MSE))

    results = pd.DataFrame(res, columns=['rep', 'rank', 'MSE'])\
        .set_index(['rank', 'rep'])   
    return results

def objective(trial, MSE_trial, rank_1se_trial, A, rank_max, reps, n):
    from scipy.stats import sem 
    import pandas as pd
    import numpy as np
    
    m = A.shape[0]
    spar_l = (np.sqrt(m) - np.sqrt(m - 1)) / (np.sqrt(m) - 1) + 1e-6
    spar_u = (np.sqrt(m) + np.sqrt(m - 1)) / (np.sqrt(m) - 1) - 1e-6
    spar = trial.suggest_float('spar', spar_l, spar_u, log=True)
    
    res = cross_validate(A, rank_max, spar, reps, n)    

    grouped = res.groupby('rank')
    results = pd.DataFrame({
        'Mean_MSE': grouped['MSE'].mean(),
        'SE_MSE': grouped['MSE'].sem()})
    rank_best = results['Mean_MSE'].idxmin()
    
    threshold = results.loc[rank_best, 'Mean_MSE'] + \
        sem(res.loc[rank_best, 'MSE'])
    rank_1se = results.index[results['Mean_MSE'] <= threshold].min()
    print(f'{rank_1se=}')

    MSE_trial[spar] = results
    rank_1se_trial[spar] = rank_1se
    MSE = results.loc[rank_1se, 'Mean_MSE']
    return MSE

def plot_MSE_trial(MSE_trial, rank_1se_trial, spar_select, rank_select,
                   filename):
    import matplotlib.pylab as plt
    fig, ax = plt.subplots(figsize=(8, 7)) 
    MSE_values = []
    spar_min = min(MSE_trial.keys())
    for current_spar, df in MSE_trial.items():
        MSE_values.extend(df['Mean_MSE'].values)
        rank_1se = rank_1se_trial[current_spar]
        ax.plot(df.index, df['Mean_MSE'], color='black', alpha=0.1)
        ax.scatter(rank_1se, df.loc[rank_1se, 'Mean_MSE'], 
                   color='black', s=16, alpha=0.1)
    df_select = MSE_trial[spar_select]
    rank_select = rank_1se_trial[spar_select]
    lower, upper = np.quantile(MSE_values, [0, 0.8])
    ax.set_ylim(bottom=lower-0.05, top=upper)
    ax.plot(df_select.index, df_select['Mean_MSE'], 
            linewidth=1, color='red', alpha=1)
    ax.fill_between(df_select.index, 
                    df_select['Mean_MSE'] - df_select['SE_MSE'],
                    df_select['Mean_MSE'] + df_select['SE_MSE'],
                    color='red', alpha=0.4)
    ax.scatter(rank_select, df_select.loc[rank_select, 'Mean_MSE'],
               color='red', s=80)
    ax.set_xticks(ticks=df_select.index)
    ax.set_yscale('log')
    ax.set_title(rf'$\mathbf{{MSE\ across\ Optuna\ trials}}$'
                + f"\n Best rank: {rank_select}, "
                + f"Best spar: {spar_select:.2g}, Min spar: {spar_min:.2g}")
    ax.set_xlabel('Rank', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean MSE', fontsize=12, fontweight='bold')
    fig.subplots_adjust(bottom=0.1, top=0.9, hspace=0.3, wspace=0.25)
    savefig(filename)

################################################################################

pb = Pseudobulk(f'{data_dir}/Green/pseudobulk/broad')['Microglia']\
    .qc(case_control_column='dx_cc',
        custom_filter=pl.col.dx_cc.is_not_null())
X = pb.X['Microglia']
log_library_size = np.log2(X.sum(axis=1) * pb._calc_norm_factors(X.T))

genes_de = DE(f'{working_dir}/output/DE/Green_broad_dx_cc').table\
    .filter(pl.col.cell_type.eq('Microglia') & pl.col.P.le(0.05))['gene']

pb = pb.filter_var(pl.col._index.is_in(genes_de))
pb = pb.with_columns_obs(
        log_num_cells=pl.col.num_cells.log(base=2),
        log_library_size=log_library_size)\
    .log_CPM()\
    .regress_out_obs(covariate_columns=[
        'pmi', 'log_num_cells', 'log_library_size'])

A = pb.X['Microglia'].T
A = np.apply_along_axis(inverse_normal_transform, 1, A)
A += abs(np.min(A))

MSE_trial, rank_1se_trial = {}, {}
sampler = optuna.samplers.TPESampler(
    multivariate=True, seed=0) 
study = optuna.create_study(
    sampler=sampler, direction='minimize')
study.optimize(lambda trial: objective(
    trial, MSE_trial, rank_1se_trial, A, rank_max=30, reps=3, n=0.05), 
    n_trials=20)

with open(f'{working_dir}/output/NMF/trial_degs.pkl', 'wb') as file:
    pickle.dump((study, MSE_trial, rank_1se_trial), file)

with open(f'{working_dir}/output/NMF/trial_degs.pkl', 'rb') as file:
    study, MSE_trial, rank_1se_trial = pickle.load(file)

spar_select = study.best_trial.params.get('spar')
rank_select = rank_1se_trial[spar_select]
MSE = MSE_trial[spar_select].loc[rank_select, 'Mean_MSE']
plot_MSE_trial(MSE_trial, rank_1se_trial, spar_select, rank_select,
               filename=f'{working_dir}/figures/MSE_plot.png')

W, H = _initialize_nmf(A, rank_select, init='nndsvd', random_state=0)
# W, H = initialize_flat(A, rank_select)
    
W, H = sparse_nmf(
    A, rank=rank_select, spar=spar_select, W=W, H=H,
    tol=1e-6, maxiter=np.iinfo(np.int32).max, 
    verbose=2)

genes = pb.var['Microglia']['_index']
samps = pb.obs['Microglia']['ID']

from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist
row_order = leaves_list(linkage(pdist(A), 'average', optimal_ordering=False)) + 1
col_order = leaves_list(linkage(pdist(A.T), 'average', optimal_ordering=False)) + 1

W = pl.DataFrame(W)\
    .rename(lambda col: col.replace('column_', 'S'))\
    .insert_column(0, pl.Series(genes).alias('gene'))
H = pl.DataFrame(H.T)\
    .rename(lambda col: col.replace('column_', 'S'))\
    .insert_column(0, pl.Series(samps).alias('ID'))

meta = pb.obs['Microglia']

to_r(A, 'A', format='matrix', rownames=genes)
to_r(W.drop('gene'), 'W', rownames=W['gene'])
to_r(H.drop('ID'), 'H', rownames=H['ID'])
to_r(row_order, 'row_order')
to_r(col_order, 'col_order')
to_r(meta, 'meta')

r('''
suppressPackageStartupMessages({
    library(ComplexHeatmap)
    library(circlize)
    library(seriation)
    library(scico)
    library(ggsci)
    library(tidyverse)
})
  
mat = A
create_color_list = function(data, palette) {
    n = ncol(data)
    cols = scico(n+1, palette = palette)[1:n]
    col_funs = mapply(function(col, max_val) {
        colorRamp2(c(0, max_val), c("white", col))
        }, cols, apply(data, 2, max), SIMPLIFY = FALSE)
    setNames(col_funs, colnames(data))
}
col_list_H = create_color_list(H, "batlow")
col_list_W = create_color_list(W, "batlow")

hr = rowAnnotation( 
    df = W, col = col_list_W,
    simple_anno_size = unit(0.3, "cm"),
    annotation_name_gp = gpar(fontsize = 8),
    annotation_name_side = "bottom",
    show_legend = FALSE
)
hb = HeatmapAnnotation(
    df = H, col = col_list_H,
    simple_anno_size = unit(0.3, "cm"),
    annotation_name_gp = gpar(fontsize = 8),
    annotation_name_side = "left",    
    show_legend = FALSE
)     
ht = HeatmapAnnotation(
    df = meta %>%
        dplyr::select(dx_cc, dx_cont, braaksc, ceradsc) %>%
        dplyr::mutate(braaksc = as.numeric(braaksc)),
    simple_anno_size = unit(0.3, "cm"),
    annotation_name_gp = gpar(fontsize = 8),
    show_legend = FALSE
)    
col_fun = colorRamp2(quantile(mat, probs = c(0.00, 1.00)), 
    hcl_palette = "Batlow", reverse = TRUE
)  
h = Heatmap(
    mat,
    row_order = row_order,
    column_order = col_order,
    cluster_rows = F,
    cluster_columns = F,
    show_row_names = FALSE,
    show_column_names = FALSE,
    bottom_annotation = hb,
    right_annotation = hr,
    top_annotation = ht,
    col = col_fun,
    name = "normalized\nexpression",
    heatmap_legend_param = list(
        title_gp = gpar(fontsize = 7, fontface = "bold"),
        labels_gp = gpar(fontsize = 5)),
    show_heatmap_legend = TRUE,
    use_raster = FALSE
)
file_name = paste0("projects/def-wainberg/karbabi/pseudobulk-nmf/figures/A.png")
png(file = file_name, width=7, height=6, units="in", res=1200)
draw(h)
dev.off()
''')

r('''
suppressPackageStartupMessages({
    library(tidyverse)
    library(seriation)
    library(corrplot)
})
metadata = meta %>%
    select(ID, dx_cc, dx_cont,
        num_cells, log_num_cells, log_library_size,
        sex, braaksc, ceradsc, pmi, niareagansc, 
        apoe4_dosage, tomm40_hap, age_death, age_first_ad_dx, gpath,
        amyloid, hspath_typ, dlbdx, tangles, tdp_st4, arteriol_scler,
        caa_4gp, cvda_4gp2, ci_num2_gct, ci_num2_mct, tot_cog_res, 
        cogn_global, bmi, hypertension_cum, sbp_avg, heart_cum, stroke_cum
    ) %>%
    rename(c("Number of cells" = num_cells,
        "Log number of cells" = log_num_cells,
        "Log library size" = log_library_size,  
        "Sex" = sex, "Braak stage" = braaksc, "Cerad score" = ceradsc, 
        "PMI" = pmi, "NIA-Reagan diagnosis" = niareagansc, 
        "APOE4 dosage" = apoe4_dosage, "TOMM40 haplotype" = tomm40_hap, 
        "Age of death" = age_death, "Age of diagnosis" = age_first_ad_dx,
        "Global AD pathology" = gpath, "Amyloid level" = amyloid, 
        "Hippocampal sclerosis" = hspath_typ, "Lewy body disease" = dlbdx, 
        "Neurofibrillary tangles" = tangles, "TDP-43 IHC" = tdp_st4, 
        "Arteriolosclerosis" = arteriol_scler,
        "Cerebral amyloid angiopathy" = caa_4gp, 
        "Cerebral atherosclerosis" = cvda_4gp2, 
        "Chronic infarcts" = ci_num2_gct,
        "Chronic microinfarcts" = ci_num2_mct,
        "Total cognitive reserve" = tot_cog_res,
        "Global congitive function" = cogn_global,
        "BMI" = bmi, "History of hypertension" = hypertension_cum,
        "History of heart condition" = heart_cum,
        "History of stroke" = stroke_cum)) %>%
    mutate(across(where(is.factor), as.numeric)) %>%
    mutate(across(where(is.numeric), 
            ~ifelse(is.na(.), median(., na.rm = TRUE), .))) %>%
    mutate(`Cerad score` = rev(`Cerad score`), 
           `NIA-Reagan diagnosis` = rev(`NIA-Reagan diagnosis`)) %>%
    column_to_rownames(var = "ID")

cor_mat = t(cor(H, metadata))
p_mat = matrix(NA, ncol(H), ncol(metadata))
for (i in 1:ncol(H)) {
    for (j in 1:ncol(metadata)) {
        p_mat[i, j] = cor.test(H[, i], metadata[, j])$p.value
    }
}
p_mat = t(p_mat)
row_order = get_order(seriate(dist(cor_mat), method = "OLO"))
row_order = order(rownames(cor_mat))

cor_mat = cor_mat[as.numeric(row_order),]
p_mat = p_mat[as.numeric(row_order),]
rownames(p_mat) = rownames(cor_mat)
colnames(p_mat) = colnames(cor_mat)

png(paste0("projects/def-wainberg/karbabi/pseudobulk-nmf/figures/cor.png"), 
    width = 5, height = 9, units = "in", res=300)
corrplot(cor_mat, is.corr = FALSE,  
        p.mat = p_mat, sig.level = 0.05, 
        insig = 'label_sig', pch.cex = 2, pch.col = "white",
        tl.col = "black")
dev.off() 
''')



import polars as pl
sys.path.append('/home/karbabi/projects/def-wainberg/karbabi/utils')
from single_cell import SingleCell
from utils import print_df

sc = SingleCell('projects/def-wainberg/single-cell/SEAAD/'
                'Reference_MTG_RNAseq_final-nuclei.2022-06-07.h5ad',
                num_threads=None)

print_df(
    pl.DataFrame({'Exp': sc[:, 'MALAT1'].X.toarray().flatten(),
                  'cell_type': sc.obs['subclass_label']})\
    .group_by('cell_type')\
    .agg([pl.col.Exp.count().alias('total'),
          pl.col.Exp.ne(0).sum().alias('non_zero')])
)


import numpy as np
from scipy.sparse import csr_array

result_sc = sc.pipe_X(
    lambda X, obs, var, gene, group_col: csr_array(
        obs.group_by(group_col).transform(
            lambda g: (X[:, var.index.get_loc(gene)][g.index].data != 0).sum() / len(g) * 100
        ).to_numpy().reshape(-1, 1)
    ),
    obs=sc.obs,
    var=sc.var,
    gene='MALAT1',
    group_col='subclass_label'
)