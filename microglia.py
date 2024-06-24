import sys, optuna
import polars as pl, polars.selectors as cs
import numpy as np, pandas as pd

sys.path.append('/home/karbabi/projects/def-wainberg/karbabi/utils')
from single_cell import SingleCell, Pseudobulk
from utils import inverse_normal_transform, print_df
from ryp import r, to_r

sys.path.append('/home/karbabi/projects/def-wainberg/karbabi/pseudobulk-nmf')
from sklearn.decomposition._nmf import _initialize_nmf
from sparse_nmf import sparse_nmf

def sparseness_hoyer(x):
    """
    The sparseness of array x is a real number in [0, 1], where sparser array
    has value closer to 1. Sparseness is 1 if the vector contains a single
    nonzero component and is equal to 0 if all components of the vector are 
    the same.
        
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
            W, H = _initialize_nmf(A_masked, rank, init='nndsvd')
            W, H = sparse_nmf(A_masked, rank=rank, spar=spar, W=W, H=H,
                              tol=1e-3, maxiter=np.iinfo(np.int64).max, 
                              verbose=1)
            A_r = W @ H
            MSE = np.mean((A[mask] - A_r[mask]) ** 2)
            sparseness = sparseness_hoyer(H)
            res.append((rep, rank, MSE, sparseness))
            
    results = pd.DataFrame(
        res, columns=['rep', 'rank', 'MSE', 'sparseness'])\
        .set_index(['rank', 'rep'])   
    return results

def objective(trial, MSE_trial, rank_1se_trial, A, rank_max,
              reps=3, n=0.05):
    
    from scipy.stats import sem 
    m = A.shape[0]
    spar_l = (np.sqrt(m) - np.sqrt(m - 1)) / (np.sqrt(m) - 1) + 1e-10
    spar_u = (np.sqrt(m) + np.sqrt(m - 1)) / (np.sqrt(m) - 1) - 1e-10
    spar = trial.suggest_float('spar', spar_l, spar_u, log=True)
    
    results = cross_validate(A, rank_max, spar, reps, n)    
    print(results.groupby('rep'))
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

sc = SingleCell('projects/def-wainberg/single-cell/Green/'
                'Green_qced_labelled.h5ad', num_threads=None)\
    .filter_obs(cell_type_broad='Microglia', pmAD=1)

mic_states = sc.obs.to_dummies('state').group_by('projid').sum()\
    .select('projid', cs.contains('state'))\
    .cast({'projid': pl.String})
    
hvg_sc = sc.drop_var(['highly_variable', 'highly_variable_rank'])\
    .hvg(num_genes=1000)\
    .var.filter(pl.col.highly_variable)['_index']

pb = Pseudobulk('projects/def-wainberg/single-cell/Green/pseudobulk/broad')\
    ['Microglia']\
    .qc(case_control_column=None)\
    .filter_obs(pmAD=1)\
    .with_columns_obs(log_num_cells=pl.col.num_cells.log(base=2))

pb.obs['Microglia'] = pb.obs['Microglia']\
    .join(mic_states, left_on='ID', right_on='projid', coalesce=True)\
    .with_columns(cs.contains('state') / pl.col.num_cells)            

pb = pb.filter_var(pl.col._index.is_in(hvg_sc))
pb = pb.CPM()
pb = pb.regress_out_obs(covariate_columns=['pmi', 'log_num_cells'])

A = pb.X['Microglia'].T
A = np.apply_along_axis(inverse_normal_transform, 1, A)
A += abs(np.min(A))

MSE_trial, rank_1se_trial = {}, {}
study = optuna.create_study(
    sampler=optuna.samplers.TPESampler(multivariate=True),
    direction='minimize')
study.optimize(lambda trial: objective(
    trial, MSE_trial, rank_1se_trial, A, rank_max=10), 
    n_trials=20)

spar_select = study.best_trial.params.get('spar')
rank_select = rank_1se_trial[spar_select]

genes = pb.var['Microglia']['_index']
samps = pb.obs['Microglia']['ID']

W, H = _initialize_nmf(A, n_components=rank_select, init='nndsvd')
W, H = sparse_nmf(A, rank=rank_select, spar=spar_select, W=W, H=H,
                  tol=1e-6, maxiter=np.iinfo('int32').max, verbose=2)

W = pl.DataFrame(W)\
    .rename(lambda col: col.replace('column_', 'S'))\
    .insert_column(0, pl.Series(genes).alias('gene'))
H = pl.DataFrame(H.T)\
    .rename(lambda col: col.replace('column_', 'S'))\
    .insert_column(0, pl.Series(samps).alias('ID'))

to_r(A, 'A', format='matrix', rownames=genes)
to_r(W.drop('gene'), 'W', rownames=W['gene'])
to_r(H.drop('ID'), 'H', rownames=H['ID'])
meta = pb.obs['Microglia']
to_r(meta, 'meta')

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

hr2 = rowAnnotation( 
    df = W, col = col_list_W,
    simple_anno_size = unit(0.3, "cm"),
    annotation_name_gp = gpar(fontsize = 8),
    annotation_name_side = "bottom",
    show_legend = FALSE)

hb = HeatmapAnnotation(
    df = H, col = col_list_H,
    simple_anno_size = unit(0.3, "cm"),
    annotation_name_gp = gpar(fontsize = 8),
    annotation_name_side = "left",    
    show_legend = FALSE)         

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
    bottom_annotation = hb,
    right_annotation = hr2,
    col = col_fun,
    name = "normalized\nexpression",
    heatmap_legend_param = list(
        title_gp = gpar(fontsize = 7, fontface = "bold"),
        labels_gp = gpar(fontsize = 5)),
    show_heatmap_legend = TRUE,
    use_raster = FALSE
)
file_name = paste0("projects/def-wainberg/karbabi/pseudobulk-nmf/A_tmp.png")
png(file = file_name, width=7, height=6, units="in", res=1200)
draw(h)
dev.off()
''')

r('''
library(tidyverse)
library(corrplot)
metadata = meta %>%
    select(ID, num_cells, sex, braaksc, ceradsc, pmi, niareagansc, 
            apoe4_dosage, tomm40_hap, age_death, age_first_ad_dx, gpath,
            amyloid, hspath_typ, dlbdx, tangles, tdp_st4, arteriol_scler,
            caa_4gp, cvda_4gp2, ci_num2_gct, ci_num2_mct, tot_cog_res, 
            state_Mic.1, state_Mic.10, state_Mic.11, state_Mic.12,
            state_Mic.13, state_Mic.14, state_Mic.15, state_Mic.16,
            state_Mic.2, state_Mic.3, state_Mic.4, state_Mic.5, 
            state_Mic.6, state_Mic.7, state_Mic.8, state_Mic.9) %>%
    rename(c("Number of cells" = num_cells,  
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
            "Total cognitive reserve" = tot_cog_res)) %>%
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

png(paste0("projects/def-wainberg/karbabi/pseudobulk-nmf/cor_tmp.png"), 
    width = 6, height = 9, units = "in", res=300)
corrplot(cor_mat, is.corr = FALSE,  
        p.mat = p_mat, sig.level = 0.05, 
        insig = 'label_sig', pch.cex = 2, pch.col = "white",
        tl.col = "black")
dev.off() 
''')









