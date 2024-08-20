import sys, gc, optuna, pickle
import polars as pl, polars.selectors as cs
import numpy as np, pandas as pd
from tabulate import tabulate

sys.path.append('/home/karbabi/projects/def-wainberg/karbabi/utils')
from single_cell import SingleCell, Pseudobulk, DE
from utils import inverse_normal_transform, get_coding_genes, \
    debug, print_df, savefig
from ryp import r, to_r, to_py

sys.path.append('/home/karbabi/projects/def-wainberg/karbabi/pseudobulk-nmf')
from sklearn.decomposition._nmf import _initialize_nmf
from sparse_nmf import sparse_nmf

debug(third_party=True)

data_dir = 'projects/def-wainberg/single-cell'
working_dir = 'projects/def-wainberg/karbabi/pseudobulk-nmf' 

def sparseness_hoyer(x):
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

def initialize_flat(A, rank):
    W = np.full((A.shape[0], rank), 1e-8)
    H = np.full((rank, A.shape[1]), 1e-8)
    return W, H

def cross_validate(A, rank_max, init_type, spar, reps, n):
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
            if init_type == 'nndsvd':
                W, H = _initialize_nmf(A_masked, rank, init='nndsvd', 
                                       random_state=rep*rank_max+rank-1)
            elif init_type == 'flat':
                W, H = initialize_flat(A_masked, rank)
                
            W, H = sparse_nmf(A_masked, rank=rank, spar=spar, W=W, H=H,
                              tol=1e-2, maxiter=np.iinfo(np.int32).max, 
                              verbose=0)
            A_r = W @ H
            MSE = np.mean((A[mask] - A_r[mask]) ** 2)
            res.append((rep, rank, MSE))
            
    results = pd.DataFrame(res, columns=['rep', 'rank', 'MSE'])\
        .set_index(['rank', 'rep'])   
    return results

def objective(trial, MSE_trial, rank_1se_trial, A, rank_max, 
              init_type, reps, n):
    
    from scipy.stats import sem 
    m = A.shape[0]
    spar_l = (np.sqrt(m) - np.sqrt(m - 1)) / (np.sqrt(m) - 1) + 1e-6
    spar_u = (np.sqrt(m) + np.sqrt(m - 1)) / (np.sqrt(m) - 1) - 1e-6
    spar = trial.suggest_float('spar', spar_l, spar_u, log=True)
    
    results = cross_validate(A, rank_max, init_type, spar, reps, n)    

    print(results.groupby('rep'))
    mean_MSE = results.groupby('rank').mean()['MSE']
    rank_best = int(mean_MSE.idxmin())
    rank_1se = int(mean_MSE.index[mean_MSE <= mean_MSE[rank_best] + \
        sem(results.loc[rank_best, 'MSE'])][0])
    
    MSE_trial[spar] = mean_MSE
    rank_1se_trial[spar] = rank_1se
    print(f'{rank_1se=}')
    MSE = mean_MSE[rank_1se]
    return MSE

def plot_MSE_trial(MSE_trial, rank_1se_trial, spar_select, rank_select,
                   filename):
    import matplotlib.pylab as plt
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
    lower, upper = np.quantile(MSE_values, [0, 0.8])
    ax.set_ylim(bottom=lower-0.05, top=6)
    ax.plot(mean_MSE_select.index, mean_MSE_select.values, 
            linewidth = 3, color='red')
    ax.scatter(rank_select, mean_MSE_select[rank_select], color='red', s=80)
    ax.set_xticks(ticks=mean_MSE_select.index)
    ax.set_yscale('log')
    ax.set_title(rf'$\mathbf{{MSE\ across\ Optuna\ trials}}$'
                + f"\n Best rank: {rank_select}, "
                + f"Best spar: {spar_select:.2g}, Min spar: {spar_min:.2g}")
    ax.set_xlabel('Rank', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean MSE', fontsize=12, fontweight='bold')
    fig.subplots_adjust(bottom=0.1, top=0.9, hspace=0.3, wspace=0.25)
    savefig(filename, dpi=300)

################################################################################

sc = SingleCell(f'{data_dir}/Green/Green_qced_labelled.h5ad', 
                num_threads=None)\
    .filter_obs(pl.col.state.cast(pl.String).str.contains('Mic'))

pb = sc.pseudobulk(
        ID_column='projid', 
        cell_type_column='state',
        num_threads=None)\
    .filter_var(
        pl.col._index.is_in(get_coding_genes()['gene']))\
    .with_columns_obs(
        dx_cc=pl.col.pmAD,
        apoe4_dosage=pl.col.apoe_genotype.cast(pl.String)
            .str.count_matches('4').fill_null(strategy='mean'),
        pmi=pl.col.pmi.fill_null(strategy='mean'))

min_people = round(0.8 * min(
    max(pb.obs.items(), key=lambda x: x[1].height)[1]
    .filter(pl.col.dx_cc.is_not_null())['dx_cc']
    .value_counts()['count']))
drop_cell_types = [
    cell_type for cell_type, (_, obs, _) in pb.items()
    if obs.filter(pl.col.dx_cc.is_not_null())['dx_cc']
    .value_counts()['count'].min() < min_people]
pb = pb.drop_cell_types(drop_cell_types)

de = pb.qc(case_control_column='dx_cc',
           custom_filter=pl.col.dx_cc.is_not_null(),
           verbose=False)\
    .DE(label_column='dx_cc', 
        case_control=True,
        covariate_columns=['age_death', 'sex', 'pmi', 'apoe4_dosage'])

del sc; gc.collect()

sc = SingleCell(f'{data_dir}/Green/Green_qced_labelled.h5ad', 
                num_threads=None)\
    .filter_obs(cell_type_broad='Microglia', pmAD=1)

mic_states = sc.obs.to_dummies('state').group_by('projid').sum()\
    .select('projid', cs.contains('state'))\
    .with_columns(pl.exclude('projid').add(1).log(2))\
    .cast({'projid': pl.String})    

pb = Pseudobulk(f'{data_dir}/Green/pseudobulk/broad')['Microglia']\
    .qc(case_control_column='dx_cc',
        custom_filter=pl.col.dx_cc.is_not_null())\
    .with_columns_obs(log_num_cells=pl.col.num_cells.log(base=2))

to_r(pb.X['Microglia'].T, 'counts', format='matrix', 
     rownames=pb.var['Microglia']['_index'], 
     colnames=pb.obs['Microglia']['ID'])
to_r(pb.obs['Microglia']
     [['dx_cc', 'age_death', 'sex', 'pmi', 'apoe4_dosage', 'log_num_cells']],
    'meta', format='data.frame')

r('''
library(limma)
dge = DGEList(counts)
dge = calcNormFactors(dge)
meta$log_library_size = log2(dge$samples$lib.size * dge$samples$norm.factors)
meta$dx_cc = factor(meta$dx_cc, levels = c(0, 1))
meta$sex = factor(as.character(meta$sex))
design = model.matrix(~ dx_cc + sex + pmi + apoe4_dosage + log_num_cells +
  log_library_size, meta) 
v = voom(counts, design, plot=FALSE)
fit = lmFit(v)
eb = eBayes(fit)
summary(decideTests(eb, p.value=0.10))
gene_variances = eb$s2.post
ranked_genes = rownames(counts)[order(gene_variances, decreasing=TRUE)]
genes_hvg_bulk_ebayes = ranked_genes
''')

pb = Pseudobulk(f'{data_dir}/Green/pseudobulk/broad')['Microglia']\
    .qc(case_control_column=None,
        custom_filter=pl.col.dx_cc.is_not_null() & pl.col.dx_cc.eq(1))\
    .with_columns_obs(log_num_cells=pl.col.num_cells.log(base=2))\
    .join_obs(mic_states, left_on='ID', right_on='projid')\
    .with_columns_obs(cs.contains('state') / pl.col.num_cells)  
genes_filt = pb.var['Microglia']['_index'].to_list()

genes_hvg_bulk_ebayes = to_py('genes_hvg_bulk_ebayes').to_list()[0:5000]

r('''
suppressPackageStartupMessages(library(DESeq2))
dds = DESeqDataSetFromMatrix(
    counts, colData = DataFrame(row.names=colnames(counts)), 
    design = ~ 1)
vsd = vst(dds)
row_variances = rowVars(assay(vsd))
genes_hvg_bulk_vst = rownames(vsd)[
    order(row_variances, decreasing = TRUE)]
''')

genes_hvg_bulk_vst = to_py('genes_hvg_bulk_vst').to_list()[0:5000]

genes_hvg_sc = sc.drop_var(['highly_variable', 'highly_variable_rank'])\
    .hvg(num_genes=10000).var\
    .filter(pl.col._index.is_in(genes_filt))\
    .filter(pl.col.highly_variable_rank.is_not_null())\
    .rename({'_index': 'gene', 'highly_variable_rank': 'rank'})\
    .sort('rank')['gene'].to_list()[0:5000]
    
genes_de_broad = DE(f'{working_dir}/output/DE/Green_broad_dx_cc').table\
    .filter(pl.col.cell_type.eq('Microglia'))\
    .sort('P')['gene'].to_list()[0:5000]

genes_de_fine = de.table\
    .group_by('gene').agg(pl.all().sort_by('P').first())\
    .sort('P')['gene'].to_list()[0:5000]

#'sex', 'braaksc', 
clin_covars = [
    'ceradsc',  'niareagansc', 'apoe4_dosage', 'tomm40_hap', 'age_death', 
    'age_first_ad_dx', 'gpath', 'amyloid', 'hspath_typ', 'dlbdx', 'tangles', 
    'tdp_st4', 'arteriol_scler', 'caa_4gp', 'cvda_4gp2', 'ci_num2_gct',
    'ci_num2_mct', 'tot_cog_res', 'bmi', 'hypertension_cum', 'sbp_avg', 
    'heart_cum', 'stroke_cum', 'cardiac_rx']
tec_covars = [
    'pmi', 'log_num_cells', 'state_Mic.1', 'state_Mic.2', 'state_Mic.3', 
    'state_Mic.4', 'state_Mic.5', 'state_Mic.6', 'state_Mic.7', 'state_Mic.8', 
    'state_Mic.9', 'state_Mic.10', 'state_Mic.11', 'state_Mic.12', 
    'state_Mic.13', 'state_Mic.14', 'state_Mic.15', 'state_Mic.16']
genesets = {
    'hvg_bulk_voom': genes_hvg_bulk_ebayes, 'hvg_bulk_vst': genes_hvg_bulk_vst,
    'hvg_sc': genes_hvg_sc, 'de_broad': genes_de_broad, 
    'de_fine': genes_de_fine}
top_gene_nums = [50, 200, 800, 5000]
init_types = ['flat', 'nndsvd', 'nndsvd_reg']

headers = ['geneset', 'top_num', 'init_type', 'rank_select', 'MSE', 
           'H_sparseness', 'W_sparseness', 'clin_cor_n', 'clin_cor_avg', 
           'tec_cor_n', 'tec_cor_avg']

res = {}; stats = []
for geneset in genesets.keys():
    for init_type in init_types:
        for top_num in top_gene_nums:
            genes = genesets[geneset][0:top_num]
            lcpm = pb.filter_var(pl.col._index.is_in(genes)).log_CPM()
            if init_type == 'nndsvd_reg':
                lcpm = lcpm.regress_out_obs(
                    covariate_columns=['pmi', 'log_num_cells'])
                init_type = 'nndsvd'

            A = lcpm.X['Microglia'].T
            A = np.apply_along_axis(inverse_normal_transform, 1, A)
            A += abs(np.min(A))
            
            MSE_trial, rank_1se_trial = {}, {}
            sampler = optuna.samplers.TPESampler(
                multivariate=True, seed=4) 
            study = optuna.create_study(
                sampler=sampler, direction='minimize')
            study.optimize(lambda trial: objective(
                trial, MSE_trial, rank_1se_trial, A, rank_max=15,
                init_type=init_type, reps=3, n=0.05), 
                n_trials=10)
            
            spar_select = study.best_trial.params.get('spar')
            rank_select = rank_1se_trial[spar_select]
            MSE = MSE_trial[spar_select][rank_select]

            if init_type == 'nndsvd':
                W, H = _initialize_nmf(A, rank_select, init='nndsvd', 
                                       random_state=0)
            elif init_type == 'flat':
                W, H = initialize_flat(A, rank_select)
                
            W, H = sparse_nmf(
                A, rank=rank_select, spar=spar_select, W=W, H=H,
                tol=1e-6, maxiter=np.iinfo(np.int32).max, 
                verbose=2)

            res[geneset, top_num, init_type] = A, W, H, \
                rank_select, spar_select, MSE_trial, rank_1se_trial
            H_sparseness = sparseness_hoyer(H)
            W_sparseness = sparseness_hoyer(W)

            significant_cors_clin = 0.0
            significant_cors_tec = 0.0
            sum_cors_clin = 0
            sum_cors_tec = 0

            for covars in [clin_covars, tec_covars]:
                from scipy.stats import pearsonr
                obs = pb.obs['Microglia']\
                    .select(covars)\
                    .with_columns(pl.all().fill_null(strategy='mean'))
                
                for i, row in enumerate(H):
                    for col in obs.columns:
                        col_data = obs.get_column(col).to_numpy()
                        corr, p_value = pearsonr(row, col_data)
                        if p_value < 0.05:
                            if covars == clin_covars:
                                significant_cors_clin += 1
                                sum_cors_clin += abs(corr) 
                            else:
                                significant_cors_tec += 1
                                sum_cors_tec += abs(corr)  

            avg_cor_clin = (sum_cors_clin / significant_cors_clin 
                            if significant_cors_clin > 0 else 0)
            avg_cor_tec = (sum_cors_tec / significant_cors_tec 
                            if significant_cors_tec > 0 else 0)
            stats.append((
                geneset,
                len(genes),
                init_type,
                rank_select,
                float(f"{MSE:.4g}"),
                float(f"{H_sparseness:.4g}"),
                float(f"{W_sparseness:.4g}"),
                significant_cors_clin,
                float(f"{avg_cor_clin:.4g}"),
                significant_cors_tec,
                float(f"{avg_cor_tec:.4g}")
            ))
            print(tabulate(
                stats, headers, tablefmt="presto", numalign="right"))

stats_df = pd.DataFrame(stats, columns=headers)
stats_df.to_csv(f'{working_dir}/output/NMF/stats.csv')

# with open(f'{working_dir}/output/NMF/res.pkl', 'wb') as file:
#     pickle.dump(res, file)

with open(f'{working_dir}/output/NMF/res.pkl', 'rb') as file:
    res = pickle.load(file)

A, W, H, rank_select, spar_select, MSE_trial, rank_1se_trial = \
    res['hvg_sc', 200, 'nndsvd']

genes = genesets['hvg_sc'][0:200]
samps = pb.obs['Microglia']['ID']

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
suppressPackageStartupMessages({
    library(tidyverse)
    library(corrplot)
})
metadata = meta %>%
    select(ID, num_cells, sex, braaksc, ceradsc, pmi, niareagansc, 
            apoe4_dosage, tomm40_hap, age_death, age_first_ad_dx, gpath,
            amyloid, hspath_typ, dlbdx, tangles, tdp_st4, arteriol_scler,
            caa_4gp, cvda_4gp2, ci_num2_gct, ci_num2_mct, tot_cog_res, 
            cogn_global, bmi, hypertension_cum, sbp_avg, heart_cum, stroke_cum
    ) %>%
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

png(paste0("projects/def-wainberg/karbabi/pseudobulk-nmf/cor_tmp.png"), 
    width = 5, height = 9, units = "in", res=300)
corrplot(cor_mat, is.corr = FALSE,  
        p.mat = p_mat, sig.level = 0.05, 
        insig = 'label_sig', pch.cex = 2, pch.col = "white",
        tl.col = "black")
dev.off() 
''')





            # state_Mic.1, state_Mic.10, state_Mic.11, state_Mic.12,
            # state_Mic.13, state_Mic.14, state_Mic.15, state_Mic.16,
            # state_Mic.2, state_Mic.3, state_Mic.4, state_Mic.5, 
            # state_Mic.6, state_Mic.7, state_Mic.8, state_Mic.9




print_df(pl.DataFrame({
    "Column Name": obs.columns,
    "Data Type": [str(dtype) for dtype in obs.dtypes]
}))
