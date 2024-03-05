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

import os, nimfa, optuna
import polars as pl, pandas as pd, numpy as np
import matplotlib.pylab as plt, seaborn as sns

os.chdir('projects/def-wainberg/karbabi/single-cell-nmf')
from utils import Timer, print_df, debug, savefig, Pseudobulk
from ryp import r, to_py, to_r    
from project_utils import normalize_matrix, \
    plot_volcano, plot_k_MSE, plot_k_stats

debug(third_party=True)

def cross_validate(mat, kmax, beta, reps, n, verbose=False):
    import nimfa    
    res = []
    for rep in range(1, reps + 1):
        mask = np.random.rand(*mat.shape) < (1 - n) 
        mat_masked = mat * mask 
        for k in range(1, kmax + 1):
            snmf = nimfa.Snmf(mat_masked, rank=k, version='l',
                              beta=beta, max_iter=30, n_run=1)
            mat_est = snmf().fitted()
            mat_est = np.asarray(mat_est)
            MSE = np.mean((mat[~mask] - mat_est[~mask]) ** 2)
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

kmax = 15
n_trials = 20
save_name = '_rint_l'; to_r(save_name, 'save_name')
MSE_trial, k_1se_trial = {}, {}
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

with Timer (f'sNFM CV'):
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
                        rank=k_final, version='l', beta=best_beta,
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
        'apoe_genotype', 'tomm40_hap','age_death', 'age_first_ad_dx', 'cogdx',
        'ad_reagan', 'gpath', 'amyloid', 'hspath_typ', 'dlbdx', 'tangles',
        'tdp_st4', 'arteriol_scler', 'caa_4gp', 'cvda_4gp2','ci_num2_gct',
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

























































