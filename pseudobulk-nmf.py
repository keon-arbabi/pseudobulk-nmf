import sys, os, gc
import polars as pl, numpy as np

sys.path.append('/home/karbabi/projects/def-wainberg/karbabi/utils')
from single_cell import SingleCell, Pseudobulk
from utils import Timer, print_df, get_coding_genes, debug
    
os.chdir('projects/def-wainberg/karbabi/pseudobulk-nmf')
debug(third_party=True)

# load single cell data ########################################################

os.makedirs('output/pseudobulk/', exist_ok=True)
os.makedirs('output/DE/voom', exist_ok=True)

save_name = 'Green_broad'

with Timer(f'[{save_name}] loading single cell'):
    sc = SingleCell('../../single-cell/Green/p400_qced_shareable.h5ad',
                    num_threads=os.cpu_count())\
        .qc(cell_type_confidence_column='cell.type.prob',
            doublet_column='is.doublet.df',
            custom_filter=pl.col.projid.is_not_null())\
        .with_columns_obs(
            subset=pl.col.subset.replace({'CUX2+': 'Excitatory'}),
            projid=pl.col.projid.cast(pl.String))

with Timer(f'[{save_name}] pseudobulking'):
    # radc.rush.edu/docs/var/variables.htm
    rosmap_meta = pl.read_csv(
        '../../single-cell/Green/dataset_978_basic_04-21-2023_with_pmAD.csv',
        dtypes={'projid': pl.String})\
        .unique(subset='projid')\
        .drop([col for col in sc.obs.columns if col != 'projid'])
    pb = sc.pseudobulk(ID_column='projid', 
            cell_type_column='subset',
            additional_obs=rosmap_meta)\
        .filter_var(pl.col._index.is_in(get_coding_genes()['gene']))\
        .with_columns_obs(
            dx_cont=pl.when(pl.col.cogdx == 1).then(0)
                .when(pl.col.cogdx.is_in([2, 3])).then(1)
                .when(pl.col.cogdx.is_in([4, 5])).then(2)
                .otherwise(None),
            apoe4_dosage=pl.col.apoe_genotype.cast(pl.String)
                .str.count_matches('4').fill_null(strategy='mean'),
            pmi=pl.col.pmi.fill_null(strategy='mean'))
    if not os.path.exists(f'output/pseudobulk/{save_name}'):
        pb.save(f'output/pseudobulk/{save_name}')

################################################################################

save_name = 'Mathys_broad'

with Timer(f'[{save_name}] loading single cell'):
    data_dir = '../../single-cell/Mathys'
    # cells.ucsc.edu/ad-aging-brain/ad-aging-brain/meta.tsv
    basic_meta = pl.read_csv(
        f'{data_dir}/meta.tsv', 
        columns=['cellName', 'Dataset', 'Major_Cell_Type', 
                 'Cell_Type', 'Individual'], 
        separator='\t')
    assert basic_meta.shape[0] == 2327742
    # syn21323366
    id_map1 = pl.read_csv(
        f'{data_dir}/MIT_ROSMAP_Multiomics_individual_metadata.csv',
        columns=['individualID', 'individualIdSource', 'subject'])\
        .filter(pl.col.subject.is_not_null())\
        .unique(subset='subject')
    # syn3191087
    id_map2 = pl.read_csv(
        f'{data_dir}/ROSMAP_clinical.csv',
        columns=['projid', 'apoe_genotype', 'individualID'],
        dtypes={'projid': pl.String}, null_values='NA')
    # personal.broadinstitute.org/cboix/ad427_data/Data/Metadata/
    # individual_metadata_deidentified.tsv
    subject_meta = pl.read_csv(
        f'{data_dir}/individual_metadata_deidentified.tsv', 
        separator='\t', null_values='NA')
    # radc.rush.edu/docs/var/variables.htm
    rosmap_meta = pl.read_csv(
        f'{data_dir}/dataset_978_basic_04-21-2023_with_pmAD.csv',
        dtypes={'projid': pl.String})\
        .unique(subset='projid') 
    full_meta = basic_meta\
        .join(subject_meta, 
              left_on='Individual', right_on='subject', how='left')\
        .join(id_map1, left_on='Individual', right_on='subject', how='left')\
        .join(id_map2, on='individualID', how='left')\
        .join(rosmap_meta, on='projid', how='left')
    assert full_meta.shape[0] == 2327742
    # personal.broadinstitute.org/cboix/ad427_data/Data/Raw_data 
    # subsetting to cells in basic_meta, which pass authors' QC (custom_filter)
    # no doublet or cell type confidence filter applied      
    sc = SingleCell(f'{data_dir}/PFC427_raw_data.h5ad', 
                    num_threads=os.cpu_count())\
        .join_obs(full_meta, left_on='_index', right_on='cellName',
                  validate='1:1')\
        .qc(cell_type_confidence_column=None, 
            custom_filter=pl.col.Major_Cell_Type.is_not_null(),
            doublet_column=None, allow_float=True)

with Timer(f'[{save_name}] pseudobulking'):
    pb = sc.pseudobulk(ID_column='projid', 
            cell_type_column='Major_Cell_Type',
            num_threads=os.cpu_count())\
        .filter_var(pl.col._index.is_in(get_coding_genes()['gene']))\
        .with_columns_obs(
            dx_cont=pl.when(pl.col.cogdx == 1).then(0)
                .when(pl.col.cogdx.is_in([2, 3])).then(1)
                .when(pl.col.cogdx.is_in([4, 5])).then(2)
                .otherwise(None),
            sex=pl.coalesce(['msex_right', 'msex']),
            age_death=pl.when(pl.col.age_death == "90+").then(90.0)
               .otherwise(pl.col.age_death.str.extract_all(r"(\d+)")
                    .list.eval(pl.element().cast(pl.Int32).mean())
                    .list.first()),
            pmi=pl.coalesce(['pmi_right', 'pmi'])
                .fill_null(strategy='mean'),
            apoe4_dosage=pl.coalesce(['apoe_genotype_right', 'apoe_genotype'])
                .cast(pl.String).str.count_matches('4')
                .fill_null(strategy='mean'))
    if not os.path.exists(f'output/pseudobulk/{save_name}'):
        pb.save(f'output/pseudobulk/{save_name}')
    
# differential expression ######################################################

data_names = ['Green_broad']
for data in data_names:
    with Timer(f'[{data}] qc and case-control differential expression'):
        pb = Pseudobulk(f'output/pseudobulk/{data}')
        de = pb.qc(case_control_column='pmAD', 
                custom_filter=pl.col.pmAD.is_not_null())\
            .DE(label_column='pmAD', 
                case_control=True,
                covariate_columns=['age_death', 'sex', 'pmi', 'apoe4_dosage'],
                include_library_size_as_covariate=True)
        de.plot_voom(save_to=f'figures/DE/voom/{data}_cc', overwrite=True)
        de.get_hits(threshold=1).write_csv(f'output/DE/{data}_cc.csv')     
        
    with Timer(f'[{data}] qc and continous differential expression'):
        de = pb.qc(case_control_column=None, 
            custom_filter=pl.col.dx_cont.is_not_null())\
        .DE(label_column='dx_cont', 
            case_control=False,
            covariate_columns=['age_death', 'sex', 'pmi', 'apoe4_dosage'],
            include_library_size_as_covariate=True)
        de.plot_voom(save_to=f'figures/DE/voom/{data}_cont', 
                    overwrite=True)
        de.get_hits(threshold=1).write_csv(f'output/DE/{data}_cont.csv')  

de = pl.read_csv('output/DE/Green_broad_cc.csv')
print_df(de.get_num_hits(threshold=0.05).sort('cell_type'))

'''
Green
cont
cell_type         num_hits 
 Astrocytes        322      
 Endothelial       1        
 Excitatory        689      
 Inhibitory        169      
 Microglia         8        
 OPCs              2        
 Oligodendrocytes  184      

cc
cell_type         num_hits 
 Astrocytes        348      
 Endothelial       13       
 Excitatory        398      
 Inhibitory        86       
 Microglia         9        
 OPCs              43       
 Oligodendrocytes  139  

Mathys 
cont
 cell_type  num_hits 
 Ast        189      
 Exc        15       
 Inh        2        
 Mic        12       
 Oli        420      
 Opc        10       
 Vas        1   
 
cc
cell_type  num_hits 
 Ast        573      
 Exc        14       
 Inh        2        
 Mic        29       
 Oli        641      
 Opc        47       
 Vas        2    
'''

################################################################################

import sys, os, optuna, pickle
import polars as pl, pandas as pd, numpy as np

sys.path.append('/home/karbabi/projects/def-wainberg/karbabi/utils')
from single_cell import Pseudobulk
from utils import linear_regressions, fdr, debug, savefig
from ryp import r, to_r    

os.chdir('projects/def-wainberg/karbabi/pseudobulk-nmf')
from sklearn.decomposition._nmf import _initialize_nmf
from sparseNMF import sparse_nmf

debug(third_party=True)

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
                              tol=1e-4, maxiter=np.iinfo('int32').max, 
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
    lower, upper = np.quantile(MSE_values, [0, 0.9])
    ax.set_ylim(bottom=lower-0.05, top=upper)
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

data_name = 'Green_broad'
save_name = 'cont_fdr05'

de = pl.read_csv(f'output/DE/{data_name}_cont.csv')
pb = Pseudobulk(f'output/pseudobulk/{data_name}')\
    .filter_obs(pl.col.dx_cont.is_in([1, 2]))
shared_ids = sorted(
    set.intersection(*(set(obs['ID']) for obs in pb.iter_obs())))
lcpm = pb\
    .filter_obs(pl.col.ID.is_in(shared_ids))\
    .log_CPM()\
    .regress_out_obs(covariate_columns='pmi')

matrices, cell_types, genes = [], [], []
for cell_type, (X, obs, var) in lcpm.items():
    gene_mask = var['_index'].is_in(
        de.filter((pl.col.cell_type == cell_type) & \
            (pl.col.FDR < 0.05))['gene']) 
    matrices.append(X.T[gene_mask]) 
    gene_select = var['_index'].filter(gene_mask).to_list()    
    genes.extend(gene_select)    
    cell_types.extend([cell_type] * len(gene_select))

from utils import inverse_normal_transform
A = np.vstack(matrices)
A = np.apply_along_axis(inverse_normal_transform, 1, A)
A += abs(np.min(A))

MSE_trial, rank_1se_trial = {}, {}
study = optuna.create_study(
    sampler=optuna.samplers.TPESampler(multivariate=True),
    direction='minimize')
study.optimize(lambda trial: objective(
    trial, MSE_trial, rank_1se_trial, A, rank_max=20), 
    n_trials=30)

# with open(f'output/NMF/trial/{data_name}_{save_name}.pkl', 'wb') as file:
#     pickle.dump((study, MSE_trial, rank_1se_trial), file)
with open(f'output/NMF/trial/{data_name}_{save_name}.pkl', 'rb') as file:
    study, MSE_trial, rank_1se_trial = pickle.load(file)

spar_select = study.best_trial.params.get('spar')
rank_select = rank_1se_trial[spar_select]
plot_MSE_trial(MSE_trial, rank_1se_trial, spar_select, rank_select, 
               filename=f"figures/NMF/MSE/{save_name}.png")

W, H = _initialize_nmf(A, n_components=rank_select, init='nndsvd')
W, H = sparse_nmf(A, rank=rank_select, spar=spar_select, W=W, H=H,
                  tol=1e-8, maxiter=np.iinfo('int32').max, verbose=2)
A_r = W @ H

W = pl.DataFrame(W)\
    .rename(lambda col: col.replace('column_', 'S'))\
    .insert_column(0, pl.Series(genes).alias('gene'))
H = pl.DataFrame(H.T)\
    .rename(lambda col: col.replace('column_', 'S'))\
    .insert_column(0, pl.Series(shared_ids).alias('ID'))
    
W.write_csv(
    f'output/NMF/factors/W_{data_name}_{save_name}.tsv', separator='\t')
H.write_csv(
    f'output/NMF/factors/H_{data_name}_{save_name}.tsv', separator='\t')
pl.DataFrame(A, shared_ids)\
    .insert_column(0, pl.Series(genes).alias('gene'))\
    .write_csv(f'output/NMF/A/p400_A{save_name}.tsv', separator='\t')
pl.DataFrame(A_r, shared_ids)\
    .insert_column(0, pl.Series(genes).alias('gene'))\
    .write_csv(f'output/NMF/A/p400_Ar{save_name}.tsv', separator='\t')

meta = lcpm.obs[next(lcpm.keys())]\
    [['age_death', 'sex', 'pmi', 'apoe4_dosage']]\
    .with_columns(sex=pl.when(pl.col.sex == 'Male').then(1).otherwise(0))
    
H_de = pl.concat([
    pl.DataFrame({
        'gene': genes, 
        'beta': res.beta[0], 
        'SE': res.SE[0], 
        'lower_CI': res.lower_CI[0], 
        'upper_CI': res.upper_CI[0], 
        'p': res.p[0]
    }).with_columns(
        fdr=fdr(pl.col('p')), 
        H=pl.lit(f'S{i}'))
    for i in range(H.shape[1]-1)
    for res in [linear_regressions(
        X=H[:, i+1].to_frame().hstack(meta), 
        Y=A.T, return_significance=True)]
])

print_df(H_de.filter((pl.col.H=='S2') & (pl.col.beta > 0) & (pl.col.fdr < 0.05))
         .sort('fdr')[['gene']], num_rows=200)

print_df(H_de)


to_r(A, 'A', format='matrix', rownames=genes)
to_r(A_r, 'A_r', format='matrix', rownames=genes)
to_r(W.drop('gene'), "W", rownames=W['gene'])
to_r(H.drop('ID'), "H", rownames=shared_ids)
to_r(np.array(cell_types), 'cell_types')
to_r(save_name, 'save_name')
to_r(data_name, 'data_name')
meta = lcpm.obs[next(lcpm.keys())]
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
    left_annotation = hr1,
    right_annotation = hr2,
    col = col_fun,
    name = "normalized\nexpression",
    heatmap_legend_param = list(
        title_gp = gpar(fontsize = 7, fontface = "bold"),
        labels_gp = gpar(fontsize = 5)),
    show_heatmap_legend = TRUE,
    use_raster = FALSE
)
file_name = paste0("figures/NMF/A/", data_name, "_", save_name, ".png")
png(file = file_name, width=7, height=7, units="in", res=1200)
draw(h)
dev.off()
''')

r('''
library(tidyverse)
library(corrplot)
meta = meta %>%
    select(ID, num_cells, sex, Cdx, braaksc, ceradsc, pmi, niareagansc, 
            apoe4_dosage, tomm40_hap, age_death, age_first_ad_dx, gpath,
            amyloid, hspath_typ, dlbdx, tangles, tdp_st4, arteriol_scler,
            caa_4gp, cvda_4gp2, ci_num2_gct, ci_num2_mct) %>%
    rename(c("Number of cells" = num_cells, "Cognitive diagnosis" = Cdx, 
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
            "Chronic microinfarcts" = ci_num2_mct)) %>%
    mutate(across(where(is.factor), as.numeric)) %>%
    mutate(across(where(is.numeric), 
            ~ifelse(is.na(.), median(., na.rm = TRUE), .))) %>%
    mutate(`Cerad score` = rev(`Cerad score`), 
           `NIA-Reagan diagnosis` = rev(`NIA-Reagan diagnosis`)) %>%
    column_to_rownames(var = "ID")

cor_mat = t(cor(H, meta))
p_mat = matrix(NA, ncol(H), ncol(meta))
for (i in 1:ncol(H)) {
    for (j in 1:ncol(meta)) {
        p_mat[i, j] = cor.test(H[, i], meta[, j])$p.value
    }
}
p_mat = t(p_mat)

row_order = get_order(seriate(dist(cor_mat), method = "OLO"))
cor_mat = cor_mat[as.numeric(row_order),]
p_mat = p_mat[as.numeric(row_order),]
rownames(p_mat) = rownames(cor_mat)
colnames(p_mat) = colnames(cor_mat)

png(paste0("figures/NMF/corr/", data_name, "_", save_name, ".png"), 
    width = 7, height = 10, units = "in", res=300)
corrplot(cor_mat, is.corr = FALSE,  
        p.mat = p_mat, sig.level = 0.05,
        insig = 'label_sig', pch.cex = 2, pch.col = "white",
        tl.col = "black")
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





# meta = lcpm.obs[next(lcpm.keys())]
# meta.write_csv('results/NMF/A/p400_metadata.tsv', separator='\t')
# pd.DataFrame(cell_types).to_csv('results/NMF/A/p400_celltypes.tsv', 
#                                 sep='\t', index=False)

















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

