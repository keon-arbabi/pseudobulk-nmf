




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



























from matplotlib.colors import ListedColormap

data1 = mat[0:50, 0:50] 
data2 = mask[0:50, 0:50]
fig, ax = plt.subplots(figsize=(5,5))    
sns.heatmap(data1, cbar=False, xticklabels=False, yticklabels=False,
            rasterized=True)
sns.heatmap(data2, cmap = ListedColormap(['white']), 
            cbar=False, xticklabels=False, yticklabels=False,
            rasterized=True, square=True, mask=data2)
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

