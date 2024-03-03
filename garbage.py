
cell_type = 'Inhibitory'
X = lcpm.X[cell_type]
obs = lcpm.obs[cell_type]
var = lcpm.var[cell_type]

gene_mask = var['_index'].is_in(
    de.filter((pl.col.cell_type==cell_type) & (pl.col.FDR < 0.1))['gene']) 
V = X.T[gene_mask, :] 

import nimfa
snmf = nimfa.Snmf(V)

ranks = range(1, 20, 1)
summary = snmf.estimate_rank(rank_range=ranks, n_run=30, what='all')
summary[1].keys()
    
rss = [summary[rank]['rss'] for rank in ranks]
coph = [summary[rank]['cophenetic'] for rank in ranks]
disp = [summary[rank]['dispersion'] for rank in ranks]
evar = [summary[rank]['evar'] for rank in ranks]    
spar = [summary[rank]['sparseness'] for rank in ranks]
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

