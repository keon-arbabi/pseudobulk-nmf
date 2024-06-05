import sys, os, gc, polars as pl, polars.selectors as cs
import matplotlib.pyplot as plt, seaborn as sns

sys.path.append('/home/karbabi/projects/def-wainberg/karbabi/utils')
from single_cell import SingleCell
from utils import Timer, get_coding_genes, savefig
    
os.chdir('projects/def-wainberg/karbabi/pseudobulk-nmf')
os.makedirs('output/pseudobulk/plots', exist_ok=True)

def confusion_matrix_plot(sc, original_labels_column,
                          transferred_labels_column, save_to):
    confusion_matrix = sc.obs\
        .select(original_labels_column, transferred_labels_column)\
        .to_pandas()\
        .groupby([original_labels_column, transferred_labels_column],
                 observed=True)\
        .size()\
        .unstack(fill_value=0)\
        .sort_index(axis=1)\
        .assign(broad_cell_type=lambda df: df.index.str.split('.').str[0],
                cell_type_cluster=lambda df: df.index.str.split('.').str[1]
                .astype('Int64').fillna(0))\
        .sort_values(['broad_cell_type', 'cell_type_cluster'])\
        .drop(['broad_cell_type', 'cell_type_cluster'], axis=1)
    print(confusion_matrix)
    ax = sns.heatmap(confusion_matrix.T.div(confusion_matrix.T.sum()),
                     xticklabels=1, yticklabels=1, rasterized=True,
                     square=True, linewidths=0.5, cmap='rocket_r',
                     cbar_kws=dict(pad=0.01), vmin=0, vmax=1)
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    cbar.set_ticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
    w, h = plt.gcf().get_size_inches()
    plt.gcf().set_size_inches(3.5 * w, h)
    savefig(save_to)

def cell_type_annotation(sc, study_name, original_labels_column):
    with Timer('[Mini SEAAD] loading single cell'):
        sc_ref = SingleCell(
            '../../single-cell/SEAAD/'
            'Reference_MTG_RNAseq_final-nuclei.2022-06-07.h5ad',
            num_threads=None)\
            .qc(cell_type_confidence_column='subclass_confidence',
                doublet_column=None, allow_float=True)
    with Timer(f'[{study_name}] highly-variable genes'):
        sc, sc_ref = sc.hvg(sc_ref, allow_float=True)
    with Timer(f'[{study_name}] PCA'):
        sc, sc_ref = sc.PCA(sc_ref, allow_float=True, verbose=True)
    with Timer(f'[{study_name}] Harmony'):
        sc, sc_ref = sc.harmonize(sc_ref, pytorch=True, num_threads=None)
    with Timer(f'[{study_name}] label transfer'):
        sc = sc.label_transfer_from(
            sc_ref, 
            cell_type_column='subclass_label',
            cell_type_confidence_column='subclass_confidence',
            min_cell_type_confidence=0.90,
            num_neighbors=20,
            num_index_neighbors=100)\
            .rename_obs(({
                'subclass_label': 'cell_type_fine',
                'subclass_confidence': 'cell_type_fine_confidence'}))
    with Timer(f'[{study_name}] UMAP'):
        sc = sc.UMAP(seed=None, num_threads=24)
    with Timer(f'[{study_name}] plots'):
        sc.plot_UMAP(
            'cell_type_fine', 
            f'output/pseudobulk/plots/{study_name}_umap.png',
            legend_kwargs={'fontsize': 'x-small'})
        confusion_matrix_plot(
            sc, original_labels_column, 'cell_type_fine', 
            f'output/pseudobulk/plots/{study_name}_confusion_matrix.png')
    return(sc)

# Green et al. 2023 ############################################################
study_name = 'Green'

with Timer(f'[{study_name}] loading single cell'):
    sc_file = '../../single-cell/Green/Green_qced_labelled.h5ad'
    if os.path.exists(sc_file):
        sc = SingleCell(sc_file, num_threads=None)
    else:
        sc = SingleCell('../../single-cell/Green/p400_qced_shareable.h5ad',
                        num_threads=None)
        rosmap_meta = pl.read_csv(
            '../../single-cell/Green/'
            'dataset_978_basic_04-21-2023_with_pmAD.csv',
            dtypes={'projid': pl.Int32})\
            .unique(subset='projid')\
            .drop([col for col in sc.obs.columns if col != 'projid'])
        sc = sc\
            .join_obs(rosmap_meta, on='projid', validate='m:1')\
            .with_columns_obs(
                projid=pl.col.projid.cast(pl.String).fill_null(''),
                cell_type_broad=pl.col.subset.replace(
                    {'CUX2+': 'Excitatory'}))\
            .qc(cell_type_confidence_column='cell.type.prob',
                doublet_column='is.doublet.df',
                custom_filter=pl.col.projid.ne(''))
        sc = cell_type_annotation(sc, study_name, 'state')\
            .with_columns_obs(cs.string().cast(pl.Categorical))
        sc.save(sc_file)
 
for level in ['broad', 'fine']:
    with Timer(f'[{study_name}] pseudobulking at the {level} level'):
        pb = sc\
            .pseudobulk(
                ID_column='projid', 
                cell_type_column=f'cell_type_{level}')\
            .filter_var(
                pl.col._index.is_in(get_coding_genes()['gene']))\
            .with_columns_obs(
                dx_cont=pl.when(pl.col.cogdx == 1).then(0)
                    .when(pl.col.cogdx.is_in([2, 3])).then(1)
                    .when(pl.col.cogdx.is_in([4, 5])).then(2)
                    .otherwise(None),
                apoe4_dosage=pl.col.apoe_genotype.cast(pl.String)
                    .str.count_matches('4').fill_null(strategy='mean'),
                pmi=pl.col.pmi.fill_null(strategy='mean'))
        if not os.path.exists(f'output/pseudobulk/{study_name}_{level}'):
            pb.save(f'output/pseudobulk/{study_name}_{level}')

del sc, pb; gc.collect()

# Mathys et al. 2023 ############################################################
# # basic_meta: cells.ucsc.edu/ad-aging-brain/ad-aging-brain/meta.tsv
# # id_map1: synapse.org/#!Synapse:syn21323366
# # id_map2: synapse.org/#!Synapse:syn3191087
# # subject_meta: personal.broadinstitute.org/cboix/ad427_data/Data/Metadata/individual_metadata_deidentified.tsv
# # rosmap_meta: radc.rush.edu/docs/var/variables.htm
study_name = 'Mathys'

with Timer('[Mathys] loading single cell'):
    data_dir = '../../single-cell/Mathys'
    basic_meta = pl.read_csv(
        f'{data_dir}/meta.tsv', 
        columns=['cellName', 'Dataset', 'Major_Cell_Type', 
                 'Cell_Type', 'Individual'], 
        separator='\t')\
        .with_columns(cell_type_broad=pl.col.Major_Cell_Type.replace({
            'Exc': 'Excitatory', 'Inh': 'Inhibitory',
            'Ast': 'Astrocytes', 'Oli': 'Oligodendrocytes',
            'Mic': 'Microglia', 'Vas': 'Endothelial',
            'OPCs': 'Opc'}))
    assert basic_meta.shape[0] == 2327742
    id_map1 = pl.read_csv(
        f'{data_dir}/MIT_ROSMAP_Multiomics_individual_metadata.csv',
        columns=['individualID', 'individualIdSource', 'subject'])\
        .filter(pl.col.subject.is_not_null())\
        .unique(subset='subject')
    id_map2 = pl.read_csv(
        f'{data_dir}/ROSMAP_clinical.csv',
        columns=['projid', 'apoe_genotype', 'individualID'],
        dtypes={'projid': pl.String}, null_values='NA')
    subject_meta = pl.read_csv(
        f'{data_dir}/individual_metadata_deidentified.tsv', 
        separator='\t', null_values='NA')
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
    sc = SingleCell(f'{data_dir}/PFC427_raw_data.h5ad', 
                    num_threads=os.cpu_count())\
        .join_obs(full_meta, 
                  left_on='_index', right_on='cellName', validate='1:1')\
        .qc(cell_type_confidence_column=None, 
            doublet_column=None,
            custom_filter=pl.col.cell_type_broad.is_not_null(),
            allow_float=True)

with Timer(f'[{save_name}] pseudobulking'):
    pb = sc\
        .pseudobulk(ID_column='projid', 
            cell_type_column='cell_type_broad',
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
                .fill_null(strategy='mean'),
            log_num_cells=pl.col.num_cells.log(base=2))
    if not os.path.exists(f'output/pseudobulk/{save_name}'):
        pb.save(f'output/pseudobulk/{save_name}')
        
del sc, pb; gc.collect()

# Gabitto et al. 2023 ############################################################
save_name = 'Gabitto_broad'

with Timer(f'[{save_name}] loading single cell'):
    data_dir = '../../single-cell/SEAAD/'
    # sea-ad-single-cell-profiling.s3.amazonaws.com/index.html#DLPFC/RNAseq/
    sc = SingleCell(
        f'{data_dir}/SEAAD_DLPFC_RNAseq_all-nuclei.2024-02-13.h5ad',
        num_threads=os.cpu_count())\
        .with_columns_obs(pl.col('Donor ID').cast(pl.String))
    # portal.brain-map.org/explore/seattle-alzheimers-disease/
    # seattle-alzheimers-disease-brain-cell-atlas-download?edit&language=en
    donor_metadata = pl.read_excel(
        f'{data_dir}/sea-ad_cohort_donor_metadata.xlsx')
    pseudoprogression_scores = pl.read_csv(
        f'{data_dir}/pseudoprogression_scores.csv')
    sc = sc\
        .join_obs(donor_metadata.select(['Donor ID'] +
            list(set(donor_metadata.columns).difference(sc.obs.columns))),
            on='Donor ID', validate='m:1')\
        .join_obs(pseudoprogression_scores, on='Donor ID', validate='m:1')\
        .with_columns_obs(
            pl.col('Doublet score').gt(0.5).alias('Is doublet'),
            pl.coalesce(
                pl.col.Subclass.replace({
                    'Astro': 'Astrocytes', 
                    'Endo': 'Endothelial', 
                    'Micro-PVM': 'Microglia', 
                    'Oligo': 'Oligodendrocytes',
                    'Opc': 'OPCs', 
                    'VLMC': 'Endothelial'}, default=None),
                pl.col.Class.replace({
                    'exc': 'Excitatory', 'inh': 'Inhibitory'}, default=None))
                .alias('cell_type_broad'))\
        .qc(doublet_column='Is doublet',
            cell_type_confidence_column='Class confidence',
            custom_filter='Used in analysis',
            allow_float=True)

with Timer(f'[{save_name}] pseudobulking'):
    pb = sc\
        .pseudobulk(ID_column='Donor ID', 
            cell_type_column='cell_type_broad',
            num_threads=os.cpu_count())\
        .filter_var(pl.col._index.is_in(get_coding_genes()['gene']))\
        .filter_obs(~pl.col.ID.is_in([
            'H18.30.002', 'H19.30.002', 'H19.30.001']))\
        .with_columns_obs(
            dx_cc=pl.when(
                pl.col('Consensus Clinical Dx (choice=Alzheimers disease)')
                .eq('Checked')).then(1)
                .when(pl.col('Consensus Clinical Dx (choice=Control)')
                .eq('Checked')).then(0)
                .otherwise(None),
            dx_cont=pl.when(
                pl.col('Overall AD neuropathological Change')
                .eq('Not AD')).then(0)
                .when(pl.col('Overall AD neuropathological Change')
                .eq('Low')).then(1)
                .when(pl.col('Overall AD neuropathological Change')
                .eq('Intermediate')).then(2)
                .when(pl.col('Overall AD neuropathological Change')
                .eq('High')).then(3)
                .otherwise(None),
            apoe4_dosage=pl.col('APOE Genotype')
                .cast(pl.String).str.count_matches('4')
                .fill_null(strategy='mean'),
            log_num_cells=pl.col.num_cells.log(base=2))
    if not os.path.exists(f'output/pseudobulk/{save_name}'):
        pb.save(f'output/pseudobulk/{save_name}')
        
del sc, pb; gc.collect()