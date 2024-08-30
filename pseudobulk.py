import sys, os, gc
import polars as pl
import matplotlib.pyplot as plt, seaborn as sns

sys.path.append('/home/karbabi/projects/def-wainberg/karbabi/utils')
from single_cell import SingleCell
from utils import Timer, get_coding_genes, savefig, print_df, debug
    
debug(third_party=True)

def confusion_matrix_plot(sc, original_labels_column,
                          transferred_labels_column, directory):
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
    confusion_matrix.to_csv(f'{directory}/cell_type_fine_confusion.csv')
    ax = sns.heatmap(
        confusion_matrix.T.div(confusion_matrix.T.sum()),
        xticklabels=1, yticklabels=1, rasterized=True,
        square=True, linewidths=0.5, cmap='rocket_r',
        cbar_kws=dict(pad=0.01), vmin=0, vmax=1)
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    cbar.set_ticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
    w, h = plt.gcf().get_size_inches()
    plt.gcf().set_size_inches(3.5 * w, h)
    savefig(f'{directory}/cell_type_fine_confusion.png')

def cell_type_annotation(sc, study_name, original_labels_column, directory):
    with Timer('[Mini SEAAD] Loading single cell'):
        sc_ref = SingleCell(
            'projects/def-wainberg/single-cell/SEAAD/'
            'Reference_MTG_RNAseq_final-nuclei.2022-06-07.h5ad',
            num_threads=None)\
            .qc(custom_filter=pl.col('subclass_confidence').ge(0.9),
                allow_float=True,
                num_threads=None)
    with Timer(f'[{study_name}] Highly-variable genes'):
        sc, sc_ref = sc.hvg(sc_ref, allow_float=True, num_threads=None)
    with Timer(f'[{study_name}] Normalize'):
        sc = sc.normalize(allow_float=True, num_threads=None)
        sc_ref = sc_ref.normalize(allow_float=True, num_threads=None)
    with Timer(f'[{study_name}] PCA'):
        sc, sc_ref = sc.PCA(sc_ref, num_threads=1)
    with Timer(f'[{study_name}] Harmony'):
        sc, sc_ref = sc.harmonize(sc_ref, num_threads=1)
    with Timer(f'[{study_name}] Label transfer'):
        sc = sc.label_transfer_from(
            sc_ref, 
            original_cell_type_column='subclass_label',
            cell_type_column='cell_type_fine',
            cell_type_confidence_column='cell_type_fine_confidence')\
            .with_columns_obs(
                passed_cell_type_fine=pl.col.cell_type_fine_confidence.ge(0.9))
    print(sc.obs['passed_cell_type_fine'].value_counts())
    print_df(sc.obs.group_by('cell_type_fine')
        .agg(mean=pl.col('cell_type_fine_confidence').mean(),
             count=pl.col('cell_type_fine_confidence').count())
        .sort('mean'))    
    with Timer(f'[{study_name}] PaCMAP'):
        sc = sc.embed(num_threads=None)
    with Timer(f'[{study_name}] Plots'):
        sc.plot_embedding(
            'cell_type_fine', 
            f'{directory}/cell_type_fine_pacmap.png',
            cells_to_plot_column='passed_cell_type_fine',
            label=True, label_kwargs={'size': 6},
            legend_kwargs={'fontsize': 'x-small', 'ncols': 1})
        confusion_matrix_plot(
            sc, original_labels_column, 'cell_type_fine', directory)
    return(sc)

################################################################################

rosmap_basic = pl.read_csv(
    'projects/def-wainberg/single-cell/Green/'
    'dataset_978_basic_04-21-2023.csv')\
    .unique(subset='projid')\
    .drop('study', 'scaled_to')
rosmap_long = pl.read_csv(
    'projects/def-wainberg/single-cell/Green/'
    'dataset_978_long_04-21-2023.csv',
    infer_schema_length=10000)\
    .sort('projid', 'fu_year', descending=[False, True])\
    .unique(subset='projid', keep='first')
rosmap_all = rosmap_basic\
    .join(rosmap_long, on='projid', how='full', coalesce=True)\
    .pipe(lambda tdf: tdf.drop([
        col for col in tdf.columns 
        if tdf[col].null_count() == tdf.height]))

# Green et al. 2023 ############################################################

study_name = 'Green'
sc_dir = f'projects/def-wainberg/single-cell/{study_name}'
sc_file = f'{sc_dir}/{study_name}_qced_labelled.h5ad'

if os.path.exists(sc_file):
    with Timer(f'[{study_name}] Loading single cell'):
        sc = SingleCell(sc_file, num_threads=None)
else:
    with Timer(f'[{study_name}] Preprocessing single cell'):
        sc = SingleCell(f'{sc_dir}/p400_qced_shareable.h5ad', num_threads=None)
        drop_cols = set(rosmap_all.columns) & set(sc.obs.columns) - {'projid'}
        sc = sc\
            .drop_obs(drop_cols)\
            .join_obs(rosmap_all.cast({'projid': pl.Int32}), on='projid')\
            .with_columns_obs(
                projid=pl.col.projid.cast(pl.String),
                cell_type_broad=pl.col.subset.cast(pl.String)
                    .replace({'CUX2+': 'Excitatory'}))\
            .qc(custom_filter=pl.col('cell.type.prob').ge(0.9) & 
                    pl.col.projid.is_not_null() &
                    pl.col('is.doublet.df').not_(),
                num_threads=None)
        sc_orig = sc.copy()
        sc = cell_type_annotation(
            sc, 
            study_name=study_name, 
            original_labels_column='state', 
            directory=sc_dir)
        sc.X = sc_orig.X
        sc = sc.drop_uns('normalized')
        sc.save(sc_file, overwrite=True)

for level in ['broad', 'fine']:
    with Timer(f'[{study_name}] Pseudobulking at the {level} level'):
        QC_column = 'passed_cell_type_fine' if level == 'fine' else None 
        pb = sc\
            .pseudobulk(
                ID_column='projid', 
                cell_type_column=f'cell_type_{level}',
                QC_column=QC_column,
                sort_genes=True,
                num_threads=None)\
            .filter_var(
                pl.col._index.is_in(get_coding_genes()['gene']))\
            .with_columns_obs(
                dx_cc=pl.col.pmAD,
                dx_cont=pl.when(pl.col.cogdx == 1).then(0)
                    .when(pl.col.cogdx.is_in([2, 3])).then(1)
                    .when(pl.col.cogdx.is_in([4, 5])).then(2)
                    .otherwise(None),
                apoe4_dosage=pl.col.apoe_genotype.cast(pl.String)
                    .str.count_matches('4').fill_null(strategy='mean'),
                pmi=pl.col.pmi.fill_null(strategy='mean'))
        pb.save(f'{sc_dir}/pseudobulk/{level}', overwrite=True)

del sc, pb; gc.collect()

# Mathys et al. 2023 ############################################################
# # basic_meta: cells.ucsc.edu/ad-aging-brain/ad-aging-brain/meta.tsv
# # id_map1: synapse.org/#!Synapse:syn21323366
# # id_map2: synapse.org/#!Synapse:syn3191087
# # subject_meta: personal.broadinstitute.org/cboix/ad427_data/Data/Metadata/individual_metadata_deidentified.tsv
# # rosmap_meta: radc.rush.edu/docs/var/variables.htm

study_name = 'Mathys'
sc_dir = f'projects/def-wainberg/single-cell/{study_name}'
sc_file = f'{sc_dir}/{study_name}_qced_labelled.h5ad'

if os.path.exists(sc_file):
    with Timer(f'[{study_name}] Loading single cell'):
        sc = SingleCell(sc_file, num_threads=None)
else:
    with Timer(f'[{study_name}] Preprocessing single cell'):
        basic_meta = pl.read_csv(
            f'{sc_dir}/meta.tsv', 
            columns=['cellName', 'Dataset', 'Major_Cell_Type', 
                    'Cell_Type', 'Individual'], separator='\t')\
            .with_columns(
                cell_type_broad=pl.col.Major_Cell_Type.replace({
                    'Exc': 'Excitatory', 'Inh': 'Inhibitory',
                    'Ast': 'Astrocytes', 'Oli': 'Oligodendrocytes',
                    'Mic': 'Microglia', 'Vas': 'Endothelial',
                    'Opc': 'OPCs'}))
        assert basic_meta.shape[0] == 2327742
        id_map1 = pl.read_csv(
            f'{sc_dir}/MIT_ROSMAP_Multiomics_individual_metadata.csv',
            columns=['individualID', 'individualIdSource', 'subject'])\
            .filter(pl.col.subject.is_not_null())\
            .unique(subset='subject')
        id_map2 = pl.read_csv(
            f'{sc_dir}/ROSMAP_clinical.csv',
            columns=['projid', 'apoe_genotype', 'individualID'],
            schema_overrides={'projid': pl.String}, null_values='NA')
        subject_meta = pl.read_csv(
            f'{sc_dir}/individual_metadata_deidentified.tsv', 
            separator='\t', null_values='NA')
        full_meta = basic_meta\
            .join(subject_meta, 
                  left_on='Individual', right_on='subject', how='left',
                  coalesce=True)\
            .join(id_map1, left_on='Individual', right_on='subject', how='left',
                  coalesce=True)\
            .join(id_map2, on='individualID', how='left', coalesce=True)
        drop_cols = \
            set(full_meta.columns) & set(rosmap_all.columns) - {'projid'}
        full_meta = full_meta\
            .drop(drop_cols)\
            .join(rosmap_all.cast({'projid': pl.String}),
                  on='projid', how='left', coalesce=True)
        assert full_meta.shape[0] == 2327742 

        sc = SingleCell(f'{sc_dir}/PFC427_raw_data.h5ad', num_threads=None)
        sc = sc\
            .join_obs(full_meta, 
                      left_on='_index', right_on='cellName', validate='1:1')\
            .qc(custom_filter=pl.col.cell_type_broad.is_not_null() &
                    pl.col.projid.is_not_null(),
                MALAT1_filter=False,
                allow_float=True,
                num_threads=None)
        sc_orig = sc.copy()
        sc = cell_type_annotation(
            sc, 
            study_name=study_name, 
            original_labels_column='Cell_Type', 
            directory=sc_dir)
        sc.X = sc_orig.X
        sc.save(sc_file, overwrite=True)

for level in ['broad', 'fine']:
    with Timer(f'[{study_name}] Pseudobulking at the {level} level'):
        QC_column = 'passed_QC_fine' if level == 'broad' else 'passed_QC'
        pb = sc\
            .pseudobulk(
                ID_column='projid', 
                cell_type_column=f'cell_type_{level}',
                QC_column=QC_column,
                num_threads=None)\
            .filter_var(pl.col._index.is_in(get_coding_genes()['gene']))\
            .with_columns_obs(
                dx_cc=pl.col.pmAD,
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
                apoe4_dosage=pl.coalesce([
                    'apoe_genotype_right', 'apoe_genotype'])
                    .cast(pl.String).str.count_matches('4')
                    .fill_null(strategy='mean'))
        pb.save(f'{sc_dir}/pseudobulk/{level}', overwrite=True)
        
del sc, pb; gc.collect()

# Gabitto et al. 2023 ############################################################
# sea-ad-single-cell-profiling.s3.amazonaws.com/index.html#DLPFC/RNAseq/
# portal.brain-map.org/explore/seattle-alzheimers-disease/seattle-alzheimers-disease-brain-cell-atlas-download?edit&language=en

study_name = 'SEAAD'
sc_dir = f'projects/def-wainberg/single-cell/{study_name}'

with Timer(f'[{study_name}] Loading single cell'):
    sc = SingleCell(
        f'{sc_dir}/SEAAD_DLPFC_RNAseq_all-nuclei.2024-02-13.h5ad',
        num_threads=None)\
        .cast_obs({'Donor ID': pl.String, 'Class': pl.String, 
                   'Subclass': pl.String})
    donor_metadata = pl.read_excel(
        f'{sc_dir}/sea-ad_cohort_donor_metadata.xlsx')
    pseudoprogression_scores = pl.read_csv(
        f'{sc_dir}/pseudoprogression_scores.csv')
    sc = sc\
        .join_obs(donor_metadata.select(['Donor ID'] +
            list(set(donor_metadata.columns).difference(sc.obs.columns))),
            on='Donor ID', validate='m:1')\
        .join_obs(pseudoprogression_scores, on='Donor ID', validate='m:1')\
        .with_columns_obs(
            pl.col('Doublet score').gt(0.5).alias('Is doublet'),
            pl.coalesce(
                pl.col.Subclass.replace({
                    'Astro': 'Astrocytes', 'Endo': 'Endothelial', 
                    'Micro-PVM': 'Microglia', 'Oligo': 'Oligodendrocytes',
                    'OPC': 'OPCs', 'VLMC': 'Endothelial'},
                    default=None),
                pl.col.Class.replace({
                    'exc': 'Excitatory', 'inh': 'Inhibitory'}, 
                    default=None))
                .alias('cell_type_broad'),
            pl.col.Subclass.alias('cell_type_fine'))\
        .qc(doublet_column='Is doublet',
            cell_type_confidence_column='Class confidence',
            custom_filter='Used in analysis',
            allow_float=True)

for level in ['broad', 'fine']:
    with Timer(f'[{study_name}] Pseudobulking at the {level} level'):
        QC_column = 'passed_QC_fine' if level == 'broad' else 'passed_QC'
        pb = sc\
            .pseudobulk(
                ID_column='Donor ID', 
                cell_type_column=f'cell_type_{level}',
                QC_column=QC_column,
                num_threads=None)\
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
                    .fill_null(strategy='mean'))
        if not os.path.exists(f'output/pseudobulk/{study_name}_{level}'):
            pb.save(f'{sc_dir}/pseudobulk/{study_name}_{level}')
        
del sc, pb; gc.collect()