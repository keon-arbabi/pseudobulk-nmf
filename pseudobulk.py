import sys, os, gc, polars as pl

sys.path.append('/home/karbabi/projects/def-wainberg/karbabi/utils')
from single_cell import SingleCell
from utils import Timer, get_coding_genes
    
os.chdir('projects/def-wainberg/karbabi/pseudobulk-nmf')
os.makedirs('output/pseudobulk', exist_ok=True)

# Green et al. 2023 ############################################################
save_name = 'Green_broad'

with Timer(f'[{save_name}] loading single cell'):
    sc = SingleCell('../../single-cell/Green/p400_qced_shareable.h5ad',
            num_threads=os.cpu_count())\
        .with_columns_obs(
            cell_type_broad=pl.col.subset.replace(
                {'CUX2+': 'Excitatory'}),
            projid=pl.col.projid.cast(pl.String))\
        .qc(cell_type_confidence_column='cell.type.prob',
            doublet_column='is.doublet.df',
            custom_filter=pl.col.projid.is_not_null())

with Timer(f'[{save_name}] pseudobulking'):
    # radc.rush.edu/docs/var/variables.htm
    rosmap_meta = pl.read_csv('../../single-cell/Green/'
            'dataset_978_basic_04-21-2023_with_pmAD.csv',
            dtypes={'projid': pl.String})\
        .unique(subset='projid')\
        .drop([col for col in sc.obs.columns if col != 'projid'])
    pb = sc\
        .pseudobulk(ID_column='projid', 
            cell_type_column='cell_type_broad',
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

del sc, pb; gc.collect()

# Mathys et al. 2023 ############################################################
save_name = 'Mathys_broad'

with Timer(f'[{save_name}] loading single cell'):
    data_dir = '../../single-cell/Mathys'
    # cells.ucsc.edu/ad-aging-brain/ad-aging-brain/meta.tsv
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
    # synapse.org/#!Synapse:syn21323366
    id_map1 = pl.read_csv(
        f'{data_dir}/MIT_ROSMAP_Multiomics_individual_metadata.csv',
        columns=['individualID', 'individualIdSource', 'subject'])\
        .filter(pl.col.subject.is_not_null())\
        .unique(subset='subject')
    # synapse.org/#!Synapse:syn3191087
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
                .fill_null(strategy='mean'))
    if not os.path.exists(f'output/pseudobulk/{save_name}'):
        pb.save(f'output/pseudobulk/{save_name}')
        
del sc, pb; gc.collect()

# Gabitto et al. 2023 ############################################################
save_name = 'Gabitto_fine'

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
        .with_columns_obs(pl.col('Doublet score').gt(0.5).alias('Is doublet'))\
        .qc(doublet_column='Is doublet',
            cell_type_confidence_column='Subclass confidence',
            custom_filter='Used in analysis',
            allow_float=True)

with Timer(f'[{save_name}] pseudobulking'):
    pb = sc\
        .pseudobulk(ID_column='Donor ID', 
            cell_type_column='Subclass',
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
                .fill_null(strategy='mean'))
    if not os.path.exists(f'output/pseudobulk/{save_name}'):
        pb.save(f'output/pseudobulk/{save_name}')
        
del sc, pb; gc.collect()