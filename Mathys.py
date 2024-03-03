import sys, os
import polars as pl, pandas as pd, numpy as np

os.chdir('projects/def-wainberg/karbabi/single-cell-nmf')
from utils import Timer, get_coding_genes, debug, \
    print_df, print_row, \
    SingleCell, Pseudobulk

debug(third_party=True)

with Timer('Load single-cell data'):
    sc = SingleCell('data/single-cell/Mathys/PFC427_raw_data.h5ad')
    
with Timer('QC single-cell data and pseudobulk'):
    pb = sc\
        .qc(cell_type_confidence_column='cell.type.prob',
            doublet_confidence_column='doublet.score',
            custom_filter=pl.col.projid.is_not_null())\
        .with_columns_obs(projid=pl.col.projid.cast(pl.String))\
        .pseudobulk(ID_column='projid', 
                    cell_type_column='subset',
                    additional_obs=rosmap_pheno)

with Timer('QC pseudobulk data'):
    pb = Pseudobulk('data/pseudobulk/p400')
    pb = pb\
        .filter_var(pl.col._index.is_in(get_coding_genes()['gene']))\
        .with_columns_obs(
            apoe4_dosage=pl.col.apoe_genotype.cast(pl.String)\
                .str.count_matches('4').fill_null(strategy='mean'),
            pmi=pl.col.pmi.fill_null(strategy='mean'))\
        .qc(case_control_column='pmAD', 
            custom_filter=pl.col.pmAD.is_not_null())
        