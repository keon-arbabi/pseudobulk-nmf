import sys, os
import polars as pl, pandas as pd, numpy as np
import matplotlib.pylab as plt
import seaborn as sns

os.chdir('projects/def-wainberg/karbabi/single-cell-nmf')
from utils import Timer, print_df, print_row, debug, \
    SingleCell, Pseudobulk
from ryp import r, to_py, to_r    

with Timer('Load single-cell data'):
    sc = SingleCell('data/single-cell/Maitra/matrix.mtx.gz')
    meta = pl.read_csv('data/single-cell/Maitra/meta.tsv', separator='\t')\
        .with_columns(pl.col.Cell.str.replace(r"\..*", "").alias("ID"))
        
    sc.obs = meta

with Timer('QC single-cell data and pseudobulk'):
    pb = sc\
        .pseudobulk(ID_column='ID', 
                    cell_type_column='Cluster',
                    QC_column=None)

pb['Ast1', :, :]