import sys, polars as pl
sys.path.append('/home/karbabi/projects/def-wainberg/karbabi/utils')
from single_cell import SingleCell
from utils import Timer, print_df, debug, reload 

debug(third_party=True)
import single_cell; reload(single_cell)

study_name = 'Green'

with Timer(f'[{study_name}] Preprocessing single cell'):
    sc = SingleCell('projects/def-wainberg/single-cell/Temp/p400_50K.h5ad',
                     num_threads=None)\
        .with_columns_obs(
            projid=pl.col.projid.cast(pl.String),
            cell_type_broad=pl.col.subset.cast(pl.String).replace(
                {'CUX2+': 'Excitatory'}))\
        .qc(custom_filter= pl.col('cell.type.prob').ge(0.9) & 
                pl.col.projid.is_not_null() &
                pl.col('is.doublet.df').not_(),
            subset=True,
            num_threads=None)

with Timer('[Mini SEAAD] Loading single cell'):
    sc_ref = SingleCell(
        'projects/def-wainberg/single-cell/Temp/mini_SEAAD_50k.h5ad',
        num_threads=None)\
        .qc(custom_filter=pl.col('subclass_confidence').ge(0.9),
            subset=True,
            allow_float=True,
            num_threads=None)
    
with Timer(f'[{study_name}] Highly-variable genes'):
    sc, sc_ref = sc.hvg(sc_ref, allow_float=True, num_threads=None)

with Timer(f'[{study_name}] Normalize'):
    sc = sc.normalize(allow_float=True, num_threads=None)
    sc_ref = sc_ref.normalize(allow_float=True, num_threads=None)

with Timer(f'[{study_name}] PCA'):
    sc, sc_ref = sc.PCA(sc_ref, verbose=True, num_threads=None)

with Timer(f'[{study_name}] Harmony'):
    sc, sc_ref = sc.harmonize(sc_ref, num_threads=None)

with Timer(f'[{study_name}] Label transfer'):
    sc = sc.label_transfer_from(
        sc_ref, 
        original_cell_type_column='subclass_label',
        cell_type_column='cell_type_fine',
        cell_type_confidence_column='cell_type_fine_confidence',
        num_index_neighbors=100)\
        .with_columns_obs(
            passed_QC_fine=pl.col.cell_type_fine_confidence.ge(0.9))
    
print_df(sc.obs.group_by('cell_type_fine')
        .agg(mean=pl.col('cell_type_fine_confidence').mean(),
             count=pl.col('cell_type_fine_confidence').count())
        .sort('mean'))

with Timer(f'[{study_name}] PaCMAP'):
    sc = sc.embed(QC_column=None, num_threads=None)

'''
 cell_type_fine   mean      count 
 Sst Chodl        0.803125  16    
 Endothelial      0.827592  299   
 VLMC             0.86996   248   
 L2/3 IT          0.883767  11107 
 Sncg             0.892033  364   
 Pax6             0.927128  188   
 Sst              0.94233   1747  
 L5 IT            0.958571  2351  
 Lamp5            0.965779  732   
 Pvalb            0.985179  2014  
 L4 IT            0.986183  3076  
 Lamp5 Lhx6       0.990393  229   
 Vip              0.99063   2001  
 L6 IT            0.998447  837   
 OPC              1.0       1833  
 L6 CT            1.0       334   
 Chandelier       1.0       294   
 Oligodendrocyte  1.0       9769  
 L5/6 NP          1.0       453   
 Astrocyte        1.0       6620  
 L6 IT Car3       1.0       335   
 L6b              1.0       418   
 Microglia-PVM    1.0       2508  
'''

