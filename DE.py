import sys, os, polars as pl

sys.path.append('/home/karbabi/projects/def-wainberg/karbabi/utils')
from single_cell import Pseudobulk, DE
from utils import Timer, print_df
    
os.chdir('projects/def-wainberg/karbabi/pseudobulk-nmf')
os.makedirs('output/DE', exist_ok=True)
os.makedirs('figures/DE/voom', exist_ok=True)

resolution = 'broad'
study_names = ['Green', 'Mathys', 'Gabitto']
dx_column = {
    'Green': 'pmAD', 'Mathys': 'pmAD', 'Gabitto': 'dx_cc'}
covariates = {
    'Green': ['age_death', 'sex', 'pmi', 'apoe4_dosage', 'log_num_cells'],
    'Mathys': ['age_death', 'sex', 'pmi', 'apoe4_dosage', 'log_num_cells'],
    'Gabitto': ['Age at Death', 'Sex', 'PMI', 'apoe4_dosage', 'log_num_cells']}

for study in study_names:
    with Timer(f'[{study}] differential expression'):
        de = Pseudobulk(f'output/pseudobulk/{study}_{resolution}')\
            .qc(case_control_column=None, 
                custom_filter=pl.col(dx_column[study]).is_not_null())\
            .DE(label_column=dx_column[study], 
                case_control=False,
                covariate_columns=covariates[study],
                include_library_size_as_covariate=True)
        save_name = f'{study}_{resolution}_{dx_column[study]}'
        de.plot_voom(save_to=f'figures/DE/voom/{save_name}', overwrite=True)
        de.save(f'output/DE/{save_name}', overwrite=True)    
        print_df(de.get_num_hits(threshold=0.05).sort('cell_type'))



de_1 = DE(f'output/DE/Green_{resolution}_{dx_column['Green']}').table 
de_2 = DE(f'output/DE/Mathys_{resolution}_{dx_column['Mathys']}').table

de_1.join(de_2, on=['cell_type', 'gene'], how='inner')\
    .with_columns(same_sign = pl.col.logFC.mul(pl.col.logFC_right) > 0)\
    .group_by('cell_type')\
    .agg(n_overlaps_same_dir = pl.col.gene.filter(
        pl.col.FDR.lt(0.05) & pl.col.same_sign).n_unique(),
        n_overlaps = pl.col.gene.filter(
        pl.col.FDR.lt(0.05)).n_unique(),
        ratio = pl.col.gene.filter(
        pl.col.FDR.lt(0.05) & pl.col.same_sign).n_unique() /
        pl.col.gene.filter(
        pl.col.FDR.lt(0.05)).n_unique())\
    .sort('cell_type')
    
print_df(de.get_num_hits(threshold=0.2).sort('cell_type'))

'''
 cell_type         n_overlaps_same_dir  n_overlaps  ratio    
 Astrocytes        80                   145         0.551724 
 Endothelial       1                    1           1.0      
 Excitatory        301                  572         0.526224 
 Inhibitory        63                   131         0.480916 
 Microglia         3                    5           0.6      
 Oligodendrocytes  38                   74          0.513514 

Green
cell_type         num_hits 
 Astrocytes        238      
 Endothelial       1        
 Excitatory        704      
 Inhibitory        173      
 Microglia         8        
 OPCs              2        
 Oligodendrocytes  113    

Mathys 
cell_type         num_hits 
 Astrocytes        211      
 Endothelial       1        
 Excitatory        14       
 Inhibitory        2        
 Microglia         8        
 Oligodendrocytes  450      
 Opc               10  
 
Gabitto 
cell_type   num_hits 
 Inhibitory  26        
 
'''

