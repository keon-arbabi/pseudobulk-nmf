import sys, os, polars as pl
sys.path.append('/home/karbabi/projects/def-wainberg/karbabi/utils')
from single_cell import Pseudobulk, DE
from utils import Timer, print_df

sc_dir = 'projects/def-wainberg/single-cell'
working_dir = 'projects/def-wainberg/karbabi/pseudobulk-nmf' 
os.makedirs(f'{working_dir}/output/DE', exist_ok=True)
os.makedirs(f'{working_dir}/figures/DE/voom', exist_ok=True)

#TODO: double-check dtypes (make sure categoricals )

study_names = ['Green', 'Mathys', 'SEAAD']
covariates = {
    'Green': ['age_death', 'sex', 'pmi', 'apoe4_dosage'],
    'Mathys': ['age_death', 'sex', 'pmi', 'apoe4_dosage'],
    'SEAAD': ['Age at Death', 'Sex', 'PMI', 'apoe4_dosage']}

de_results = {}
for type in ['cc', 'cont']:
    dx_column = {
        'Green': 'pmAD' if type == 'cc' else 'dx_cont',
        'Mathys': 'pmAD' if type == 'cc' else 'dx_cont',
        'SEAAD': 'dx_cc' if type == 'cc' else 'dx_cont'}
    for study in study_names:
        for level in ['broad', 'fine']:
            with Timer(f'[{study}] differential expression at {level} level'):
                de = Pseudobulk(f'{sc_dir}/{study}/pseudobulk/{level}')\
                    .qc(case_control_column=\
                        dx_column[study] if type == 'cc' else None, 
                        custom_filter=pl.col(dx_column[study]).is_not_null(),
                        verbose=False)\
                    .DE(label_column=dx_column[study], 
                        case_control=type == 'cc',
                        covariate_columns=covariates[study],
                        verbose=False)
                de_results[type, study, level] = de
                save_name = f'{study}_{level}_{dx_column[study]}'
                de.plot_voom(f'{working_dir}/figures/DE/voom/{save_name}', 
                            overwrite=True, PNG=True)
                de.save(f'{working_dir}/output/DE/{save_name}', overwrite=True)
                print(save_name)    
                print_df(de.get_num_hits(threshold=0.1).sort('cell_type'))


'''
 cell_type         num_hits 
 Astrocytes        630      
 Endothelial       16       
 Excitatory        1353     
 Inhibitory        600      
 Microglia         26       
 OPCs              7        
 Oligodendrocytes  243   
 
'''


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

