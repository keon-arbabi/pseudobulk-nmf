import sys, os, polars as pl

sys.path.append('/home/karbabi/projects/def-wainberg/karbabi/utils')
from single_cell import Pseudobulk, DE
from utils import Timer, print_df
    
os.chdir('projects/def-wainberg/karbabi/pseudobulk-nmf')
os.makedirs('output/DE', exist_ok=True)
os.makedirs('figures/DE/voom', exist_ok=True)

resolution = 'broad'
study_names = ['Green', 'Mathys']
dx_column = {
    'Green': 'dx_cont', 'Mathys': 'dx_cont', 'Gabitto': 'dx_cont'}
covariates = {
    'Green': ['age_death', 'sex', 'pmi', 'apoe4_dosage', 'num_cells'],
    'Mathys': ['age_death', 'sex', 'pmi', 'apoe4_dosage', 'num_cells'],
    'Gabitto': ['Age at Death', 'Sex', 'PMI', 'apoe4_dosage', 'num_cells']}

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
        print_df(de.get_num_hits().sort('cell_type'))
        
de_green = DE(f'output/DE/Green_{resolution}_{dx_column['Green']}')       
        

'''
Green
cont
cell_type         num_hits 
 Astrocytes        322      
 Endothelial       1        
 Excitatory        689      
 Inhibitory        169      
 Microglia         8        
 OPCs              2        
 Oligodendrocytes  184      

cc
cell_type         num_hits 
 Astrocytes        348      
 Endothelial       13       
 Excitatory        398      
 Inhibitory        86       
 Microglia         9        
 OPCs              43       
 Oligodendrocytes  139  

Mathys 
cont
 cell_type  num_hits 
 Ast        189      
 Exc        15       
 Inh        2        
 Mic        12       
 Oli        420      
 Opc        10       
 Vas        1   
 
cc
cell_type  num_hits 
 Ast        573      
 Exc        14       
 Inh        2        
 Mic        29       
 Oli        641      
 Opc        47       
 Vas        2    
'''