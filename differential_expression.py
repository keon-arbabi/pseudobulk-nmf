import sys, os, pickle, polars as pl
sys.path.append('/home/karbabi/projects/def-wainberg/karbabi/utils')
from single_cell import Pseudobulk, DE
from utils import Timer, print_df, debug

Pseudobulk.num_threads = -1
debug(third_party=True)

data_dir = 'projects/def-wainberg/single-cell'
working_dir = 'projects/def-wainberg/karbabi/pseudobulk-nmf' 

study_names = ['Green', 'Mathys', 'SEAAD']
covariates = {
    'Green': ['age_death', 'sex', 'pmi', 'apoe4_dosage'],
    'Mathys': ['age_death', 'sex', 'pmi', 'apoe4_dosage'],
    'SEAAD': ['Age at Death', 'Sex', 'PMI', 'apoe4_dosage']
}
dx_column = 'dx_cc'

de_results = {}
for study in study_names:
    for level in ['broad', 'fine']:
        with Timer(f'[{study}] differential expression '
                   f'at {level} level on {dx_column}'):
            pb = Pseudobulk(f'{data_dir}/{study}/pseudobulk/{level}')\
                .qc(case_control_column=dx_column,
                    custom_filter=pl.col(dx_column).is_not_null(),
                    verbose=False)
            de = pb\
                .DE(label_column=dx_column, 
                    covariate_columns=covariates[study],
                    case_control=True,
                    verbose=False)
            de_results[study, level] = de

            save_name = f'{study}_{level}_{dx_column}'
            os.makedirs(f'{working_dir}/output/DE/{save_name}', exist_ok=True)
            os.makedirs(f'{working_dir}/figures/DE/voom/{save_name}', 
                        exist_ok=True)
            
            de.save(f'{working_dir}/output/DE/{save_name}', overwrite=True)
            for cell_type in pb.keys():
                de.plot_voom(cell_type,
                             f'{working_dir}/figures/DE/voom/{save_name}/'
                             f'{cell_type.replace('/', '_')}.png')
            
            print(save_name)    
            print_df(de.get_num_hits(threshold=0.1).sort('cell_type'))

with open(f'{working_dir}/output/DE/de_results.pkl', 'wb') as f:
    pickle.dump(de_results, f)
with open(f'{working_dir}/output/DE/de_results.pkl', 'rb') as f:
    de_results = pickle.load(f)










def overlap(cell_type, fdr_threshold=0.05):
    broad = set(de_results['Green', 'broad'].table\
                .filter((pl.col('cell_type') == cell_type) & 
                        (pl.col('FDR') < fdr_threshold))['gene'])
    fine = set(de_results['Green', 'fine'].table\
               .filter((pl.col('cell_type') == cell_type) & 
                       (pl.col('FDR') < fdr_threshold))['gene'])
    return len(broad & fine)

cell_type = 'Endothelial'
print(de_results['Green', 'fine'].get_num_hits(threshold=0.1).filter(
    pl.col('cell_type') == cell_type))
print(de_results['Green', 'broad'].get_num_hits(threshold=0.1).filter(
    pl.col('cell_type') == cell_type))
f'Overlapping genes in Astrocyte (FDR < 0.05): ' \
    f'{overlap(cell_type, 0.1)}'


'''
[Green] differential expression at broad level on dx_cc...
Green_broad_dx_cc
 cell_type        num_hits 
 Astrocyte        13       
 Endothelial      659      
 Excitatory       1485     
 Inhibitory       606      
 Microglia-PVM    21       
 OPC              8        
 Oligodendrocyte  273      

Green_fine_dx_cc
 cell_type        num_hits 
 Astrocyte        5        
 Chandelier       9        
 Endothelial      859      
 L2/3 IT          1508     
 L4 IT            2        
 L5 ET            104      
 L5 IT            1465     
 L5/6 NP          1        
 L6 CT            10       
 L6 IT            4        
 L6b              15       
 Lamp5            180      
 Lamp5 Lhx6       10       
 Microglia-PVM    199      
 OPC              55       
 Oligodendrocyte  340      
 Pvalb            458      
 Sst              2        
 VLMC             459      
 Vip              52       

Mathys_broad_dx_cc
 cell_type        num_hits 
 Astrocyte        361      
 Endothelial      831      
 Inhibitory       4        
 Microglia-PVM    8        
 OPC              27       
 Oligodendrocyte  2        

Mathys_fine_dx_cc
 cell_type        num_hits 
 Astrocyte        236      
 L4 IT            111      
 L5 IT            419      
 L5/6 NP          8        
 L6 CT            261      
 L6 IT            18       
 L6 IT Car3       5        
 L6b              252      
 Lamp5            18       
 Lamp5 Lhx6       14       
 Microglia-PVM    25       
 OPC              72       
 Oligodendrocyte  69       
 Pvalb            37       
 Sst              14       
 VLMC             96       
 Vip              230      
'''

'''
Green_broad_pmAD
 cell_type         num_hits 
 Astrocytes        630      
 Endothelial       16       
 Excitatory        1353     
 Inhibitory        600      
 Microglia         26       
 OPCs              7        
 Oligodendrocytes  243      

Green_fine_pmAD
 cell_type        num_hits 
 Astrocyte        646      
 Chandelier       27       
 Endothelial      6        
 L2/3 IT          756      
 L4 IT            1395     
 L5 IT            1536     
 L5/6 NP          134      
 L6 CT            61       
 L6 IT            742      
 L6 IT Car3       15       
 L6b              336      
 Lamp5            76       
 Lamp5 Lhx6       14       
 Microglia-PVM    21       
 OPC              7        
 Oligodendrocyte  288      
 Pvalb            361      
 Sst              3        
 VLMC             1        
 Vip              84 

 
Green_broad_dx_cont
 cell_type         num_hits 
 Astrocytes        697      
 Endothelial       108      
 Excitatory        874      
 Inhibitory        226      
 Microglia         18       
 OPCs              125      
 Oligodendrocytes  320  


Green_fine_dx_cont
 cell_type        num_hits 
 Astrocyte        723      
 Chandelier       33       
 Endothelial      26       
 L2/3 IT          366      
 L4 IT            1507     
 L5 IT            1287     
 L5/6 NP          194      
 L6 CT            122      
 L6 IT            846      
 L6 IT Car3       36       
 L6b              377      
 Lamp5            79       
 Lamp5 Lhx6       15       
 Microglia-PVM    20       
 OPC              106      
 Oligodendrocyte  304      
 Pvalb            261      
 Sst              9        
 VLMC             1        
 Vip              116  
'''
