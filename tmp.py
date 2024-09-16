import sys, polars as pl 
sys.path.append('/home/karbabi/projects/def-wainberg/karbabi/utils')
from single_cell import SingleCell, Pseudobulk
from utils import debug, print_df, get_coding_genes

debug(third_party=True)

sc = SingleCell(
    'projects/def-wainberg/single-cell/Green/Green_qced_labelled.h5ad',
    num_threads=None)

print(sc.obs['cell_type_broad'].dtype)
#Enum(categories=['Astrocyte', 'Excitatory', 'Endothelial', 'Inhibitory',
# 'Microglia-PVM', 'Oligodendrocyte', 'OPC'])

pb = sc\
    .pseudobulk(
        ID_column='projid',
        cell_type_column=f'cell_type_broad',
        QC_column=None,
        sort_genes=True,
        num_threads=None)
print(pb.X['Astrocyte'])
'''
[[  94  151    4 ... 2719  607  942]
 [ 102  113   19 ... 2324  762 1155]
 [  66   95    7 ... 2147  798  986]
 ...
 [  46  121    0 ... 1240  283  621]
 [ 333  527   16 ... 5214 1181 3743]
 [   2    3    0 ...   26   11   16]]
'''

sc_alt = sc.cast_obs({'cell_type_broad': pl.String})
pb = sc_alt\
    .pseudobulk(
        ID_column='projid',
        cell_type_column=f'cell_type_broad',
        QC_column=None,
        sort_genes=True,
        num_threads=None)
print(pb.X['Astrocyte'])

'''
[[ 0  2  0 ... 23  8 16]
 [ 0  0  0 ...  8  5  7]
 [ 3  0  0 ... 16 11 20]
 ...
 [ 4  4  0 ... 35 11 46]
 [ 3  6  0 ... 37 23 35]
 [ 0  0  0 ...  0  1  0]]
'''


rosmap_all = pl.read_csv(
    'projects/def-wainberg/single-cell/Green/rosmap_meta_all.csv')

sc = SingleCell(
    'projects/def-wainberg/single-cell/Green/Green_qced_labelled.h5ad',
    num_threads=None)

print(sc.obs['cell_type_broad'].dtype)
#Enum(categories=['Astrocyte', 'Excitatory', 'Endothelial', 'Inhibitory',
# 'Microglia-PVM', 'Oligodendrocyte', 'OPC'])

pb = sc_alt\
    .pseudobulk(
        ID_column='projid',
        cell_type_column=f'cell_type_broad',
        QC_column=None,
        sort_genes=True,
        num_threads=None)\
    .drop_obs('braaksc', 'ceradsc', 'niareagansc')\
    .cast_obs({'ID': pl.String})\
    .join_obs(
        rosmap_all.cast({'projid': pl.String}),
        left_on='ID', right_on='projid')\
    .with_columns_obs(
        pl.col('pmAD').alias('dx_cc'),
        pl.col.apoe_genotype.cast(pl.String) 
            .str.count_matches('4').fill_null(strategy='mean')
            .round()
            .alias('apoe4_dosage'),  
        pl.col.pmi.fill_null(strategy='mean') 
            .alias('pmi'),
        pl.when(pl.col.msex.eq(1)).then(pl.lit('M'))
            .when(pl.col.msex.eq(0)).then(pl.lit('F'))
            .otherwise(None)
            .cast(pl.Categorical(ordering='lexical'))
            .alias('sex'))\
    .filter_var(pl.col._index.is_in(get_coding_genes()['gene']))

pb_filt = pb\
    .qc(case_control_column='dx_cc',
        custom_filter=pl.col('dx_cc').is_not_null(),
        verbose=False)
print(pb_filt.X['Astrocyte'])
'''
[[  94    4  154 ... 2719  607  942]
 [ 102   19  206 ... 2324  762 1155]
 [  66    7  218 ... 2147  798  986]
 ...
 [ 234   19  354 ... 5245 1314 3696]
 [ 214    8  335 ... 6865 1111 3258]
 [  46    0   71 ... 1240  283  621]]
'''

de = pb_filt\
    .DE(label_column='dx_cc', 
        covariate_columns=['age_death', 'sex', 'pmi', 'apoe4_dosage'],
        case_control=True,
        verbose=False)
print_df(de.get_num_hits(threshold=0.1))
'''
 cell_type        num_hits 
 Astrocyte        1449     
 Endothelial      578      
 Excitatory       21       
 Inhibitory       606      
 Microglia-PVM    21       
 OPC              8        
 Oligodendrocyte  273     
'''














pb = sc_filt\
    .pseudobulk(
        ID_column='projid',
        cell_type_column='cell_type_fine',
        QC_column=None,
        sort_genes=True,
        num_threads=None)
print(pb.X['Endothelial'])

de = pb\
    .cast_obs({'msex': pl.String})\
    .cast_obs({'msex': pl.Categorical})\
    .DE(label_column='dx_cc', 
        covariate_columns=['age_death', 'msex', 'pmi', 'apoe4_dosage'],
        case_control=True,
        verbose=False)
print_df(de.get_num_hits(threshold=0.1))



print_df(sc.obs.group_by('cell_type_fine', 'passed_cell_type_fine')
         .len().sort('len'))

print_df(sc.obs.group_by('cell_type_broad', 'cell_type_fine').len()
         .filter(pl.col.cell_type_broad.is_in([
             'Astrocyte', 'Endothelial', 'Microglia-PVM', 
             'Oligodendrocyte', 'OPC'])).sort('len', descending=True))

print_df(sc.obs.filter(pl.col.cell_type_broad.eq('Endothelial'))
         .sort('_index')['_index'], 20)
print_df(sc.obs.filter(pl.col.cell_type_fine.eq('Endothelial'))
         .sort('_index')['_index'], 20)


sc = SingleCell(
    'projects/def-wainberg/single-cell/Green/Green_qced_labelled.h5ad',
    num_threads=None)

print_df(sc.obs.group_by('cell_type_broad', 'cell_type_fine')
         .len().filter(pl.col.cell_type_broad.eq('Endothelial'))
         .sort('len', descending=True))

sc_filt = sc.filter_obs(
    pl.col.cell_type_fine.eq('Endothelial') &
    pl.col.cell_type_broad.eq('Endothelial'))

pb = sc\
    .pseudobulk(
        ID_column='projid',
        cell_type_column='cell_type_fine',
        QC_column=None,
        sort_genes=True,
        num_threads=None)

print(pb.X['Endothelial'])

pb = sc_filt\
    .pseudobulk(
        ID_column='projid',
        cell_type_column='cell_type_fine',
        QC_column=None,
        sort_genes=True,
        num_threads=None)

print(pb.X['Endothelial'])


level = 'broad'
pb = sc_filt\
    .pseudobulk(
        ID_column='projid',
        cell_type_column=f'cell_type_{level}',
        QC_column=None,
        sort_genes=True,
        num_threads=None)\
    .drop_obs('braaksc', 'ceradsc', 'niareagansc')\
    .cast_obs({'ID': pl.String})\
    .join_obs(
        rosmap_all.cast({'projid': pl.String}),
        left_on='ID', right_on='projid')\
    .with_columns_obs(
        pl.col('pmAD').alias('dx_cc'),
        pl.when(pl.col.cogdx == 1).then(0)
            .when(pl.col.cogdx.is_in([2, 3])).then(1)
            .when(pl.col.cogdx.is_in([4, 5])).then(2)
            .otherwise(None)
            .alias('dx_cogn'),
        pl.col.apoe_genotype.cast(pl.String) # 1 missing filled
            .str.count_matches('4').fill_null(strategy='mean')
            .alias('apoe4_dosage'),  
        pl.col.pmi.fill_null(strategy='mean') # 1 missing filled
            .alias('pmi'))\
    .filter_var(pl.col._index.is_in(get_coding_genes()['gene']))

print(pb.obs['Endothelial']['ID'])
'''
[
        "21191200"
        "10669174"
        "57978756"
        "98953007"
        "3283241"
        …
        "90544686"
        "35072859"
        "5102083"
        "20522343"
        "21191200"
]
'''
print(pb.var['Endothelial']['_index'])
'''
[
        "A1BG"
        "A1CF"
        "A2M"
        "A2ML1"
        "A3GALT2"
        …
        "ZXDC"
        "ZYG11A"
        "ZYG11B"
        "ZYX"
        "ZZEF1"
]
'''
print(pb.X['Endothelial'])
'''
[[   0    0   60 ...  175    1  196]
 [   0    0   68 ...  154    7  141]
 [   0    0   46 ...  125   25  152]
 ...
 [  11    1   77 ...  374   88  482]
 [   7    3  160 ...  299   67  418]
 [ 103   17  200 ... 2947  987 1477]]
'''

pb_filt = pb\
    .select_cell_types('Endothelial')\
    .with_columns_obs(
        pl.col('msex')
        .cast(pl.String).cast(pl.Categorical))\
    .qc(case_control_column='dx_cc',
        custom_filter=pl.col('dx_cc').is_not_null(),
        verbose=True)

print(pb.X['Endothelial'])
'''
[[ 98   6   3 ...  23   9  12]
 [ 99   7   2 ...  12   6  11]
 [ 49  10   1 ...   7   5   6]
 ...
 [147   5   4 ...  15  17  13]
 [ 71   1   3 ...   6   5   2]
 [398  21   9 ...  29  10  32]]
'''

de = pb_filt\
    .DE(label_column='dx_cc', 
        covariate_columns=['age_death', 'msex', 'pmi', 'apoe4_dosage'],
        case_control=True,
        verbose=False)

print_df(de.get_num_hits(threshold=0.1))
'''
 cell_type    num_hits 
 Endothelial  578      
'''

print_df(de.get_hits(threshold=0.1), 5)
'''
 cell_type  gene   logFC     SE        LCI        UCI        AveExpr   P         Bonferro  FDR      
                                                                                 ni                 
 Endotheli  HES4   -0.44112  0.988831  -0.584236  -0.298005  3.90893   3.0640e-  0.000047  0.000047 
 al                                                                    9                            
 Endotheli  WDR64  0.751135  2.760351  0.499091   1.00318    3.442422  9.4995e-  0.000146  0.000073 
 al                                                                    9                            
 Endotheli  NRIP2  0.322248  2.019211  0.212461   0.432035   1.874637  1.5486e-  0.000238  0.000079 
 al                                                                    8                            
 …          …      …         …         …          …          …         …         …         …        
 Endotheli  MAGI3  0.07542   0.568008  0.022193   0.128647   7.901898  0.005592  1.0       0.099921 
 al                                                                                                 
 Endotheli  VIM    0.441796  0.546829  0.12997    0.753622   1.56694   0.005597  1.0       0.099921 
 al 
'''