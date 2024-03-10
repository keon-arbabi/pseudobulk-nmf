library(tidyverse)
library(umap)
library(corrplot)
library(ggsci)
setwd('projects/def-wainberg/karbabi/single-cell-nmf')

A = read_tsv(file='results/NMF/A/p400/Combined_A.tsv') 

W = read_tsv(file='results/NMF/factors/p400/Combined_W.tsv')
W = read_tsv(file='results/NMF/factors/p400/Combined_W_NNDSVD.tsv')

H = read_tsv(file='results/NMF/factors/p400/Combined_H.tsv')
H = read_tsv(file='results/NMF/factors/p400/Combined_H_NNDSVD.tsv')

# flip ceradsc and niaregansc

H = column_to_rownames(H, var = "ID")
meta = read_tsv('results/NMF/A/meta.tsv') %>%
  select(ID, num_cells, sex, Cdx, braaksc, ceradsc, pmi, niareagansc, 
        apoe4_dosage, tomm40_hap, age_death, age_first_ad_dx, cogdx, ad_reagan,
        gpath, amyloid, hspath_typ, dlbdx, tangles, tdp_st4, arteriol_scler,
        caa_4gp, cvda_4gp2, ci_num2_gct, ci_num2_mct) %>%
  column_to_rownames(var = 'ID') %>%
  mutate(sex = as.numeric(as.factor(sex))) %>%
  mutate(across(where(is.numeric), 
        ~ifelse(is.na(.), median(., na.rm = TRUE), .)))

cor_mat = t(cor(H, meta))
p_mat = matrix(NA, ncol(H), ncol(meta))
for (i in 1:ncol(H)) {
  for (j in 1:ncol(meta)) {
    p_mat[i, j] = cor.test(H[, i], meta[, j])$p.value
  }
}
p_mat = t(p_mat)
rownames(p_mat) = rownames(cor_mat)
colnames(p_mat) = colnames(cor_mat)

png(filename='corrplot.png', width = 5.5, height = 10, units = "in", res=300)
corrplot(cor_mat, is.corr = FALSE,  
        p.mat = p_mat, sig.level = 0.05,
        insig = 'label_sig', pch.cex = 2, pch.col = "white",
        tl.col = "black", col.lim = c(-0.4, 0.4))
dev.off()












umap_setting = umap.defaults
umap_setting$spread = 1.5 
umap_setting$n_neighbors = 15 
umap_setting$min_dist = 0.1 

subtype_membership = names(H)[-1][apply(H[-1], 1, which.max)]

umap_df = as.data.frame(umap(H[-1], umap_setting)$layout)
umap_df$ID = H$ID
umap_df$subtype = subtype_membership 

ggplot(umap_df, aes(x = V1, y = V2, color = subtype)) +
    geom_point(size = 5) +
    scale_color_d3() +
    labs(x = "UMAP 1", y = "UMAP 2", color = "Subtype") + 
    theme_classic() +
    theme(axis.text = element_text(size = 12, color = "black"),
        axis.title = element_text(size = 16, face = "bold"))
        

pca_result = prcomp(H[-1], center = TRUE, scale. = TRUE) # Standardize data
pca_df = as.data.frame(pca_result$x)
pca_df$ID = H$ID
pca_df$subtype = subtype_membership 

ggplot(pca_df, aes(x = PC1, y = PC2, color = subtype)) +
    geom_point(size = 5) +
    scale_color_d3() +
    labs(x = "PCA 1", y = "PCA 2", color = "Subtype") + 
    theme_classic() +
    theme(axis.text = element_text(size = 12, color = "black"),
          axis.title = element_text(size = 16, face = "bold"))
