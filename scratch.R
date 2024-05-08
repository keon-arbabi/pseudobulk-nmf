suppressPackageStartupMessages({
  library(tidyverse)
  library(umap) 
  library(corrplot)
  library(ComplexHeatmap)
  library(circlize)
  library(seriation)
  library(ggpubr)
  library(patchwork)
  library(scico)
  library(ggsci)
  library(ggbeeswarm)
  library(viridis)
})

setwd('projects/def-wainberg/karbabi/single-cell-nmf')

A = read_tsv(file='results/NMF/A/p400_A_fdr05.tsv') 
W = read_tsv(file='results/NMF/factors/p400_W_fdr05.tsv')
H = read_tsv(file='results/NMF/factors/p400_H_fdr05.tsv') %>%
  column_to_rownames(var = "ID")
cell_types = read_tsv(file='results/NMF/A/p400_celltypes.tsv', 
  col_names = "cell_type", skip = 1)
cell_types = cell_types$cell_type

meta = read_tsv('results/NMF/A/p400_metadata.tsv') %>%
  select(ID, num_cells, sex, Cdx, braaksc, ceradsc, pmi, niareagansc, 
        apoe4_dosage, tomm40_hap, age_death, age_first_ad_dx, gpath,
        amyloid, hspath_typ, dlbdx, tangles, tdp_st4, arteriol_scler,
        caa_4gp, cvda_4gp2, ci_num2_gct, ci_num2_mct) %>%
  rename(c("Number of cells" = num_cells, "Cognitive diganosis" = Cdx, 
        "Sex" = sex, "Braak stage" = braaksc, "Cerad score" = ceradsc, 
        "PMI" = pmi, "NIA-Reagan diagnosis" = niareagansc, 
        "APOE4 dosage" = apoe4_dosage, "TOMM40 haplotype" = tomm40_hap, 
        "Age of death" = age_death, "Age of diagnosis" = age_first_ad_dx,
        "Global AD pathology" = gpath, "Amyloid level" = amyloid, 
        "Hippocampal sclerosis" = hspath_typ, "Lewy body disease" = dlbdx, 
        "Neurofibrillary tangles" = tangles, "TDP-43 IHC" = tdp_st4, 
        "Arteriolosclerosis" = arteriol_scler,
        "Cerebral amyloid angiopathy" = caa_4gp, 
        "Cerebral atherosclerosis" = cvda_4gp2, 
        "Chronic infarcts" = ci_num2_gct,
        "Chronic microinfarcts" = ci_num2_mct)) %>%
  column_to_rownames(var = 'ID') %>%
  mutate(Sex = as.numeric(as.factor(Sex))) %>%
  mutate(across(where(is.numeric), 
        ~ifelse(is.na(.), median(., na.rm = TRUE), .))) %>%
  mutate(`Cerad score` = rev(`Cerad score`), 
        `NIA-Reagan diagnosis` = rev(`NIA-Reagan diagnosis`))

cor_mat = t(cor(H, meta))
p_mat = matrix(NA, ncol(H), ncol(meta))
for (i in 1:ncol(H)) {
  for (j in 1:ncol(meta)) {
    p_mat[i, j] = cor.test(H[, i], meta[, j])$p.value
  }
}
p_mat = t(p_mat)

row_order = get_order(seriate(dist(cor_mat), method = "OLO"))
cor_mat = cor_mat[as.numeric(row_order),]
p_mat = p_mat[as.numeric(row_order),]
rownames(p_mat) = rownames(cor_mat)
colnames(p_mat) = colnames(cor_mat)

png(filename='corrplot.png', width = 7, height = 10, units = "in", res=300)
corrplot(cor_mat, is.corr = FALSE,  
        p.mat = p_mat, sig.level = 0.05,
        insig = 'label_sig', pch.cex = 2, pch.col = "white",
        tl.col = "black")
dev.off()

df = left_join(H %>% rownames_to_column(var = "ID"),
              meta %>% rownames_to_column(var = "ID"), "ID") %>%
      pivot_longer(cols = matches("^S\\d+"), names_to = "subtype", 
                  values_to = "loading") %>%
      select(ID, subtype, loading, everything())

p1 = df %>% 
  filter(subtype %in% c("S0", "S1", "S2")) %>%
  ggplot(., aes(x = loading, y = `Global AD pathology`, color = subtype)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE, lwd = 1.1) +
  stat_cor(method = "pearson") +
  scale_color_locuszoom()+ 
  labs(x = "", color = "Subtype") + 
  facet_wrap(~ subtype) + 
  theme_classic() +
  theme(legend.position = "None") 

p2 = df %>% 
  mutate(`TDP-43 IHC` = factor(`TDP-43 IHC`)) %>%
  filter(subtype %in% c("S0", "S1", "S2")) %>%
  ggplot(., aes(x = `TDP-43 IHC`, y = loading, fill = subtype)) +
  geom_boxplot() + 
  geom_quasirandom(size = 2) +
  scale_fill_locuszoom() +
  labs(y = "", color = "Subtype") + 
  coord_flip() +
  facet_wrap(~ subtype) + 
  theme_classic()

p4 = df %>% 
  mutate(`Braak stage` = factor(`Braak stage`)) %>%
  filter(subtype %in% c("S0", "S1", "S2")) %>%
  ggplot(., aes(x = `Braak stage`, y = loading, fill = subtype)) +
  geom_boxplot() + 
  geom_quasirandom(size = 2) +
  scale_fill_locuszoom() +
  labs(y = "", color = "Subtype") + 
  coord_flip() +
  facet_wrap(~ subtype) + 
  theme_classic()

p3 = df %>% 
  mutate(Sex = factor(Sex, levels = c(1, 2), labels = c("Female", "Male"))) %>%
  filter(subtype %in% c("S0", "S1", "S2")) %>%
  ggplot(., aes(y = loading,  x = Sex, fill = subtype)) +
  geom_boxplot() + 
  geom_quasirandom(size = 2) +
  scale_fill_locuszoom() +
  labs(x = "Sex", y = "Subtype Loading", color = "Subtype") + 
  coord_flip() +
  facet_wrap(~ subtype) + 
  theme_classic() +
  theme(legend.position = "None") 

p1 / p2 / p3 
ggsave(file="subtype_meta_examples.png", width = 9, height = 12, units = "in")

top_genes = lapply(W[,-1], function(column) {
  index = head(order(column, decreasing = TRUE), nrow(W))
  data.frame(gene = W$gene[index], score = column[index], 
    cell_type = cell_types[index])
})
names(top_genes) = names(W)[-1]

hist(top_genes$S0[['score']], breaks=100)
top_genes$S0 %>% filter(score < 0.025) %>% pull(gene)

W %>%
  select(gene, S0) %>%
  mutate(cell_type = cell_types) %>%
  group_by(cell_type) %>%
  filter(n() >= 30) %>%
  ggplot(., aes(x = cell_type, y = S0, color = cell_type)) +
  geom_boxplot() + 
  geom_quasirandom(size = 1) +
  scale_color_frontiers()+ 
  labs(x = "Cell Type", color = "Cell Type") + 
  coord_flip() +
  theme_classic() + 
  theme(legend.position = "None") 
ggsave(file="cell_type_enrich.png", width = 5, height = 6, units = "in")



umap_df = as.data.frame(umap(scale(H))$layout)
umap_df$ID = rownames(H)
umap_df$subtype = names(H)[apply(H, 1, which.max)] 
umap_df = left_join(umap_df, meta %>% rownames_to_column(var = "ID"), by = "ID")

p1 = ggplot(umap_df, aes(x = V1, y = V2, color = subtype)) +
    geom_point(size = 5) +
    scale_color_frontiers() +
    labs(x = "UMAP 1", y = "UMAP 2", color = "Subtype") + 
    theme_classic() +
    theme(axis.text = element_text(size = 12, color = "black"),
        axis.title = element_text(size = 16, face = "bold"))

p2 = ggplot(umap_df, aes(x = V1, y = V2, color = `Global AD pathology` )) +
    geom_point(size = 4) +
    scale_color_viridis(option = "rocket", direction = -1) +
    labs(x = "UMAP 1", y = "UMAP 2") +  
    theme_classic() +
    theme(axis.text = element_text(size = 12, color = "black"),
        axis.title = element_text(size = 16, face = "bold"))
p1 + p2
ggsave(file="subtype_umap.png", width = 16, height = 8, units = "in")

pca_result = prcomp(H, center = F, scale = T) 
pca_df = as.data.frame(pca_result$x)
pca_df$ID = H$ID
pca_df$subtype = subtype_membership 

ggplot(pca_df, aes(x = PC1, y = PC2, color = subtype)) +
    geom_point(size = 5) +
    scale_color_frontiers() +
    labs(x = "PCA 1", y = "PCA 2", color = "Subtype") + 
    theme_classic() +
    theme(axis.text = element_text(size = 12, color = "black"),
          axis.title = element_text(size = 16, face = "bold"))
