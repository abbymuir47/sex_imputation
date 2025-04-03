#this file pulls the homeosapien gene expression data from ensembl, including gene ID, chromosome name, gene type, and gene name 

library(biomaRt)

ensembl <- useEnsembl(biomart = "genes", dataset = "hsapiens_gene_ensembl")

chromosome_names = c("X","Y", "1", "2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23")

filtered_df = getBM(attributes = c("ensembl_gene_id", "chromosome_name", "gene_biotype", "external_gene_name"),
      filters = c("chromosome_name","biotype"),
      values = list(chromosome_names, "protein_coding"),
      mart = ensembl)

write.table(filtered_df, file = "ensembl_data.tsv", sep = "\t", row.names = FALSE, quote = FALSE)
