library(PheWAS)
library(data.table)

set.seed(100)
sample_sizes = c(5000, 10000, 50000, 100000, 250000, 500000)
for (sample_size in sample_sizes) {
  ex = generateExample(n=sample_size, phenotypes.per = 20, hit="250.2")
  covars <- ex$id.sex %>% 
    mutate(
      sex = case_when(sex == "M" ~ 0, sex =="F" ~ 1),
      age = round(runif(sample_size, min=20, max=90)),
      age2 = age ^ 2,
      race = round(sample(0:3, sample_size, replace=TRUE))
    ) %>%
    inner_join(ex$genotypes, by='id')
  phenotypes = createPhenotypes(ex$id.vocab.code.count, aggregate.fun = sum, id.sex=ex$id.sex)
  phenotypes[, -1] <- lapply(phenotypes[, -1], as.integer)
  test_df = covars %>% inner_join(phenotypes, by='id')
  output_file = sprintf('phewas_example_%s_samples.csv', sample_size)
  fwrite(test_df, output_file)
}