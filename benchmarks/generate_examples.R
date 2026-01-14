library(PheWAS)
library(data.table)

set.seed(100)
sample_sizes = c(5000, 10000)
for (sample_size in sample_sizes) {
  ex = generateExample(n=sample_size, phenotypes.per = 20, hit="250.2")
  covars <- ex$id.sex %>% 
    mutate(
      sex = case_when(sex == "M" ~ 0, sex =="F" ~ 1),
      age = round(runif(sample_size, min=20, max=90)),
      age2 = age ^ 2,
      race_1 = round(sample(0:1, sample_size, replace=TRUE)),
      race_2 = round(sample(0:1, sample_size, replace=TRUE)),
      race_3 = round(sample(0:1, sample_size, replace=TRUE)),
      bmi = round(runif(sample_size, min=15, max=50)),
      smoking_status = round(sample(0:1, sample_size, replace=TRUE)),
      alcohol_use = round(sample(0:1, sample_size, replace=TRUE)),
      height = round(runif(sample_size, min=150, max=200)),
      weight = round(runif(sample_size, min=50, max=150)),
      hba1c = round(runif(sample_size, min=4, max=10), 1),
      cholesterol = round(runif(sample_size, min=150, max=300)),
      triglycerides = round(runif(sample_size, min=50, max=200)),
      ldl = round(runif(sample_size, min=50, max=200)),
      hdl = round(runif(sample_size, min=30, max=100)),
      creatinine = round(runif(sample_size, min=0.5, max=1.5), 2),
      uric_acid = round(runif(sample_size, min=3, max=10), 1),
      glucose = round(runif(sample_size, min=70, max=200)),
      on_insulin = round(sample(0:1, sample_size, replace=TRUE))
    ) %>%
    inner_join(ex$genotypes, by='id')
  phenotypes = createPhenotypes(ex$id.vocab.code.count, aggregate.fun = sum, id.sex=ex$id.sex)
  phenotypes[, -1] <- lapply(phenotypes[, -1], as.integer)
  test_df = covars %>% inner_join(phenotypes, by='id')
  output_file = sprintf('phewas_example_%s_samples_20_covariates.csv', sample_size)
  fwrite(test_df, output_file)
}