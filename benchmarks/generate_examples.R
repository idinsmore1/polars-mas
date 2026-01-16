library(PheWAS)
library(data.table)

set.seed(100)
sample_sizes = c(100000)
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
      on_insulin = round(sample(0:1, sample_size, replace=TRUE)),
      systolic_bp = round(runif(sample_size, min=90, max=180)),
      diastolic_bp = round(runif(sample_size, min=60, max=120)),
      heart_rate = round(runif(sample_size, min=50, max=120)),
      temperature = round(runif(sample_size, min=36, max=38), 1),
      respiratory_rate = round(runif(sample_size, min=12, max=25)),
      oxygen_saturation = round(runif(sample_size, min=90, max=100)),
      hemoglobin = round(runif(sample_size, min=10, max=18), 1),
      hematocrit = round(runif(sample_size, min=30, max=55)),
      platelet_count = round(runif(sample_size, min=150, max=400)),
      white_blood_cell_count = round(runif(sample_size, min=4, max=11), 1),
      albumin = round(runif(sample_size, min=3, max=5), 1),
      alt = round(runif(sample_size, min=10, max=100)),
      ast = round(runif(sample_size, min=10, max=100)),
      alkaline_phosphatase = round(runif(sample_size, min=30, max=150)),
      bilirubin = round(runif(sample_size, min=0.2, max=2), 1),
      sodium = round(runif(sample_size, min=135, max=145)),
      potassium = round(runif(sample_size, min=3.5, max=5.5), 1),
      chloride = round(runif(sample_size, min=95, max=110)),
      calcium = round(runif(sample_size, min=8.5, max=10.5), 1),
      magnesium = round(runif(sample_size, min=1.5, max=2.5), 1),
      phosphorus = round(runif(sample_size, min=2.5, max=4.5), 1),
      vitamin_d = round(runif(sample_size, min=10, max=100)),
      vitamin_b12 = round(runif(sample_size, min=200, max=900)),
      folate = round(runif(sample_size, min=2, max=20)),
      iron = round(runif(sample_size, min=50, max=180)),
      ferritin = round(runif(sample_size, min=20, max=300)),
      tsh = round(runif(sample_size, min=0.5, max=5), 2),
      t4 = round(runif(sample_size, min=5, max=12), 1),
      inr = round(runif(sample_size, min=0.8, max=1.5), 2),
      physical_activity = round(sample(0:1, sample_size, replace=TRUE))
    ) %>%
    inner_join(ex$genotypes, by='id')
  phenotypes = createPhenotypes(ex$id.vocab.code.count, aggregate.fun = sum, id.sex=ex$id.sex)
  phenotypes[, -1] <- lapply(phenotypes[, -1], as.integer)
  test_df = covars %>% inner_join(phenotypes, by='id')
  output_file = sprintf('phewas_example_%s_samples_50_covariates.csv', sample_size)
  fwrite(test_df, output_file)
}