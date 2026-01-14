library(data.table)
library(PheWAS)
library(logistf)
data = fread('example_data/phewas_T2d_example.csv')
predictor = "rsEXAMPLE"
covariates = c("age", "sex", "race")
phecodes = names(data)[c(8:length(names(data)))]
fit <- logistf(`250.2`~rsEXAMPLE+age+sex+race, data=data)
print(summary(fit))
