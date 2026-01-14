#!/usr/bin/env Rscript
# Benchmark script for R PheWAS covariate scaling
#
# Tests performance with increasing numbers of covariates (1, 3, 5, 10, 15, 20)
# using the phewas_example_5000_samples_20_covariates.csv file.
#
# This benchmark demonstrates how polars-mas handles larger numbers of covariates
# more efficiently than the R PheWAS package.

library(data.table)
library(PheWAS)
library(logistf)

# Data file
data_file <- "phewas_example_5000_samples_20_covariates.csv"

# Common parameters
predictor <- "rsEXAMPLE"
cores <- 8
method <- "logistf"

# Covariate sets for testing (progressively larger)
covariate_sets <- list(
    "1" = c("age"),
    "3" = c("age", "sex", "bmi"),
    "5" = c("age", "age2", "sex", "bmi", "smoking_status"),
    "10" = c("age", "age2", "sex", "race_1", "race_2", "race_3", "bmi", "smoking_status", "alcohol_use", "hba1c"),
    "15" = c("age", "age2", "sex", "race_1", "race_2", "race_3", "bmi", "smoking_status", "alcohol_use", "height", "weight", "hba1c", "cholesterol", "triglycerides", "ldl"),
    "20" = c("age", "age2", "sex", "race_1", "race_2", "race_3", "bmi", "smoking_status", "alcohol_use", "height", "weight", "hba1c", "cholesterol", "triglycerides", "ldl", "hdl", "creatinine", "uric_acid", "glucose", "on_insulin")
)

# Results storage
results <- data.frame(
    n_covariates = integer(),
    time_seconds = numeric(),
    covariates = character(),
    stringsAsFactors = FALSE
)

cat("============================================================\n")
cat("R PheWAS Covariate Scaling Benchmark\n")
cat("============================================================\n")
cat(sprintf("Cores: %d\n", cores))
cat(sprintf("Method: %s\n", method))
cat(sprintf("Data file: %s\n", data_file))
cat("\n")

if (!file.exists(data_file)) {
    cat(sprintf("ERROR: Data file not found: %s\n", data_file))
    quit(status = 1)
}

# Load data once
cat("Loading data...\n")
data <- fread(data_file)
n_samples <- nrow(data)
phecodes <- names(data)[23:length(names(data))]  # Phecodes start at column 23 (1-indexed)
n_phecodes <- length(phecodes)

cat(sprintf("Samples: %d, PheCodes: %d\n\n", n_samples, n_phecodes))

# Ensure results directory exists
if (!dir.exists("results")) {
    dir.create("results")
}

# Test each covariate set
covariate_counts <- c("1", "3", "5", "10", "15", "20")

for (n_covs in covariate_counts) {
    covariates <- covariate_sets[[n_covs]]
    n_covariates <- as.integer(n_covs)

    cat(sprintf("  Running with %d covariate(s)...", n_covariates))

    start_time <- Sys.time()
    output <- phewas_ext(
        data = data,
        phenotypes = phecodes,
        predictors = predictor,
        covariates = covariates,
        cores = cores,
        method = method,
        additive.genotypes = FALSE
    )
    end_time <- Sys.time()

    elapsed <- as.numeric(difftime(end_time, start_time, units = "secs"))
    cat(sprintf(" Done in %.2f seconds\n", elapsed))

    # Save phewas output to CSV
    phewas_output_file <- sprintf("results/benchmark_covariate_scaling_%d_covs_r_phewas_output.csv", n_covariates)
    write.csv(output, phewas_output_file, row.names = FALSE)

    results <- rbind(results, data.frame(
        n_covariates = n_covariates,
        time_seconds = elapsed,
        covariates = paste(covariates, collapse = ","),
        stringsAsFactors = FALSE
    ))
}

# Print summary
cat("\n")
cat("============================================================\n")
cat("Benchmark Results Summary\n")
cat("============================================================\n")
cat(sprintf("%12s %12s\n", "Covariates", "Time (s)"))
cat(strrep("-", 26), "\n")
for (i in 1:nrow(results)) {
    cat(sprintf("%12d %12.2f\n", results$n_covariates[i], results$time_seconds[i]))
}

# Save results
output_file <- "results/benchmark_covariate_scaling_r_results.csv"
write.csv(results, output_file, row.names = FALSE)
cat(sprintf("\nResults saved to %s\n", output_file))
