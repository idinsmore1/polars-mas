#!/usr/bin/env Rscript
# Benchmark script for R PheWAS covariate scaling (up to 50 covariates)
#
# Tests performance with increasing numbers of covariates (1, 3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50)
# using a 5000 sample subset from phewas_example_1e+05_samples_50_covariates.csv file.

library(data.table)
library(PheWAS)
library(logistf)

# Data file
data_file <- "phewas_example_1e+05_samples_50_covariates.csv"

# Common parameters
predictor <- "rsEXAMPLE"
cores <- 8
method <- "logistf"
n_samples <- 5000

# All available covariates in order (columns 2-51 in file, 50 total)
all_covariates <- c(
    "sex", "age", "age2", "race_1", "race_2", "race_3", "bmi", "smoking_status",
    "alcohol_use", "height", "weight", "hba1c", "cholesterol", "triglycerides",
    "ldl", "hdl", "creatinine", "uric_acid", "glucose", "on_insulin",
    "systolic_bp", "diastolic_bp", "heart_rate", "temperature", "respiratory_rate",
    "oxygen_saturation", "hemoglobin", "hematocrit", "platelet_count", "white_blood_cell_count",
    "albumin", "alt", "ast", "alkaline_phosphatase", "bilirubin",
    "sodium", "potassium", "chloride", "calcium", "magnesium",
    "phosphorus", "vitamin_d", "vitamin_b12", "folate", "iron",
    "ferritin", "tsh", "t4", "inr", "physical_activity"
)

# Covariate sets for testing (progressively larger)
covariate_sets <- list(
    "1" = all_covariates[1],
    "3" = all_covariates[1:3],
    "5" = all_covariates[1:5],
    "10" = all_covariates[1:10],
    "15" = all_covariates[1:15],
    "20" = all_covariates[1:20],
    "25" = all_covariates[1:25],
    "30" = all_covariates[1:30],
    "35" = all_covariates[1:35],
    "40" = all_covariates[1:40],
    "45" = all_covariates[1:45],
    "50" = all_covariates[1:50]
)

# Results storage
results <- data.frame(
    n_covariates = integer(),
    time_seconds = numeric(),
    covariates = character(),
    stringsAsFactors = FALSE
)

cat("============================================================\n")
cat("R PheWAS Covariate Scaling Benchmark (up to 50 covariates)\n")
cat("============================================================\n")
cat(sprintf("Cores: %d\n", cores))
cat(sprintf("Method: %s\n", method))
cat(sprintf("Data file: %s\n", data_file))
cat(sprintf("Sample size: %d\n", n_samples))
cat("\n")

if (!file.exists(data_file)) {
    cat(sprintf("ERROR: Data file not found: %s\n", data_file))
    quit(status = 1)
}

# Load full data
cat("Loading data...\n")
full_data <- fread(data_file)
n_total_samples <- nrow(full_data)
phecodes <- names(full_data)[53:length(names(full_data))]  # Phecodes start at column 53 (1-indexed)
n_phecodes <- length(phecodes)

cat(sprintf("Total samples available: %d, PheCodes: %d\n", n_total_samples, n_phecodes))

# Ensure results directory exists
if (!dir.exists("results")) {
    dir.create("results")
}

# Load pre-generated indices (should exist from sample scaling benchmark)
sample_indices_file <- "results/sample_indices.csv"
if (!file.exists(sample_indices_file)) {
    cat("Generating sample indices for reproducibility...\n")
    set.seed(42)
    max_samples <- 50000
    indices <- sample(1:n_total_samples, max_samples, replace = FALSE)
    write.csv(data.frame(index = indices), sample_indices_file, row.names = FALSE)
}

sample_indices <- fread(sample_indices_file)$index

# Get 5000 sample subset
current_indices <- sample_indices[1:n_samples]
data <- full_data[current_indices, ]
cat(sprintf("Using %d samples for covariate scaling test\n\n", nrow(data)))

# Test each covariate set
covariate_counts <- c("1", "3", "5", "10", "15", "20", "25", "30", "35", "40", "45", "50")

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
    phewas_output_file <- sprintf("results/benchmark_covariate_scaling_50_%d_covs_r_phewas_output.csv", n_covariates)
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
output_file <- "results/benchmark_covariate_scaling_50_r_results.csv"
write.csv(results, output_file, row.names = FALSE)
cat(sprintf("\nResults saved to %s\n", output_file))
