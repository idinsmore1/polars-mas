#!/usr/bin/env Rscript
# Benchmark script for R PheWAS sample scaling
#
# Tests performance with increasing numbers of samples (1000, 5000, 10000, 20000, 30000, 40000, 50000)
# using the phewas_example_1e+05_samples_50_covariates.csv file.

library(data.table)
library(PheWAS)
library(logistf)

# Data file
data_file <- "phewas_example_1e+05_samples_50_covariates.csv"

# Common parameters
predictor <- "rsEXAMPLE"
cores <- 8
method <- "logistf"

# Standard covariates for sample scaling test
covariates <- c("sex", "age", "race_1", "bmi", "smoking_status")

# Sample sizes to test
sample_sizes <- c(1000, 5000, 10000, 20000, 30000, 40000, 50000)

# Results storage
results <- data.frame(
    n_samples = integer(),
    time_seconds = numeric(),
    stringsAsFactors = FALSE
)

cat("============================================================\n")
cat("R PheWAS Sample Scaling Benchmark\n")
cat("============================================================\n")
cat(sprintf("Cores: %d\n", cores))
cat(sprintf("Method: %s\n", method))
cat(sprintf("Data file: %s\n", data_file))
cat(sprintf("Covariates: %s\n", paste(covariates, collapse = ", ")))
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

cat(sprintf("Total samples available: %d, PheCodes: %d\n\n", n_total_samples, n_phecodes))

# Ensure results directory exists
if (!dir.exists("results")) {
    dir.create("results")
}

# Set seed for reproducible sampling
set.seed(42)

# Pre-generate sample indices for each sample size (for consistency with Python)
sample_indices_file <- "results/sample_indices.csv"
if (!file.exists(sample_indices_file)) {
    cat("Generating sample indices for reproducibility...\n")
    max_samples <- max(sample_sizes)
    indices <- sample(1:n_total_samples, max_samples, replace = FALSE)
    write.csv(data.frame(index = indices), sample_indices_file, row.names = FALSE)
}

# Load pre-generated indices
sample_indices <- fread(sample_indices_file)$index

# Test each sample size
for (n_samples in sample_sizes) {
    cat(sprintf("  Running with %d samples...", n_samples))

    # Use first n_samples from pre-generated indices
    current_indices <- sample_indices[1:n_samples]
    data <- full_data[current_indices, ]

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
    phewas_output_file <- sprintf("results/benchmark_sample_scaling_%d_samples_r_phewas_output.csv", n_samples)
    write.csv(output, phewas_output_file, row.names = FALSE)

    results <- rbind(results, data.frame(
        n_samples = n_samples,
        time_seconds = elapsed,
        stringsAsFactors = FALSE
    ))
}

# Print summary
cat("\n")
cat("============================================================\n")
cat("Benchmark Results Summary\n")
cat("============================================================\n")
cat(sprintf("%12s %12s\n", "Samples", "Time (s)"))
cat(strrep("-", 26), "\n")
for (i in 1:nrow(results)) {
    cat(sprintf("%12d %12.2f\n", results$n_samples[i], results$time_seconds[i]))
}

# Save results
output_file <- "results/benchmark_sample_scaling_r_results.csv"
write.csv(results, output_file, row.names = FALSE)
cat(sprintf("\nResults saved to %s\n", output_file))
