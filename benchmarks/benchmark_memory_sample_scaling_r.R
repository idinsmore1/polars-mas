#!/usr/bin/env Rscript
# Benchmark script for R PheWAS memory scaling by sample size
#
# Tests memory usage with increasing numbers of samples (1000, 5000, 10000, 15000, 20000)
# using the phewas_example_1e+05_samples_50_covariates.csv file.

library(data.table)
library(PheWAS)
library(logistf)
library(pryr)  # For memory profiling

# Data file
data_file <- "phewas_example_1e+05_samples_50_covariates.csv"

# Common parameters
predictor <- "rsEXAMPLE"
cores <- 8
method <- "logistf"

# Standard covariates for sample scaling test
covariates <- c("sex", "age", "race_1", "bmi", "smoking_status")

# Sample sizes to test
sample_sizes <- c(1000, 5000, 10000, 15000, 20000)

# Results storage
results <- data.frame(
    n_samples = integer(),
    time_seconds = numeric(),
    max_rss_mb = numeric(),
    stringsAsFactors = FALSE
)

cat("============================================================\n")
cat("R PheWAS Memory Scaling by Sample Size Benchmark\n")
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

# Load pre-generated indices (should exist from other benchmarks)
sample_indices_file <- "results/sample_indices.csv"
if (!file.exists(sample_indices_file)) {
    cat("Generating sample indices for reproducibility...\n")
    max_samples <- max(sample_sizes)
    indices <- sample(1:n_total_samples, max_samples, replace = FALSE)
    write.csv(data.frame(index = indices), sample_indices_file, row.names = FALSE)
}

# Load pre-generated indices
sample_indices <- fread(sample_indices_file)$index

# Function to get current memory usage in MB
get_memory_mb <- function() {
    # Try to get memory from /proc on Linux
    if (file.exists("/proc/self/status")) {
        status <- readLines("/proc/self/status")
        vmrss_line <- grep("VmRSS:", status, value = TRUE)
        if (length(vmrss_line) > 0) {
            vmrss_kb <- as.numeric(gsub("[^0-9]", "", vmrss_line))
            return(vmrss_kb / 1024)
        }
    }
    # Fallback to pryr
    return(pryr::mem_used() / 1024 / 1024)
}

# Test each sample size
for (n_samples in sample_sizes) {
    cat(sprintf("  Running with %d samples...", n_samples))

    # Use first n_samples from pre-generated indices
    current_indices <- sample_indices[1:n_samples]
    data <- full_data[current_indices, ]

    # Force garbage collection before measuring
    gc(verbose = FALSE)
    mem_before <- get_memory_mb()

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

    mem_after <- get_memory_mb()
    max_rss_mb <- max(mem_before, mem_after)

    elapsed <- as.numeric(difftime(end_time, start_time, units = "secs"))
    cat(sprintf(" Done in %.2fs, Memory: %.1f MB\n", elapsed, max_rss_mb))

    results <- rbind(results, data.frame(
        n_samples = n_samples,
        time_seconds = elapsed,
        max_rss_mb = max_rss_mb,
        stringsAsFactors = FALSE
    ))

    # Clean up
    rm(output, data)
    gc(verbose = FALSE)
}

# Print summary
cat("\n")
cat("============================================================\n")
cat("Benchmark Results Summary\n")
cat("============================================================\n")
cat(sprintf("%12s %12s %14s\n", "Samples", "Time (s)", "Memory (MB)"))
cat(strrep("-", 40), "\n")
for (i in 1:nrow(results)) {
    cat(sprintf("%12d %12.2f %14.1f\n", results$n_samples[i], results$time_seconds[i], results$max_rss_mb[i]))
}

# Save results
output_file <- "results/benchmark_memory_sample_scaling_r_results.csv"
write.csv(results, output_file, row.names = FALSE)
cat(sprintf("\nResults saved to %s\n", output_file))
