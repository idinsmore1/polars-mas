uv run polars-mas \
-i phewas_example_1e+05_samples_20_covariates.csv \
-o example_result_log_fast \
-c i:1-11 \
-p rsEXAMPLE \
-d i:22-222 \
-m firth \
-t 8 \
-n 1 \
--phewas