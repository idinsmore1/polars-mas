uv run polars-mas \
-i phewas_example_10000_samples_20_covariates.csv \
-o example_result_log_fast \
-c i:1-4 \
-p rsEXAMPLE \
-d i:22- \
-m firth \
-t 8 \
-n 2 \
--phewas