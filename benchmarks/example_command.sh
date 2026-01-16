uv run polars-mas \
-i phewas_example_10000_samples_20_covariates.csv \
-o example_result_log_fast \
-c i:1-11 \
-p rsEXAMPLE \
-d i:22- \
-m firth \
-t 2 \
-n 8 \
--phewas