uv run polars-mas \
-i phewas_example_1e+05_samples_50_covariates.csv \
-o example_result_log_fast \
-c i:2-22 \
-p rsEXAMPLE \
-d i:58- \
-m logistic \
-t 2 \
-n 12 \
--phewas