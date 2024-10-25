import time
import polars as pl
import polars_mas.mas_frame as pla
import numpy as np
from loguru import logger
from firthlogist import FirthLogisticRegression
from polars_mas.consts import sex_specific_codes

NUM_COMPLETED = 0
TIME_PER_ASSOC = 0
TIME_PER_BLOCK = 0
PREV_TIME = None


def _update_progress(num_groups: int) -> None:
    global NUM_COMPLETED
    global TIME_PER_ASSOC
    global TIME_PER_BLOCK
    global PREV_TIME
    block = 50

    NUM_COMPLETED += 1
    if PREV_TIME is None:
        PREV_TIME = time.perf_counter()
    now = time.perf_counter()
    elapsed_time = now - PREV_TIME
    # print(elapsed_time)
    TIME_PER_ASSOC += elapsed_time
    TIME_PER_BLOCK += elapsed_time
    PREV_TIME = now
    if NUM_COMPLETED % block == 0 or NUM_COMPLETED == num_groups:
        cpu_time_per_block = TIME_PER_BLOCK
        TIME_PER_BLOCK = 0
        logger.log("PROGRESS", f"Completed: [{NUM_COMPLETED}/{num_groups}] - {cpu_time_per_block:.2f}s")
    # if NUM_COMPLETED == num_groups:
    #     logger.success(f"Completed: [{NUM_COMPLETED}/{num_groups}] - {TIME_PER_ASSOC:.3f}s")


def run_firth_regression(struct: pl.Struct, num_groups: int):
    output_struct = {
        "predictor": "nan",
        "dependent": "nan",
        "pval": float("nan"),
        "beta": float("nan"),
        "se": float("nan"),
        "OR": float("nan"),
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "cases": float("nan"),
        "controls": float("nan"),
        "total_n": float("nan"),
        "failed_reason": "nan",
    }
    start = time.perf_counter()
    df = struct.struct.unnest()
    col_names = df.collect_schema().names()
    predictor = col_names[0]
    dependent = col_names[-1]
    output_struct.update({'predictor': predictor, 'dependent': dependent})
    df = df.drop_nulls([predictor, dependent])
    covariates = [col for col in col_names if col not in [predictor, dependent]]
    X = df.select([predictor, *covariates])
    y = df.get_column(dependent).to_numpy()
    non_consts = X.polars_mas.check_grouped_independents_for_constants([predictor, *covariates], dependent)
    X = X.select(non_consts)
    if predictor not in X.collect_schema().names():
        logger.warning(f"Predictor {predictor} was removed due to constant values. Skipping analysis.")
        output_struct.update(
            {
                "failed_reason": "Predictor removed due to constant values",
            }
        )
        _update_progress(num_groups)
        return output_struct
    cases = y.sum().astype(int)
    total_counts = y.shape[0]
    controls = total_counts - cases
    output_struct.update(
        {
            "cases": cases,
            "controls": controls,
            "total_n": total_counts,
        }
    )
    try:
        # We are only interested in the first predictor for the association test
        fl = FirthLogisticRegression(max_iter=1000, test_vars=0)
        fl.fit(X, y)
        # input_vars = X.collect_schema().names()
        output_struct.update(
            {
                "pval": fl.pvals_[0],
                "beta": fl.coef_[0],
                "se": fl.bse_[0],
                "OR": np.e ** fl.coef_[0],
                "ci_low": fl.ci_[0][0],
                "ci_high": fl.ci_[0][1],
                # "input_vars": ",".join(input_vars),
            }
        )
        end = time.perf_counter()
        elapsed = end - start
        _update_progress(num_groups)
        return output_struct
    except Exception as e:
        end = time.perf_counter()
        elapsed = end - start
        logger.error(f"Error in Firth regression for {dependent}: {e}")
        output_struct.update({"failed_reason": str(e)})
        _update_progress(num_groups)
        return output_struct
