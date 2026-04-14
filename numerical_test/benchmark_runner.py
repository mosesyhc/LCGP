from pathlib import Path
import traceback
import pandas as pd
import numpy as np

from numerical_test.data_generation.read_data import load_dataset
from numerical_test.call_model.lcgp_run import LCGPRun
from numerical_test.call_model.oilmm_run import OILMMRun
from numerical_test.call_model.puq_run import PUQRun


def build_model_runners(runno, data, include_oilmm=True, include_puq=True):
    runners = [
        LCGPRun(
            runno=runno,
            data=data,
            submethod="full",
            num_latent=None,
            var_threshold=None,
            diag_error_structure=None,
            robust_mean=True,
            parameter_clamp_flag=False,
            rep_standardize_ybar=True,
            verbose=False,
        ),
        LCGPRun(
            runno=runno,
            data=data,
            submethod="rep",
            num_latent=None,
            var_threshold=None,
            diag_error_structure=None,
            robust_mean=True,
            parameter_clamp_flag=False,
            rep_standardize_ybar=True,
            verbose=False,
        ),
    ]

    if include_oilmm:
        runners.append(OILMMRun(runno=runno, data=data, num_latent=3, verbose=False))

    if include_puq:
        runners.append(PUQRun(runno=runno, data=data, verbose=False))

    return runners


def parse_dataset_metadata(dataset_name):
    noise_type, rep_type = dataset_name.split("__")

    return {
        "noise_type": noise_type,
        "replication_type": rep_type,
        "equal": int(rep_type == "equal_rep"),
        "no_rep": int(rep_type == "no_rep"),
        "unequal_rep": int(rep_type == "unequal_rep"),
    }


def add_method_suffix(model_name, runner):
    """
    Make method names unique in the CSV.
    """
    if isinstance(runner, LCGPRun):
        return f"{model_name}_{runner.submethod}"
    return model_name


def _save_prediction_csv(dataset_name, model_name, data, pred, pred_dir):
    pred_dir = Path(pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)

    ymean = np.asarray(pred[0], dtype=np.float64)      # (p, n0)
    ypredvar = np.asarray(pred[1], dtype=np.float64)   # (p, n0)
    yconfvar = np.asarray(pred[2], dtype=np.float64)   # (p, n0)

    xtest = np.asarray(data["xtest"], dtype=np.float64)
    ytrue = np.asarray(data["ytrue"], dtype=np.float64)

    rows = {
        "x": xtest[:, 0],
    }

    p = ymean.shape[0]
    for i in range(p):
        rows[f"y{i+1}_true"] = ytrue[i]
        rows[f"y{i+1}_mean"] = ymean[i]
        rows[f"y{i+1}_predvar"] = ypredvar[i]
        rows[f"y{i+1}_confvar"] = yconfvar[i]
        rows[f"y{i+1}_err"] = ymean[i] - ytrue[i]

    df = pd.DataFrame(rows)

    safe_model_name = model_name.replace("/", "_").replace(" ", "_")
    out_path = pred_dir / f"{dataset_name}__{safe_model_name}.csv"
    df.to_csv(out_path, index=False)
    return out_path


def run_single_dataset(dataset_name,
                       dataset_dir,
                       runno=0,
                       seed=None,
                       n_unique=None,
                       include_oilmm=True,
                       include_puq=True,
                       pred_dir="numerical_test/results/predictions"):
    data = load_dataset(dataset_dir)
    runners = build_model_runners(runno, data, include_oilmm=include_oilmm, include_puq=include_puq)

    dataset_meta = parse_dataset_metadata(dataset_name)
    rows = []

    for runner in runners:
        try:
            out = runner.run_all(return_fullcov=False)
            row = out["metrics"]

            row["model"] = add_method_suffix(row["model"], runner)
            row["dataset"] = dataset_name
            row["seed"] = seed
            row["n"] = n_unique
            row["status"] = "ok"
            row["error"] = None
            row["traceback"] = None

            row.update(dataset_meta)

            pred_path = _save_prediction_csv(
                dataset_name=dataset_name,
                model_name=row["model"],
                data=data,
                pred=out["prediction"],
                pred_dir=pred_dir,
            )
            row["prediction_csv"] = str(pred_path)

            rows.append(row)
            print(f"[OK] {dataset_name} - {row['model']}")

        except Exception as e:
            err_row = {
                "dataset": dataset_name,
                "model": add_method_suffix(runner.model_name, runner),
                "fit_time_sec": None,
                "pred_time_sec": None,
                "total_time_sec": None,
                "rmse": None,
                "rmse_y1": None,
                "rmse_y2": None,
                "rmse_y3": None,
                "seed": seed,
                "n": n_unique,
                "status": "failed",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "prediction_csv": None,
                **dataset_meta,
            }
            rows.append(err_row)
            print(f"[FAIL] {dataset_name} - {err_row['model']}: {e}")

    return pd.DataFrame(rows)


def run_all_datasets(data_root="numerical_test/results/data",
                     out_csv="numerical_test/results/tables/benchmark_results.csv",
                     pred_dir="numerical_test/results/predictions",
                     include_oilmm=True,
                     include_puq=True,
                     seed=None,
                     n_unique=None):
    data_root = Path(data_root)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    dataset_dirs = {
        "same_noise__no_rep": data_root / "same_noise__no_rep",
        "same_noise__equal_rep": data_root / "same_noise__equal_rep",
        "same_noise__unequal_rep": data_root / "same_noise__unequal_rep",
        "different_noise__no_rep": data_root / "different_noise__no_rep",
        "different_noise__equal_rep": data_root / "different_noise__equal_rep",
        "different_noise__unequal_rep": data_root / "different_noise__unequal_rep",
    }

    all_frames = []
    for dataset_name, dataset_dir in dataset_dirs.items():
        df = run_single_dataset(
            dataset_name=dataset_name,
            dataset_dir=dataset_dir,
            runno=0,  # kept only because the model constructors still expect it
            seed=seed,
            n_unique=n_unique,
            include_oilmm=include_oilmm,
            include_puq=include_puq,
            pred_dir=pred_dir,
        )
        all_frames.append(df)

    final_df = pd.concat(all_frames, ignore_index=True)
    final_df.to_csv(out_csv, index=False)
    print(f"Saved benchmark results to: {out_csv}")
    return final_df