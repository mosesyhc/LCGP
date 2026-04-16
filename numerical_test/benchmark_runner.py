from pathlib import Path
import traceback
import pandas as pd
import numpy as np

from numerical_test.data_generation.read_data import load_dataset
from numerical_test.call_model.lcgp_run import LCGPRun
from numerical_test.call_model.oilmm_run import OILMMRun
from numerical_test.call_model.puq_run import PUQRun

from joblib import Parallel, delayed


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
    if isinstance(runner, LCGPRun):
        return f"{model_name}_{runner.submethod}"
    return model_name


def sanitize_name(name: str) -> str:
    return str(name).replace("/", "_").replace(" ", "_")


def make_run_stem(seed, n_unique, dataset_name, model_name):
    safe_dataset = sanitize_name(dataset_name)
    safe_model = sanitize_name(model_name)
    return f"seed{seed}__n{n_unique}__{safe_dataset}__{safe_model}"


def _save_prediction_csv(dataset_name, model_name, data, pred, pred_dir, seed, n_unique):
    pred_dir = Path(pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)

    ymean = np.asarray(pred[0], dtype=np.float64)
    ypredvar = np.asarray(pred[1], dtype=np.float64)
    yconfvar = np.asarray(pred[2], dtype=np.float64)

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

    run_stem = make_run_stem(seed, n_unique, dataset_name, model_name)
    out_path = pred_dir / f"{run_stem}__predictions.csv"
    df.to_csv(out_path, index=False)
    return out_path


def _save_single_row_csv(row: dict, row_dir: Path, seed, n_unique, dataset_name, model_name):
    row_dir.mkdir(parents=True, exist_ok=True)
    run_stem = make_run_stem(seed, n_unique, dataset_name, model_name)
    out_path = row_dir / f"{run_stem}__metrics.csv"
    pd.DataFrame([row]).to_csv(out_path, index=False)
    return out_path


def _run_single_model(
    runner,
    data,
    dataset_name,
    dataset_meta,
    pred_dir,
    row_dir,
    seed,
    n_unique,
):
    model_name = add_method_suffix(runner.model_name, runner)

    try:
        out = runner.run_all(return_fullcov=False)
        row = out["metrics"]

        row["model"] = model_name
        row["dataset"] = dataset_name
        row["seed"] = seed
        row["n"] = n_unique
        row["status"] = "ok"
        row["error"] = None
        row["traceback"] = None
        row.update(dataset_meta)

        pred_path = _save_prediction_csv(
            dataset_name=dataset_name,
            model_name=model_name,
            data=data,
            pred=out["prediction"],
            pred_dir=pred_dir,
            seed=seed,
            n_unique=n_unique,
        )
        row["prediction_csv"] = str(pred_path)

        metrics_path = _save_single_row_csv(
            row=row,
            row_dir=Path(row_dir),
            seed=seed,
            n_unique=n_unique,
            dataset_name=dataset_name,
            model_name=model_name,
        )
        row["metrics_csv"] = str(metrics_path)

        print(f"[OK] {dataset_name} - {model_name}")
        return row

    except Exception as e:
        err_row = {
            "dataset": dataset_name,
            "model": model_name,
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

        metrics_path = _save_single_row_csv(
            row=err_row,
            row_dir=Path(row_dir),
            seed=seed,
            n_unique=n_unique,
            dataset_name=dataset_name,
            model_name=model_name,
        )
        err_row["metrics_csv"] = str(metrics_path)

        print(f"[FAIL] {dataset_name} - {model_name}: {e}")
        return err_row


def run_single_dataset(
    dataset_name,
    dataset_dir,
    runno=0,
    seed=None,
    n_unique=None,
    include_oilmm=True,
    include_puq=True,
    pred_dir="numerical_test/results/predictions",
    row_dir="numerical_test/results/rows",
    n_jobs=1,
):
    data = load_dataset(dataset_dir)
    runners = build_model_runners(
        runno,
        data,
        include_oilmm=include_oilmm,
        include_puq=include_puq,
    )

    dataset_meta = parse_dataset_metadata(dataset_name)

    rows = Parallel(n_jobs=n_jobs)(
        delayed(_run_single_model)(
            runner=runner,
            data=data,
            dataset_name=dataset_name,
            dataset_meta=dataset_meta,
            pred_dir=pred_dir,
            row_dir=row_dir,
            seed=seed,
            n_unique=n_unique,
        )
        for runner in runners
    )

    return pd.DataFrame(rows)


def combine_row_csvs(row_dir, out_csv):
    row_dir = Path(row_dir)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    row_files = sorted(row_dir.glob("*.csv"))
    if not row_files:
        raise FileNotFoundError(f"No row CSV files found in {row_dir}")

    dfs = [pd.read_csv(f) for f in row_files]
    final_df = pd.concat(dfs, ignore_index=True)
    final_df.to_csv(out_csv, index=False)
    return final_df


def run_all_datasets(
    data_root="numerical_test/results/data",
    out_csv="numerical_test/results/tables/benchmark_results.csv",
    pred_dir="numerical_test/results/predictions",
    row_dir="numerical_test/results/rows",
    include_oilmm=True,
    include_puq=True,
    seed=None,
    n_unique=None,
    n_jobs=1,
):
    data_root = Path(data_root)
    out_csv = Path(out_csv)

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
            runno=0,
            seed=seed,
            n_unique=n_unique,
            include_oilmm=include_oilmm,
            include_puq=include_puq,
            pred_dir=pred_dir,
            row_dir=row_dir,
            n_jobs=n_jobs,
        )
        all_frames.append(df)

    final_df = pd.concat(all_frames, ignore_index=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(out_csv, index=False)
    print(f"Saved benchmark results to: {out_csv}")
    return final_df