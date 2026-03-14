from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from numerical_test.data_generation.read_data import load_dataset


MODEL_STYLE = {
    "LCGP-full": {"ls": "--", "lw": 1.6},
    "LCGP-rep": {"ls": "-", "lw": 2.0},
    "OILMM": {"ls": "-.", "lw": 1.6},
    "PUQ-multigetgp": {"ls": ":", "lw": 1.8},
    "Truth": {"ls": "-", "lw": 1.8},
}


def _style_for_model(model_name):
    return MODEL_STYLE.get(model_name, {"ls": "-", "lw": 1.5})


def plot_metric_bar(df, dataset_name, metric, out_path):
    subset = df[(df["dataset"] == dataset_name) & (df["status"] == "ok")].copy()
    subset = subset.sort_values("model")

    plt.figure(figsize=(8, 5))
    plt.bar(subset["model"], subset[metric])
    plt.ylabel(metric)
    plt.title(f"{metric} comparison on {dataset_name}")
    plt.xticks(rotation=20)
    plt.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_rmse_by_output(df, dataset_name, out_path):
    subset = df[(df["dataset"] == dataset_name) & (df["status"] == "ok")].copy()
    subset = subset.sort_values("model")

    models = subset["model"].tolist()
    y1 = subset["rmse_y1"].to_numpy()
    y2 = subset["rmse_y2"].to_numpy()
    y3 = subset["rmse_y3"].to_numpy()

    x = np.arange(len(models))
    w = 0.24

    plt.figure(figsize=(9, 5))
    plt.bar(x - w, y1, width=w, label="Output 1")
    plt.bar(x,     y2, width=w, label="Output 2")
    plt.bar(x + w, y3, width=w, label="Output 3")

    plt.xticks(x, models, rotation=20)
    plt.ylabel("RMSE")
    plt.title(f"RMSE by output on {dataset_name}")
    plt.legend()
    plt.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_prediction_comparison(dataset_name,
                               data_root,
                               benchmark_csv,
                               pred_dir,
                               out_path,
                               band_kind=None):
    """
    band_kind:
        None      -> no uncertainty band
        "conf"    -> use conf variance if available
        "pred"    -> use predictive variance if available
    """
    benchmark_df = pd.read_csv(benchmark_csv)
    bench_sub = benchmark_df[
        (benchmark_df["dataset"] == dataset_name) &
        (benchmark_df["status"] == "ok")
    ].copy()

    # load original train/test data
    data = load_dataset(Path(data_root) / dataset_name)
    xtrain = np.asarray(data["xtrain"], dtype=np.float64)
    ytrain = np.asarray(data["ytrain"], dtype=np.float64)
    xtest = np.asarray(data["xtest"], dtype=np.float64)
    ytrue = np.asarray(data["ytrue"], dtype=np.float64)

    order_test = np.argsort(xtest[:, 0])
    order_train = np.argsort(xtrain[:, 0])

    x_tr = xtrain[order_train, 0]
    x_te = xtest[order_test, 0]

    fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex="col")

    for out_idx in range(3):
        ax_left = axes[out_idx, 0]
        ax_right = axes[out_idx, 1]

        # training replicates
        ax_left.scatter(
            x_tr,
            ytrain[out_idx, order_train],
            s=10,
            alpha=0.45,
            label="Replicates" if out_idx == 0 else None,
        )

        # truth
        truth_style = _style_for_model("Truth")
        ax_left.plot(
            x_te,
            ytrue[out_idx, order_test],
            truth_style["ls"],
            lw=truth_style["lw"],
            label="Truth" if out_idx == 0 else None,
        )

        ax_right.axhline(0.0, lw=1.0)

        # each model
        for _, row in bench_sub.iterrows():
            model_name = row["model"]
            pred_csv = row["prediction_csv"]
            if pd.isna(pred_csv):
                continue

            pred_df = pd.read_csv(pred_csv)
            st = _style_for_model(model_name)

            ymean = pred_df[f"y{out_idx+1}_mean"].to_numpy()[order_test]
            yerr = pred_df[f"y{out_idx+1}_err"].to_numpy()[order_test]

            ax_left.plot(
                x_te,
                ymean,
                st["ls"],
                lw=st["lw"],
                label=model_name if out_idx == 0 else None,
            )

            if band_kind in ("conf", "pred"):
                var_col = f"y{out_idx+1}_{'confvar' if band_kind == 'conf' else 'predvar'}"
                if var_col in pred_df.columns:
                    band_var = pred_df[var_col].to_numpy()[order_test]
                    s = np.sqrt(np.maximum(band_var, 0.0))
                    lo = ymean - 1.96 * s
                    hi = ymean + 1.96 * s
                    ax_left.fill_between(x_te, lo, hi, alpha=0.10)

            ax_right.plot(
                x_te,
                yerr,
                st["ls"],
                lw=st["lw"],
                label=model_name if out_idx == 0 else None,
            )

        ax_left.set_ylabel(f"Output {out_idx+1}")
        ax_right.set_ylabel("Mean error")

    axes[-1, 0].set_xlabel("x")
    axes[-1, 1].set_xlabel("x")
    axes[0, 0].legend(loc="best", fontsize=9)
    axes[0, 0].set_title(f"{dataset_name}: predictions")
    axes[0, 1].set_title(f"{dataset_name}: prediction error")

    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def make_all_plots(results_csv, figure_dir, data_root="numerical_test/results/data",
                   pred_dir="numerical_test/results/predictions"):
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(results_csv)

    for dataset_name in sorted(df["dataset"].unique()):
        plot_metric_bar(
            df=df,
            dataset_name=dataset_name,
            metric="rmse",
            out_path=figure_dir / f"{dataset_name}_rmse.png",
        )
        plot_rmse_by_output(
            df=df,
            dataset_name=dataset_name,
            out_path=figure_dir / f"{dataset_name}_rmse_by_output.png",
        )
        plot_metric_bar(
            df=df,
            dataset_name=dataset_name,
            metric="fit_time_sec",
            out_path=figure_dir / f"{dataset_name}_fit_time.png",
        )
        plot_metric_bar(
            df=df,
            dataset_name=dataset_name,
            metric="pred_time_sec",
            out_path=figure_dir / f"{dataset_name}_pred_time.png",
        )
        plot_metric_bar(
            df=df,
            dataset_name=dataset_name,
            metric="total_time_sec",
            out_path=figure_dir / f"{dataset_name}_total_time.png",
        )

        plot_prediction_comparison(
            dataset_name=dataset_name,
            data_root=data_root,
            benchmark_csv=results_csv,
            pred_dir=pred_dir,
            out_path=figure_dir / f"{dataset_name}_prediction_compare.png",
            # band_kind=None,   
            band_kind="conf",   
        )