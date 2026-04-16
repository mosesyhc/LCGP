from pathlib import Path

from numerical_test.data_generation.generate_data import generate_all_datasets
from numerical_test.benchmark_runner import run_all_datasets
from numerical_test.plotting import make_all_plots


def main(n_unique=50, seed=2025):
    data_root = Path(f"numerical_test/results/seed{seed}/data/data_n{n_unique}")
    results_csv = Path(f"numerical_test/results/seed{seed}/tables/benchmark_results_n{n_unique}.csv")
    pred_dir = Path(f"numerical_test/results/seed{seed}/predictions/predictions_n{n_unique}")
    row_dir = Path(f"numerical_test/results/seed{seed}/rows/rows_n{n_unique}")
    figure_dir = Path(f"numerical_test/results/seed{seed}/figures/figures_n{n_unique}")

    generate_all_datasets(
        base_dir=str(data_root),
        seed=seed,
        n_unique=n_unique,
    )

    df = run_all_datasets(
        data_root=str(data_root),
        out_csv=str(results_csv),
        pred_dir=str(pred_dir),
        row_dir=str(row_dir),
        include_oilmm=False,
        include_puq=True,
        seed=seed,
        n_unique=n_unique,
        n_jobs=5,
    )

    print(df)

    make_all_plots(
        results_csv=str(results_csv),
        figure_dir=str(figure_dir),
        data_root=str(data_root),
        pred_dir=str(pred_dir),
    )


if __name__ == "__main__":
    main()