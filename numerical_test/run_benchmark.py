from numerical_test.data_generation.generate_data import generate_all_datasets
from numerical_test.benchmark_runner import run_all_datasets
from numerical_test.plotting import make_all_plots


def main(n_unique=500, seed=2025):
    data_root = f"numerical_test/results/seed{seed}/data/data_n{n_unique}"
    results_csv = f"numerical_test/results/seed{seed}/tables/benchmark_results_n{n_unique}.csv"
    pred_dir = f"numerical_test/results/seed{seed}/predictions/predictions_n{n_unique}"
    figure_dir = f"numerical_test/results/seed{seed}/figures/figures_n{n_unique}"

    # 1) Generate datasets
    generate_all_datasets(
        base_dir=data_root,
        seed=seed,
        n_unique=n_unique,
    )

    # 2) Run benchmark
    df = run_all_datasets(
        data_root=data_root,
        out_csv=results_csv,
        pred_dir=pred_dir,
        include_oilmm=False,
        include_puq=True,
    )

    print(df)

    # 3) Plot
    make_all_plots(
        results_csv=results_csv,
        figure_dir=figure_dir,
        data_root=data_root,
        pred_dir=pred_dir,
    )

if __name__ == "__main__":
    main()