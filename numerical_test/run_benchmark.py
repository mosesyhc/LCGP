from numerical_test.data_generation.generate_data import generate_all_datasets
from numerical_test.benchmark_runner import run_all_datasets
from numerical_test.plotting import make_all_plots


def main():
    # 1) Generate datasets
    generate_all_datasets(
        base_dir="numerical_test/results/data",
        seed=2026,
    )

    # 2) Run benchmark
    df = run_all_datasets(
        data_root="numerical_test/results/data",
        out_csv="numerical_test/results/tables/benchmark_results.csv",
        pred_dir="numerical_test/results/predictions",
        include_oilmm=True,
        include_puq=True,
    )

    print(df)

    # 3) Plot
    make_all_plots(
        results_csv="numerical_test/results/tables/benchmark_results.csv",
        figure_dir="numerical_test/results/figures",
        data_root="numerical_test/results/data",
        pred_dir="numerical_test/results/predictions",
    )


if __name__ == "__main__":
    main()