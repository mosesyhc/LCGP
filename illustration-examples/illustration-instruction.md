# LCGP 3D Illustration with Replicated Data - Usage Instructions

## Basic Usage
To run the code:
```
python ./illustration-examples/lcgp-rep-3d-illustration.py
```

## Dataset Selection
The code provides three different replication patterns. Comment/uncomment the desired dataset:

### CASE 1: Uniform-ish replication
Data points have relatively uniform replication counts across the input space
```python
results_fig_path = './results_figure_rep_1d_uniform/'
Path(results_fig_path).mkdir(parents=True, exist_ok=True)
xtrain, ytrain, xtest, ytrue = make_rep_data(
    n_unique=16,
    rep_choices=(1,2,3,4,5),
    noise_std=(0.05, 0.08, 0.10),
    seed=2025
)
```

### CASE 2: Skewed replication
Data has heavy replication in specific regions and light replication elsewhere
```python
results_fig_path = './results_figure_rep_1d_skewed/'
Path(results_fig_path).mkdir(parents=True, exist_ok=True)
xtrain, ytrain, xtest, ytrue = make_rep_data_skewed(
    n_unique=40,
    heavy_region=(0.20, 0.45),
    light_rep_choices=(1, 2),
    heavy_rep_choices=(8, 12, 16, 20),
    noise_std=(0.05, 0.08, 0.10),
    seed=123
)
```

### CASE 3: Hot-spots
Data has specific hotspot locations with very high replication counts
```python
results_fig_path = './results_figure_rep_1d_hotspots/'
Path(results_fig_path).mkdir(parents=True, exist_ok=True)
xtrain, ytrain, xtest, ytrue = make_rep_data_hotspots(
    n_unique=50,
    hotspots=((0.15, 10, 15), (0.50, 18, 25), (0.80, 12, 20)),
    base_rep_choices=(1,),
    noise_std=(0.05, 0.08, 0.10),
    seed=7
)
```

## Model Configuration
### SUBMETHOD Options:
- **'rep'**: Uses the replicated data submethod 
- **'full'**: Uses the full data submethod 
```python
SUBMETHOD = 'rep'   # Choose 'rep' or 'full'
```

### PLOT_MODE Options:
- **'g'**: : Plots the latent Gaussian processes g₁(x), g₂(x), g₃(x) with training points
- **'y'**: Plots the output functions f₁(x), f₂(x), f₃(x) with replicates and credible bands
```python
PLOT_MODE = 'g'     # Choose 'g' or 'y'
```