from pathlib import Path
import xarray as xr

# Reparametrized node-effects model: chokepoints
source = Path(
    r"C:\Users\Leila\OneDrive\Desktop\Thesis\code\outputs\model\node_effects_1\chokepoint\20261030T155547550840Z_chokepoint_bounded_original\chokepoint\posterior.nc"
)

destination = source.with_name("posterior_diagnostic_subset.nc")

variables = [
    "mu_node",
    "mu_raw",
    "sigma_mu",
    "node_mean",
    "node_deviation",
    "node_contrast_raw",
    "beta",
    "intercept_cont",
    "intercept_binary",
    "intercept_zero",
    "intercept_cont_centered",
    "intercept_binary_centered",
    "intercept_zero_centered",
    "loading_cont",
    "loading_binary",
    "loading_zero",
    "precision_cont",
]

if not source.exists():
    raise FileNotFoundError(f"File non trovato: {source}")

# Extract selected posterior variables
with xr.open_dataset(source, group="posterior") as posterior:

    available = [
        name for name in variables
        if name in posterior.data_vars
    ]

    missing = [
        name for name in variables
        if name not in posterior.data_vars
    ]

    print(f"Variabili disponibili: {len(available)}")

    if missing:
        print("\nVariabili mancanti:")
        for name in missing:
            print(f"  - {name}")

    if not available:
        raise ValueError("Nessuna variabile richiesta è disponibile.")

    reduced = posterior[available].load()

    reduced.to_netcdf(
        destination,
        group="posterior",
        mode="w",
        engine="h5netcdf",
        encoding={
            name: {"zlib": True, "complevel": 4}
            for name in reduced.data_vars
        },
    )

# Preserve sampling diagnostics
with xr.open_dataset(source, group="sample_stats") as stats:

    stats.to_netcdf(
        destination,
        group="sample_stats",
        mode="a",
        engine="h5netcdf",
        encoding={
            name: {"zlib": True, "complevel": 4}
            for name in stats.data_vars
        },
    )

print("\nFile creato:")
print(destination)

print(
    f"Dimensione: "
    f"{destination.stat().st_size / 1024**2:.1f} MB"
)