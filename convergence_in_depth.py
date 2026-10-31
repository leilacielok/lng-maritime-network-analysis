from pathlib import Path
import xarray as xr

"reparametrized model outputs"
source = Path(
    r""C:\Users\Leila\OneDrive\Desktop\Thesis\code\outputs\model\node_effects_1\chokepoint\20261030T155547550840Z_chokepoint_bounded_original"\chokepoint\posterior.nc"
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

with xr.open_dataset(source, group="posterior") as posterior:
    missing = [name for name in variables if name not in posterior]
    if missing:
        raise ValueError(f"Missing variables: {missing}")

    reduced = posterior[variables]
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

print(f"File creato: {destination}")
print(f"Dimensione: {destination.stat().st_size / 1024**2:.1f} MB")