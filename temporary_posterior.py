import arviz as az
import pandas as pd

idata = az.from_netcdf(r"C:\Users\Leila\OneDrive\Desktop\Thesis\code\outputs\model\baseline\terminal\20261022T142047738901Z_terminal_bounded_original\terminal\posterior.nc")

stats = idata.sample_stats

print(list(stats.data_vars))

summary = pd.DataFrame({
    "chain": range(stats.sizes["chain"]),
    "mean_n_steps": stats["n_steps"].mean(dim="draw").values,
    "max_n_steps": stats["n_steps"].max(dim="draw").values,
    "mean_tree_depth": stats["tree_depth"].mean(dim="draw").values,
    "mean_step_size": stats["step_size"].mean(dim="draw").values,
    "divergences": stats["diverging"].sum(dim="draw").values,
})

summary.to_csv("nuts_chain_summary.csv", index=False)

print(summary)

print(
    stats["reached_max_treedepth"]
    .sum(dim="draw")
    .values
)