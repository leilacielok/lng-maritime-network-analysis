"""Plot the 28 PortWatch locations and four examples of crossing geometries.

Install: python -m pip install openpyxl matplotlib geopandas cartopy
Run: python plot_chokepoint_geometries.py

The script expects the project layout shown below:
scripts/plot_chokepoint_geometries.py
data/curated/PortWatch_28_geometry_registry.xlsx
data/curated/PortWatch_28_chokepoints_geometry*.geojson

The GeoJSON must contain the final geometries and a feature property named
node_id (CP001...CP028), portwatch_id (chokepoint1...chokepoint28), or
chokepoint (matching the names in the registry). It is never approximated from
the point coordinates in the Excel file.
"""

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CURATED = PROJECT_ROOT / "data" / "curated"
REGISTRY = CURATED / "PortWatch_28_geometry_registry.xlsx"
OUTPUT = PROJECT_ROOT / "eda_outputs" / "network_level" / "maps" / "chokepoint_geometries.pdf"


def geometry_file():
    matches = sorted(CURATED.glob("PortWatch_28_chokepoints_geometry*.geojson"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one PortWatch geometry GeoJSON in {CURATED}; found {len(matches)}."
        )
    return matches[0]


EXAMPLES = [
    ("CP001", "Published polygon · Suez Canal"),
    ("CP015", "IMO routeing geometry · Lombok Strait"),
    ("CP011", "Study-defined gate · Taiwan Strait"),
    ("CP002", "Study-defined corridor · Panama Canal"),
]
EXCLUDED = {"CP003", "CP025", "CP026", "CP028"}


def load_data(registry_path, geometry_path):
    registry = pd.read_excel(registry_path, sheet_name="Geometry Registry")
    
    shapes = gpd.read_file(geometry_path)
    shapes = shapes.to_crs("EPSG:4326")
    
    if "node_id" in shapes.columns:
        shapes["node_id"] = shapes.node_id.astype(str)
    elif "portwatch_id" in shapes.columns:
        mapping = dict(zip(registry.portwatch_id.astype(str), registry.node_id))
        shapes["node_id"] = shapes.portwatch_id.astype(str).map(mapping)
    elif "chokepoint" in shapes.columns:
        mapping = dict(zip(registry.chokepoint.astype(str).str.casefold(), registry.node_id))
        shapes["node_id"] = shapes.chokepoint.astype(str).str.casefold().map(mapping)
    else:
        raise ValueError("GeoJSON needs node_id, portwatch_id, or chokepoint property.")
    if shapes.node_id.isna().any():
        raise ValueError("Some GeoJSON features could not be matched to the registry.")
    if shapes.geometry.isna().any() or shapes.geometry.is_empty.any():
        raise ValueError("Some GeoJSON features have empty geometries.")
    # Multiple features per chokepoint (e.g., separate TSS sections) are retained.
    return registry, shapes


def basemap(ax, extent=None):
    ax.add_feature(cfeature.LAND, facecolor="#e9e9e6", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="#f5f9fa", edgecolor="none", zorder=0)
    ax.coastlines(resolution="110m", linewidth=0.45, color="#656b6d", zorder=1)
    if extent:
        ax.set_extent(extent, crs=ccrs.PlateCarree())


def draw_geometry(ax, subset):
    # GeoPandas draws on an ordinary Matplotlib Axes. Insets use Plate Carree
    # coordinates; no reprojection is needed for plotting their lon/lat values.
    for geom in subset.geometry:
        if geom.geom_type in {"Polygon", "MultiPolygon"}:
            gpd.GeoSeries([geom], crs="EPSG:4326").plot(
                ax=ax, facecolor="#d98246", edgecolor="#a64a21",
                alpha=0.55, linewidth=1.0, zorder=3
            )
        else:
            gpd.GeoSeries([geom], crs="EPSG:4326").plot(
                ax=ax, color="#a64a21", linewidth=2, zorder=3
            )


def main():
    registry, shapes = load_data(REGISTRY, geometry_file())

    fig = plt.figure(figsize=(12, 10), layout="constrained")
    grid = fig.add_gridspec(3, 2, height_ratios=[1.65, 1, 1])
    world = fig.add_subplot(grid[0, :], projection=ccrs.Robinson())
    basemap(world)
    world.set_global()
    for row in registry.itertuples(index=False):
        color = "#777777" if row.node_id in EXCLUDED else "#164e63"
        world.plot(row.portwatch_lon, row.portwatch_lat, "o", ms=3.4,
                   color=color, transform=ccrs.PlateCarree(), zorder=4)
        world.annotate(
            row.node_id[2:],
            xy=(row.portwatch_lon, row.portwatch_lat),
            xycoords=ccrs.PlateCarree()._as_mpl_transform(world),
            xytext=(2, 1),             
            textcoords="offset points",
            fontsize=5,
            color=color,
            ha="left",
            va="bottom",
            zorder=5,
        )
    world.set_title("A. Locations of the 28 candidate chokepoints", loc="left", fontsize=11)
    world.legend(handles=[
        Line2D([], [], marker="o", linestyle="none", color="#164e63", label="Retained"),
        Line2D([], [], marker="o", linestyle="none", color="#777777", label="Excluded")
    ], loc="lower left", fontsize=8, frameon=True)

    for index, (node_id, title) in enumerate(EXAMPLES):
        ax = fig.add_subplot(grid[1 + index // 2, index % 2])
        subset = shapes.loc[shapes.node_id == node_id]
        bounds = subset.total_bounds
        width = max(bounds[2] - bounds[0], 0.35)
        height = max(bounds[3] - bounds[1], 0.35)
        xpad, ypad = max(width * 0.35, 0.12), max(height * 0.35, 0.12)
        ax.set_xlim(bounds[0] - xpad, bounds[2] + xpad)
        ax.set_ylim(bounds[1] - ypad, bounds[3] + ypad)
        # GeoPandas geometry and Natural Earth coastline share WGS84 axes.
        land = cfeature.NaturalEarthFeature("physical", "land", "10m")
        for geom in land.geometries():
            if geom.bounds[2] < ax.get_xlim()[0] or geom.bounds[0] > ax.get_xlim()[1] or geom.bounds[3] < ax.get_ylim()[0] or geom.bounds[1] > ax.get_ylim()[1]:
                continue
            gpd.GeoSeries([geom], crs="EPSG:4326").plot(
                ax=ax, facecolor="#e9e9e6", edgecolor="#888888", linewidth=0.3, zorder=1
            )
        draw_geometry(ax, subset)
        ax.set_aspect(1 / max(abs(__import__("math").cos(__import__("math").radians((bounds[1] + bounds[3]) / 2))), 0.25))
        ax.set_title(f"{chr(66 + index)}. {title}", loc="left", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("Longitude", fontsize=7)
        ax.set_ylabel("Latitude", fontsize=7)

    fig.suptitle("Chokepoint locations and examples of route-crossing geometries", fontsize=13)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
