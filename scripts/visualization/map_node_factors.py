r"""Map time-averaged posterior factor and residual for terminals or chokepoints.

Examples (run from repository root):
 python scripts/visualization/map_node_factors.py --node-type terminal \
   --latent PATH/terminal/latent_summary.csv \
   --panel data/processed/model_ready/terminal_criticality_model_panel.csv \
   --coordinates PATH/LNG_terminal.csv 
   --output "thesis/images/terminal_baseline_mean_factor_map.pdf"

 python scripts/visualization/map_node_factors.py --node-type chokepoint \
   --latent "C:\Users\Leila\thesis_code\outputs\model\baseline\chokepoint\chokepoint_rank1_20000_30000\chokepoint\review_20261110T105921194960Z\latent_summary.csv" \
   --panel data/processed/model_ready/chokepoint_criticality_model_panel.csv \
   --chokepoint-geometry data/curated/PortWatch_28_chokepoints_geometry.geojson \
   --output "thesis/images/chokepoint_baseline_mean_factor_map.pdf"

"""
import argparse
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def normalized(value):
    text = unicodedata.normalize('NFKD', str(value)).encode('ascii', 'ignore').decode().lower()
    return re.sub(r'[^a-z0-9]+', '', text)


def coordinates_from_geojson(path, names):
    """Get representative points from chokepoint geometries; never silently guess matches."""
    from shapely.geometry import shape
    raw = json.loads(Path(path).read_text(encoding='utf-8'))
    features = raw.get('features', [])
    if not features:
        raise ValueError('No GeoJSON features found')
    # Common PortWatch naming fields; adapt --coordinates if schema differs.
    candidates = ['node_id', 'chokepoint_id', 'id', 'name', 'Name', 'NAME',
                  'chokepoint', 'Chokepoint', 'location', 'Location', 'port_name']
    rows = []
    for feat in features:
        props = feat.get('properties') or {}
        geom = feat.get('geometry')
        if not geom:
            continue
        point = shape(geom).representative_point()
        row = {'latitude': point.y, 'longitude': point.x}
        for key in candidates:
            if props.get(key) not in (None, ''):
                row[key] = str(props[key])
        rows.append(row)
    coords = pd.DataFrame(rows)
    if coords.empty:
        raise ValueError('GeoJSON contains no usable geometries')
    print('GeoJSON properties available:', sorted((features[0].get('properties') or {}).keys()))
    # Prefer unique exact normalized match on either ID or name.
    matches = []
    for node in names.itertuples(index=False):
        target_id, target_name = str(node.node_id), normalized(node.node_name)
        found = []
        for _, row in coords.iterrows():
            idmatch = any(str(row.get(k, '')).casefold() == target_id.casefold()
                          for k in ['node_id', 'chokepoint_id', 'id'])
            namematch = any(normalized(row.get(k, '')) == target_name
                            for k in ['name', 'Name', 'NAME', 'chokepoint',
                                      'Chokepoint', 'location', 'Location', 'port_name'])
            if idmatch or namematch:
                found.append(row)
        if len(found) == 1:
            matches.append({'node_id': target_id, 'latitude': found[0]['latitude'],
                            'longitude': found[0]['longitude']})
        elif len(found) > 1:
            print(f'WARNING: ambiguous GeoJSON match for {node.node_name}; skipping')
    return pd.DataFrame(matches, columns=['node_id', 'latitude', 'longitude'])


def coordinates_from_csv(path, node_type):
    coords = pd.read_csv(path)
    coords = coords.rename(columns={
        'lat': 'latitude',
        'lon': 'longitude'
    })

    if not {'latitude', 'longitude'}.issubset(coords.columns):
        raise ValueError(
            'Coordinates CSV must contain lat/lon or latitude/longitude'
        )

    if 'node_id' in coords.columns:
        key = 'node_id'
    elif node_type == 'terminal' and 'terminal_id' in coords.columns:
        key = 'terminal_id'
    elif 'node_name' in coords.columns:
        key = 'node_name'
    elif 'name' in coords.columns:
        coords = coords.rename(columns={'name': 'node_name'})
        key = 'node_name'
    else:
        raise ValueError('No suitable identifier found in coordinates CSV')

    coords = coords[[key, 'latitude', 'longitude']].copy()
    coords['latitude'] = pd.to_numeric(coords['latitude'], errors='coerce')
    coords['longitude'] = pd.to_numeric(coords['longitude'], errors='coerce')
    coords = coords.dropna(subset=['latitude', 'longitude'])

    coords = coords.groupby(key, as_index=False)[
        ['latitude', 'longitude']
    ].mean()

    return coords, key


def plot_maps(data, output, node_type):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Select five highest and five lowest mean factor scores
    highest = data.nlargest(5, 'factor_mean').sort_values(
        'factor_mean', ascending=False
    )
    lowest = data.nsmallest(5, 'factor_mean').sort_values(
        'factor_mean', ascending=True
    )

    selected = pd.concat(
        [highest, lowest]
    ).drop_duplicates(subset='node_id').copy()

    selected['label_number'] = range(1, len(selected) + 1)

    # Figure: map on the left, legend on the right
    fig = plt.figure(figsize=(16, 7))

    # Map
    ax = fig.add_axes(
        [0.03, 0.15, 0.68, 0.80],
        projection=ccrs.Robinson()
    )

    ax.set_global()

    ax.add_feature(
        cfeature.LAND,
        facecolor='#f2f2f2',
        edgecolor='none'
    )
    ax.add_feature(
        cfeature.OCEAN,
        facecolor='white'
    )
    ax.coastlines(
        linewidth=0.5,
        color='#777777'
    )

    values = data['factor_mean'].to_numpy(dtype=float)

    limit = np.nanmax(np.abs(values))
    if not np.isfinite(limit) or limit == 0:
        limit = 1.0

    norm = TwoSlopeNorm(
        vmin=-limit,
        vcenter=0,
        vmax=limit
    )

    scatter = ax.scatter(
        data['longitude'],
        data['latitude'],
        c=values,
        cmap='RdBu_r',
        norm=norm,
        s=45 if node_type == 'chokepoint' else 25,
        edgecolors='black',
        linewidths=0.3,
        transform=ccrs.PlateCarree(),
        zorder=5
    )

    # Number selected nodes
    for row in selected.itertuples():
        ax.annotate(
            str(row.label_number),
            xy=(row.longitude, row.latitude),
            xycoords=ccrs.PlateCarree()._as_mpl_transform(ax),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9,
            fontweight='bold',
            color='black',
            bbox=dict(
                boxstyle='circle,pad=0.2',
                facecolor='white',
                edgecolor='black',
                linewidth=0.5,
                alpha=0.95
            ),
            zorder=10
        )

    # Colorbar below the map
    cbar_ax = fig.add_axes(
        [0.19, 0.12, 0.36, 0.025]
    )

    cbar = fig.colorbar(
        scatter,
        cax=cbar_ax,
        orientation='horizontal'
    )

    cbar.set_label(
        'Mean posterior latent factor',
        fontsize=11,
        labelpad=5
    )
    cbar.ax.tick_params(labelsize=9)

    # Legend area on the right
    legend_ax = fig.add_axes(
        [0.73, 0.15, 0.25, 0.75]
    )
    legend_ax.axis('off')

    legend_ax.text(
        0.0, 0.95,
        'Five highest mean factor scores',
        fontsize=12,
        fontweight='bold',
        va='top',
        transform=legend_ax.transAxes
    )

    for i, row in enumerate(selected.iloc[:5].itertuples()):
        legend_ax.text(
            0.0,
            0.88 - i * 0.075,
            f'{row.label_number}. {row.node_name}\n'
            f'    ({row.factor_mean:+.2f})',
            fontsize=10,
            va='top',
            transform=legend_ax.transAxes
        )

    legend_ax.text(
        0.0, 0.45,
        'Five lowest mean factor scores',
        fontsize=12,
        fontweight='bold',
        va='top',
        transform=legend_ax.transAxes
    )

    for i, row in enumerate(selected.iloc[5:].itertuples()):
        legend_ax.text(
            0.0,
            0.38 - i * 0.075,
            f'{row.label_number}. {row.node_name}\n'
            f'    ({row.factor_mean:+.2f})',
            fontsize=10,
            va='top',
            transform=legend_ax.transAxes
        )

    # Save PDF only
    pdf_path = output.with_suffix('.pdf')

    fig.savefig(
        pdf_path,
        bbox_inches='tight'
    )

    plt.close(fig)

    print(f'Map saved: {pdf_path}')
    
def main():

    parser = argparse.ArgumentParser(
        description='Map posterior latent factors and residuals'
    )

    parser.add_argument(
        '--node-type',
        choices=['terminal', 'chokepoint'],
        required=True
    )

    parser.add_argument(
        '--latent',
        required=True,
        help='Path to latent_summary.csv'
    )

    parser.add_argument(
        '--panel',
        required=True,
        help='Path to model panel CSV'
    )

    parser.add_argument(
        '--coordinates',
        default=None,
        help='Coordinates CSV'
    )

    parser.add_argument(
        '--chokepoint-geometry',
        default=None,
        help='Chokepoint GeoJSON'
    )

    parser.add_argument(
        '--output',
        required=True,
        help='Output PNG path'
    )

    args = parser.parse_args()

    print(f'Node type: {args.node_type}')

    latent = pd.read_csv(args.latent)
    panel = pd.read_csv(args.panel)

    print(f'Latent observations: {len(latent)}')
    print(f'Panel observations: {len(panel)}')

    # Identify node ID
    if 'node_id' in latent.columns:
        node_id = 'node_id'
    elif 'terminal_id' in latent.columns:
        node_id = 'terminal_id'
    else:
        raise ValueError('No node identifier found in latent_summary.csv')

    if 'node_name' not in latent.columns:
        names = panel[[node_id, 'node_name']].drop_duplicates()
        latent = latent.merge(
            names,
            on=node_id,
            how='left',
            validate='many_to_one'
        )

    # Identify factor and residual columns
    factor_col = 'factor_mean'

    if factor_col not in latent.columns:
        raise ValueError('factor_mean not found in latent_summary.csv')

    residual_candidates = [
        'factor_residual_mean',
        'residual_mean',
        'factor_residual'
    ]

    residual_col = next(
        (c for c in residual_candidates if c in latent.columns),
        None
    )

    if residual_col is None:
        raise ValueError(
            'No factor residual column found in latent_summary.csv'
        )

    # Average posterior quantities across months
    summary = (
        latent.groupby([node_id, 'node_name'], as_index=False)
        .agg(
            factor_mean=(factor_col, 'mean'),
            factor_residual_mean=(residual_col, 'mean'),
            n_months=(factor_col, 'size')
        )
    )

    summary = summary.rename(columns={node_id: 'node_id'})
    summary['node_id'] = summary['node_id'].astype(str)

    print(f'Nodes in posterior summary: {len(summary)}')

    # Load geographical coordinates
    if args.node_type == 'terminal':

        if args.coordinates is None:
            raise ValueError(
                '--coordinates is required for terminals'
            )

        coords, key = coordinates_from_csv(
            args.coordinates,
            args.node_type
        )

        if key == 'terminal_id':
            coords = coords.rename(columns={'terminal_id': 'node_id'})
            coords['node_id'] = coords['node_id'].astype(str)

            data = summary.merge(
                coords,
                on='node_id',
                how='left',
                validate='one_to_one'
            )

        elif key == 'node_id':
            coords['node_id'] = coords['node_id'].astype(str)

            data = summary.merge(
                coords,
                on='node_id',
                how='left',
                validate='one_to_one'
            )

        else:
            data = summary.merge(
                coords,
                on='node_name',
                how='left',
                validate='one_to_one'
            )

    else:

        if args.chokepoint_geometry is None:
            raise ValueError(
                '--chokepoint-geometry is required for chokepoints'
            )

        coords = coordinates_from_geojson(
            args.chokepoint_geometry,
            summary[['node_id', 'node_name']]
        )

        data = summary.merge(
            coords,
            on='node_id',
            how='left',
            validate='one_to_one'
        )

    # Check missing coordinates
    missing = data[
        data['latitude'].isna() |
        data['longitude'].isna()
    ]

    if not missing.empty:
        print('\nWARNING: Nodes without coordinates:')
        print(
            missing[['node_id', 'node_name']]
            .to_string(index=False)
        )

    data = data.dropna(
        subset=['latitude', 'longitude']
    )

    if data.empty:
        raise ValueError(
            'No nodes with valid coordinates. Check geographical matching.'
        )

    print(f'Nodes mapped: {len(data)}')

    # Prepare output directory
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Generate maps
    plot_maps(
        data,
        output,
        args.node_type
    )

    print('Done!')


if __name__ == '__main__':
    main()
