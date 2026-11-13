"""Exploratory PCA' terminal/chokepoint response vectors.

Run from repository root:
 python scripts/visualization/pca_criticality.py --node-type terminal
 python scripts/visualization/pca_criticality.py --node-type chokepoint
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

RESPONSES = {
    'terminal': {
        'role_specific_pagerank': 'PageRank',
        'counterparty_hhi_terminal': 'HHI',
        'role_specific_dependence': 'Dependence',
        'is_articulation_point': 'AP',
        'has_incident_global_bridge': 'GB',
        'connected_to_articulation_point': 'toAP',
    },
    'chokepoint': {
        'pagerank': 'PageRank',
        'betweenness': 'Betweenness',
        'share_monthly_network_flow': 'Flow share',
        'is_articulation_point': 'AP',
        'connected_to_articulation_point': 'toAP',
    },
}
COVARIATES = {
    'terminal': ['neighbor_activity_lag1','log1p_mean_voyage_distance_lag1',
                 'month_sin','month_cos','log1p_capacity_mtpa','terminal_age','terminal_role_lag1'],
    'chokepoint': ['neighbor_activity_lag1','log1p_mean_voyage_distance_lag1','month_sin','month_cos'],
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--node-type', required=True, choices=RESPONSES)
    parser.add_argument('--input', type=Path, help='Override default input CSV')
    parser.add_argument('--output', type=Path, help='Override default output folder')
    parser.add_argument('--permutations', type=int, default=500, help='Parallel analysis permutations')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    kind = args.node_type
    path = args.input or Path(f'data/processed/model_ready/{kind}_criticality_model_panel.csv')
    out = args.output or Path('eda_outputs/pca') / kind
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(path)
    response_map = RESPONSES[kind]
    required = list(response_map) + COVARIATES[kind]
    absent = [c for c in required + ['active', 'activity_lag1'] if c not in df]
    if absent:
        raise ValueError(f'Missing columns in {path}: {absent}')
    # Match estimation sample: active at t and t-1, all response/covariate values available.
    eligible = (df['active'] == 1) & (df['activity_lag1'] == 1)
    eligible &= df[required].notna().all(axis=1)
    selected = df.loc[eligible].copy()
    X = selected[list(response_map)].astype(float).rename(columns=response_map)
    if len(X) < 3:
        raise ValueError('Too few eligible rows.')
    if (X.std(axis=0) == 0).any():
        raise ValueError('Constant response(s): ' + ', '.join(X.columns[X.std(axis=0) == 0]))
    scaled = StandardScaler().fit_transform(X)
    pca = PCA().fit(scaled)
    components = [f'PC{i}' for i in range(1, len(X.columns)+1)]
    eig = pca.explained_variance_
    variance = pd.DataFrame({
        'component': components,
        'eigenvalue': eig,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_),
    })
    variance.to_csv(out/'explained_variance.csv', index=False)
    
        # Elbow plot
    component_numbers = np.arange(1, len(eig) + 1)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(component_numbers, eig, marker="o", linewidth=1.8)

    ax.axhline(
        y=1,
        linestyle="--",
        color="gray",
        label="Kaiser criterion (eigenvalue = 1)"
    )

    ax.set_xlabel("Principal component")
    ax.set_ylabel("Eigenvalue")
    ax.set_title(f"{kind.capitalize()} PCA: elbow plot")
    ax.set_xticks(component_numbers)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out / "elbow_plot.png", dpi=300)
    plt.close(fig)

    # Eigenvectors (unit norm), not correlations of variables with PC scores.
    pd.DataFrame(pca.components_.T, index=X.columns, columns=components).rename_axis('response').to_csv(out/'component_loadings.csv')
    pd.DataFrame(pca.components_.T * np.sqrt(eig), index=X.columns, columns=components).rename_axis('response').to_csv(out/'variable_pc_correlations.csv')
    X.corr().to_csv(out/'correlation_matrix.csv')
    scores = pd.DataFrame(pca.transform(scaled)[:, :2], columns=['PC1', 'PC2'], index=selected.index)
    for key in ('terminal_id' if kind == 'terminal' else 'node_id', 'node_name', 'period_month'):
        if key in selected:
            scores[key] = selected[key]
    scores.to_csv(out/'pc_scores.csv', index=False)
    
    loadings = pca.components_.T
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.axhline(0, color='gray', lw=.7); ax.axvline(0, color='gray', lw=.7)
    for i, label in enumerate(X.columns):
        ax.arrow(0, 0, loadings[i, 0], loadings[i, 1], length_includes_head=True,
                 head_width=.025, alpha=.75)
        ax.annotate(label, (loadings[i,0], loadings[i,1]), xytext=(4,4), textcoords='offset points')
    ax.set(xlabel=f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',
           ylabel=f'PC2 ({pca.explained_variance_ratio_[1]:.1%})',
           title=f'{kind.capitalize()} PCA: component coefficients')
    ax.set_xlim(-1.05,1.05);ax.set_ylim(-1.05,1.05)
    ax.set_aspect('equal', adjustable='box');fig.tight_layout()
    fig.savefig(out/'loadings_plot.png', dpi=200);plt.close(fig)
    
    print(f'Input: {path}\nSelected observations: {len(X)} / {len(df)}\nOutput: {out}')
    print(variance.to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    print('\nComponent coefficients:\n', pd.DataFrame(loadings[:, :2], index=X.columns, columns=['PC1','PC2']).round(3).to_string())
    print('\nNote: rows are repeated node-month observations; PCA is exploratory, not a formal Bayesian rank test.')

if __name__ == '__main__':
    main()
