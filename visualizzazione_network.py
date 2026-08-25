import pandas as pd
import networkx as nx
import plotly.graph_objects as go

# =========================================================
# SETTINGS
# =========================================================

EDGE_FILE = "LNG_multilayer_edges_monthly_v1.csv"
NODE_FILE = "LNG_multilayer_nodes_v1.csv"

MONTH = "2023-12"
TOP_N_EDGES = 75

# =========================================================
# LOAD DATA
# =========================================================

edges = pd.read_csv(EDGE_FILE)
nodes = pd.read_csv(NODE_FILE)

print("EDGE COLUMNS:")
print(edges.columns.tolist())

print("\nNODE COLUMNS:")
print(nodes.columns.tolist())

# Create node lookup
node_info = nodes.set_index("node_id").to_dict("index")

# =========================================================
# SELECT MONTH
# =========================================================

month_df = edges[
    edges["period_month"].astype(str) == MONTH
].copy()

print(f"\nMonth: {MONTH}")
print(f"Active edges before filtering: {len(month_df)}")

# Keep largest flows for readability
month_df = (
    month_df
    .sort_values("lng_flow_cmb", ascending=False)
    .head(TOP_N_EDGES)
    .copy()
)

# =========================================================
# BUILD NETWORK
# =========================================================

G = nx.DiGraph()

for _, row in month_df.iterrows():

    source = row["from_node_id"]
    target = row["to_node_id"]

    for node in [source, target]:

        info = node_info.get(node, {})

        G.add_node(
            node,
            node_type=info.get("node_type", "unknown"),
            name=info.get("node_name", node)
        )

    G.add_edge(
        source,
        target,
        flow=row["lng_flow_cmb"],
        voyages=row["voyage_count"],
        routes=row["route_count"],
        edge_type=row["edge_type"],
        weight=row["weight_row_normalized"]
    )

print(f"Nodes displayed: {G.number_of_nodes()}")
print(f"Edges displayed: {G.number_of_edges()}")

# =========================================================
# LAYOUT
# =========================================================

pos = nx.spring_layout(
    G,
    seed=42,
    k=0.8,
    iterations=150,
    weight="flow"
)

# =========================================================
# EDGES
# =========================================================

edge_x = []
edge_y = []

for source, target in G.edges():

    x0, y0 = pos[source]
    x1, y1 = pos[target]

    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])

edge_trace = go.Scatter(
    x=edge_x,
    y=edge_y,
    mode="lines",
    hoverinfo="none",
    line=dict(
        width=0.8,
        color="rgba(130,130,130,0.30)"
    )
)

# =========================================================
# NODE TRAFFIC
# =========================================================

weighted_degree = dict(G.degree(weight="flow"))

max_traffic = max(weighted_degree.values())

# =========================================================
# TERMINALS
# =========================================================

terminal_x = []
terminal_y = []
terminal_text = []
terminal_size = []

# =========================================================
# CHOKEPOINTS
# =========================================================

choke_x = []
choke_y = []
choke_text = []
choke_size = []

for node in G.nodes():

    x, y = pos[node]

    info = G.nodes[node]

    node_type = str(info.get("node_type", "unknown")).lower()
    name = info.get("name", node)

    traffic = weighted_degree.get(node, 0)

    hover = (
        f"<b>{name}</b>"
        f"<br>ID: {node}"
        f"<br>Type: {node_type}"
        f"<br>LNG flow: {traffic:,.0f} cmb"
        f"<br>Degree: {G.degree(node)}"
        f"<br>In-degree: {G.in_degree(node)}"
        f"<br>Out-degree: {G.out_degree(node)}"
    )

    size = 9 + 30 * (traffic / max_traffic) ** 0.5

    if "choke" in node_type:

        choke_x.append(x)
        choke_y.append(y)
        choke_text.append(hover)
        choke_size.append(size)

    else:

        terminal_x.append(x)
        terminal_y.append(y)
        terminal_text.append(hover)
        terminal_size.append(size)

# =========================================================
# TERMINAL TRACE
# =========================================================

terminal_trace = go.Scatter(
    x=terminal_x,
    y=terminal_y,

    mode="markers",

    name="LNG terminals",

    hoverinfo="text",
    text=terminal_text,

    marker=dict(
        size=terminal_size,
        symbol="circle",
        line=dict(width=1)
    )
)

# =========================================================
# CHOKEPOINT TRACE
# =========================================================

choke_trace = go.Scatter(
    x=choke_x,
    y=choke_y,

    mode="markers",

    name="Chokepoints",

    hoverinfo="text",
    text=choke_text,

    marker=dict(
        size=choke_size,
        symbol="diamond",
        line=dict(width=1.5)
    )
)

# =========================================================
# FIGURE
# =========================================================

fig = go.Figure(
    data=[
        edge_trace,
        terminal_trace,
        choke_trace
    ]
)

fig.update_layout(

    title=(
        f"LNG Dynamic Multilayer Network — {MONTH}"
        f"<br><sup>"
        f"Top {TOP_N_EDGES} edges by LNG flow"
        f"</sup>"
    ),

    showlegend=True,

    hovermode="closest",

    margin=dict(
        b=20,
        l=20,
        r=20,
        t=90
    ),

    xaxis=dict(
        showgrid=False,
        zeroline=False,
        showticklabels=False
    ),

    yaxis=dict(
        showgrid=False,
        zeroline=False,
        showticklabels=False
    ),

    height=850
)

fig.show()

# =========================================================
# SAVE
# =========================================================

OUTPUT = f"LNG_multilayer_network_{MONTH}.html"

fig.write_html(OUTPUT)

print(f"\nSaved as: {OUTPUT}")