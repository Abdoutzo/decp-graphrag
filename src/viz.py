"""Graph visualization with Plotly (deterministic, no CDN at render time).

Node positions come from a seeded spring layout; colors encode the node
type; hovering a node shows its type and relations.
"""
import networkx as nx
import plotly.graph_objects as go

TYPE_COLORS = {
    "contrat": "#4C8DFF",
    "acheteur": "#E67E22",
    "titulaire": "#2ECC71",
    "domaine": "#9B59B6",
    "ville": "#E74C3C",
    "lieu": "#1ABC9C",
    "équipement": "#F1C40F",
    "quantité": "#95A5A6",
    "organisme": "#34495E",
}


def _eur_short(x: float) -> str:
    if x >= 1_000_000:
        return f"{x / 1_000_000:.2f} M€".replace(".", ",")
    return f"{x / 1_000:.0f} k€"


def figure(g: nx.MultiDiGraph, height: int = 560) -> go.Figure:
    pos = nx.spring_layout(g.to_undirected(), seed=42, k=0.9)

    edge_x, edge_y, edge_text = [], [], []
    for u, v, d in g.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode="lines",
                            line=dict(width=0.8, color="#BDC3C7"),
                            hoverinfo="none", showlegend=False)

    node_x, node_y, node_color, node_size, node_text, node_label = (
        [], [], [], [], [], [])
    for n, d in g.nodes(data=True):
        x, y = pos[n]
        ntype = d.get("type", "?")
        node_x.append(x)
        node_y.append(y)
        node_color.append(TYPE_COLORS.get(ntype, "#BDC3C7"))
        node_size.append(16 if ntype == "contrat" else 11)
        rels = sorted({dd.get("relation", "?").replace("_", " ").lower()
                       for _, _, dd in g.edges(n, data=True)} |
                      {dd.get("relation", "?").replace("_", " ").lower()
                       for _, _, dd in g.in_edges(n, data=True)})
        node_text.append(f"<b>{n}</b><br>{ntype}<br>{', '.join(rels)}")
        node_label.append(n if ntype in ("acheteur", "titulaire") else "")
    node_trace = go.Scatter(x=node_x, y=node_y, mode="markers+text",
                            marker=dict(size=node_size, color=node_color,
                                        line=dict(width=1, color="white")),
                            text=node_label, textposition="top center",
                            textfont=dict(size=9),
                            hovertext=node_text, hoverinfo="text",
                            showlegend=False)

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(visible=False), yaxis=dict(visible=False),
                      plot_bgcolor="white", hovermode="closest")
    return fig
