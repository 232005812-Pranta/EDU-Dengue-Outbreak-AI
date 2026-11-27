"""
Offline ward risk map (bonus feature).

No map tiles, no topojson, no internet: a schematic map drawn with plain Plotly
scatter traces on longitude/latitude axes. The dataset has no ward coordinates, so
wards are laid out deterministically (sunflower spiral) around each city's approximate
centre. Positions are ILLUSTRATIVE; city placement is geographically approximate.
"""
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Approximate city-corporation centres (lat, lon); used only for schematic placement.
CITY_CENTERS = {
    "Dhaka North": (23.87, 90.40),
    "Dhaka South": (23.72, 90.41),
    "Gazipur": (24.00, 90.43),
    "Narayanganj": (23.62, 90.50),
    "Chattogram": (22.36, 91.80),
}
CITY_SPREAD = {"Dhaka North": 0.055, "Dhaka South": 0.055, "Gazipur": 0.035,
               "Narayanganj": 0.035, "Chattogram": 0.06}
TIER_COLORS = {"Low": "#16a34a", "Medium": "#eab308", "High": "#f97316", "Critical": "#dc2626"}
GOLDEN_ANGLE = math.pi * (3 - math.sqrt(5))


def ward_coordinates(wards: pd.DataFrame) -> pd.DataFrame:
    """Deterministic schematic (lat, lon) for every ward: sunflower layout per city."""
    rows = []
    for city, grp in wards.drop_duplicates("ward_id").groupby("city"):
        clat, clon = CITY_CENTERS.get(city, (23.8, 90.4))
        spread = CITY_SPREAD.get(city, 0.05)
        ids = sorted(grp["ward_id"])
        n = len(ids)
        for i, w in enumerate(ids):
            r = spread * math.sqrt((i + 0.5) / n)
            a = i * GOLDEN_ANGLE
            rows.append({"ward_id": w, "lat": clat + r * math.sin(a), "lon": clon + r * math.cos(a)})
    return pd.DataFrame(rows)


def _add_city(fig, df, transfers, labels, tier_names, row, col, axis_suffix, show_legend, shown):
    pos = df.set_index("ward_id")[["lat", "lon"]]
    if transfers is not None and len(transfers):
        for t in transfers.itertuples():
            if t.from_ward in pos.index and t.to_ward in pos.index:
                a, b = pos.loc[t.from_ward], pos.loc[t.to_ward]
                fig.add_annotation(x=b["lon"], y=b["lat"], ax=a["lon"], ay=a["lat"],
                                   xref=f"x{axis_suffix}", yref=f"y{axis_suffix}",
                                   axref=f"x{axis_suffix}", ayref=f"y{axis_suffix}",
                                   showarrow=True, arrowhead=3, arrowsize=1.2, arrowwidth=1.8,
                                   arrowcolor="#2563eb", opacity=0.8)
    for tier, color in TIER_COLORS.items():
        sub = df[df["risk_tier"] == tier]
        if sub.empty:
            continue
        size = 14 + 3.5 * np.sqrt(sub["predicted_cases"].clip(lower=0))
        fig.add_trace(go.Scatter(
            x=sub["lon"], y=sub["lat"], mode="markers+text",
            name=tier_names.get(tier, tier), legendgroup=tier,
            showlegend=show_legend and tier not in shown,
            text=sub["ward_id"].str.split("_").str[-1],
            textposition="middle center", textfont=dict(size=9, color="white"),
            marker=dict(size=size, color=color, opacity=0.9,
                        line=dict(width=np.where(sub["supply_deficit"], 3.5, 1.0),
                                  color=np.where(sub["supply_deficit"], "#111827", "white"))),
            customdata=np.stack([sub["ward_id"], sub["city"], sub["predicted_cases"].round(1),
                                 sub["fulfillment_pct"], sub["crews_allocated"]], axis=-1),
            hovertemplate=(f"<b>%{{customdata[0]}}</b> (%{{customdata[1]}})<br>"
                           f"{labels['cases']}: %{{customdata[2]}}<br>"
                           f"{labels['fulfil']}: %{{customdata[3]}}%<br>"
                           f"{labels['crews']}: %{{customdata[4]}}<extra></extra>"),
        ), row=row, col=col)
        shown.add(tier)


def build_risk_map(alloc: pd.DataFrame, transfers: pd.DataFrame, labels: dict,
                   tier_names: dict = None, city_filter: str = None) -> go.Figure:
    """
    alloc: allocation table (ward_id, city, risk_tier, predicted_cases, supply_deficit, ...).
    transfers: overflow routing orders (from_ward, to_ward, patients) -> drawn as arrows.
    city_filter=None -> one panel per city (each zoomed to its own wards).
    Pure Plotly scatter: no tiles, no topojson, no network.
    """
    from plotly.subplots import make_subplots

    tier_names = tier_names or {t: t for t in TIER_COLORS}
    df = alloc.merge(ward_coordinates(alloc), on="ward_id")
    cities = [city_filter] if city_filter else [c for c in CITY_CENTERS if c in set(df["city"])]
    cols = 1 if city_filter else 3
    rows = math.ceil(len(cities) / cols)
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[f"<b>{c}</b>" for c in cities],
                        horizontal_spacing=0.04, vertical_spacing=0.08)
    shown = set()
    for i, city in enumerate(cities):
        r, c = i // cols + 1, i % cols + 1
        suffix = "" if i == 0 else str(i + 1)
        _add_city(fig, df[df["city"] == city], transfers, labels, tier_names, r, c, suffix, True, shown)
        fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=r, col=c)
        fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=r, col=c,
                         scaleanchor=f"x{suffix}", scaleratio=1.0 / math.cos(math.radians(23.5)))
    fig.update_layout(
        height=520 if city_filter else 330 * rows + 60,
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(orientation="h", y=1.08, x=0),
        plot_bgcolor="rgba(241,245,249,0.7)",
    )
    return fig
