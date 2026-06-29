from __future__ import annotations
import plotly.graph_objects as go


def _sorted_items(d, top_n):
    return sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:top_n]


def champion_bar(result, top_n=12, market=None) -> go.Figure:
    items = _sorted_items(result.champion_probs, top_n)
    teams = [t for t, _ in items]
    fig = go.Figure([go.Bar(x=teams, y=[p for _, p in items], name="Model")])
    if market:
        fig.add_bar(x=teams, y=[market.get(t, 0.0) for t in teams], name="Market")
    fig.update_layout(title="Champion probability", yaxis_tickformat=".0%", barmode="group")
    return fig


def bracket_figure(result) -> go.Figure:
    ties = result.tie_probs
    labels = [f'{t["home"]} vs {t["away"]}' for t in ties]
    fig = go.Figure([go.Bar(x=[t["p_home"] for t in ties], y=labels, orientation="h")])
    fig.update_layout(title="Round-of-32 ties: P(first team advances)", xaxis_tickformat=".0%")
    return fig


def survival_heatmap(result, top_n=16) -> go.Figure:
    stages = ["reach_R16", "reach_QF", "reach_SF", "reach_final", "win"]
    items = _sorted_items(result.champion_probs, top_n)
    teams = [t for t, _ in items]
    z = [[result.round_probs[t][s] for s in stages] for t in teams]
    fig = go.Figure(go.Heatmap(z=z, x=stages, y=teams, colorscale="Blues",
                               zmin=0, zmax=1))
    fig.update_layout(title="Per-round survival probability")
    return fig


def market_compare(model_probs, market_probs, top_n=10) -> go.Figure:
    teams = [t for t, _ in _sorted_items(model_probs, top_n)]
    fig = go.Figure()
    fig.add_bar(x=teams, y=[model_probs.get(t, 0.0) for t in teams], name="Model")
    fig.add_bar(x=teams, y=[market_probs.get(t, 0.0) for t in teams], name="Market")
    fig.update_layout(title="Model vs market", barmode="group", yaxis_tickformat=".0%")
    return fig


def calibration_curve(calibration) -> go.Figure:
    fig = go.Figure()
    if calibration:
        xs = [pt["pred"] for pt in calibration]
        ys = [pt["obs"] for pt in calibration]
        fig.add_scatter(x=xs, y=ys, mode="markers+lines", name="Model")
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect",
                    line=dict(dash="dash"))
    fig.update_layout(title="Calibration", xaxis_title="Predicted", yaxis_title="Observed")
    return fig
