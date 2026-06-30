from __future__ import annotations
import json
from pathlib import Path
from fifa2026.pipeline import PredictionResult
from fifa2026 import viz

REPORTS = Path("reports")

def _load(name, default):
    p = REPORTS / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default

def build_dashboard(prediction: dict, evaluation: dict, market: dict | None) -> dict:
    result = PredictionResult.from_dict(prediction)
    figs = {
        "champion": viz.champion_bar(result, market=market),
        "bracket": viz.bracket_figure(result),
        "survival": viz.survival_heatmap(result),
        "calibration": viz.calibration_curve(evaluation.get("calibration", [])),
    }
    if market:
        figs["market"] = viz.market_compare(result.champion_probs, market)
    return figs

def main():
    import streamlit as st
    st.set_page_config(page_title="FIFA 2026 Champion Predictor", layout="wide")
    prediction = _load("prediction.json", None)
    if prediction is None:
        st.error("No prediction found. Run `make predict` first.")
        return
    evaluation = _load("evaluation.json", {"metrics": {}, "calibration": []})
    market = evaluation.get("market") or None
    st.title("🏆 FIFA World Cup 2026 — Champion Predictor")
    st.caption(f"Forecast as of {prediction.get('as_of', '')}")
    figs = build_dashboard(prediction, evaluation, market)
    tabs = st.tabs(["Champion odds", "Bracket", "Survival", "Team explorer",
                    "Calibration", "Availability", "Predicted path"])
    with tabs[0]:
        st.plotly_chart(figs["champion"], use_container_width=True)
        if "market" in figs:
            st.plotly_chart(figs["market"], use_container_width=True)
    with tabs[1]:
        st.plotly_chart(figs["bracket"], use_container_width=True)
    with tabs[2]:
        st.plotly_chart(figs["survival"], use_container_width=True)
    with tabs[3]:
        result = PredictionResult.from_dict(prediction)
        team = st.selectbox("Team", sorted(result.champion_probs))
        st.write({"champion_prob": result.champion_probs[team], **result.round_probs[team]})
    with tabs[4]:
        st.plotly_chart(figs["calibration"], use_container_width=True)
        st.write(evaluation.get("metrics", {}))
    with tabs[5]:
        st.subheader("Availability impact")
        avail = (prediction.get("meta") or {}).get("availability") or {}
        if not avail:
            st.info("No availability adjustments applied. Edit "
                    "data/reference/injuries_2026.yaml and re-run `make predict`.")
        else:
            import pandas as pd
            rows = [{"team": t, "players out": ", ".join(v.get("out", [])),
                     "elo penalty": v.get("elo_penalty")} for t, v in avail.items()]
            st.dataframe(pd.DataFrame(rows).sort_values("elo penalty"))

    with tabs[6]:
        st.subheader("Predicted path to the Final")
        st.caption("Single most-likely bracket — each tie's predicted winner advances. "
                   "Shootouts are modeled as ~50/50 (no penalty data).")
        path = (prediction.get("meta") or {}).get("predicted_path") or {}
        if not path.get("rounds"):
            st.info("No predicted path available. Run `make predict`.")
        else:
            import pandas as pd
            st.success(f"Predicted champion: {path.get('champion')}")
            for rnd in path["rounds"]:
                st.markdown(f"**{rnd['round']}**")
                rows = [{
                    "match": f"{m['team_a']} vs {m['team_b']}",
                    "P(A win)": round(m["p_a_reg"], 3),
                    "P(draw)": round(m["p_draw"], 3),
                    "P(B win)": round(m["p_b_reg"], 3),
                    "shootout A": round(m["p_a_shootout"], 2),
                    "P(adv A)": round(m["p_a_advance"], 3),
                    "winner": m["winner"],
                } for m in rnd["matches"]]
                st.dataframe(pd.DataFrame(rows))

if __name__ == "__main__":
    main()
