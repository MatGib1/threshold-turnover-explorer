"""
app.py — Interactive Threshold & Turnover Explorer

Math ported directly from parallel_threshold_calc_2026_02_18.py.
No pre-computed files needed — everything runs in-browser on each click.

Usage:
    pip install -r requirements.txt
    python app.py
    Open http://127.0.0.1:8050
"""

import math

import dash
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html
from scipy.optimize import root_scalar
from scipy.stats import binom, norm

# ─── Constants ────────────────────────────────────────────────────────────────
Z_MIN, Z_MAX, N_GRID = -15.0, 15.0, 101
Z_SPACE = np.linspace(Z_MIN, Z_MAX, N_GRID)

N_LIST   = [1, 7, 9, 11]
N_COLORS = {1: "#4C72B0", 7: "#DD8452", 9: "#55A868", 11: "#C44E52"}

N_SIM     = 50
N_SAMPLE  = 500
N_DECILES = 5
ALPHA_0   = 0.0
SEED      = 42

# ─── Core math (ported from parallel_threshold_calc_2026_02_18.py) ────────────

def continuation_value(alpha_hat: float, k: int, tau: float) -> float:
    """E_i[B_k] = sum_{l=0}^{k-1} tau*(base^l) + (1-tau)^k*(1+alpha)^k."""
    base = (1.0 - tau) * (1.0 + alpha_hat)
    val  = sum(tau * (base ** l) for l in range(k))
    val += (1.0 - tau) ** k * (1.0 + alpha_hat) ** k
    return val


def exp_ability_pivotal(y: float, z_star: float, N: int,
                         nu_y: float, nu_z: float, sigma_z: float) -> float:
    """E[alpha | y, z_star, pivotal] — posterior + inverse-Mills correction."""
    alpha_hat = y / (1.0 + nu_y)
    ai  = alpha_hat + (nu_z / (1.0 + nu_y + nu_z)) * (z_star - alpha_hat)
    z_i = (z_star - ai) / sigma_z
    phi = 0.3989422804014327 * math.exp(-0.5 * z_i * z_i)
    PHI = 0.5 * (1.0 + math.erf(z_i / 1.4142135623730951))
    N2  = (N - 1.0) / 2.0
    if PHI >= 0.99:
        yay, nay = N2 * sigma_z * z_i, 0.0
    elif PHI <= 0.01:
        yay, nay = 0.0, (N - 1.0 - N2) * sigma_z * z_i
    else:
        yay = N2 * sigma_z * phi / (1.0 - PHI)
        nay = (N - 1.0 - N2) * sigma_z * phi / PHI
    return ai + (yay - nay) / N


def F_vectorized(z_arr: np.ndarray, y: float, c_d: float, N: int, c_bar: float,
                  sigma_y: float, sigma_z: float,
                  beta: float, k: int, tau: float) -> np.ndarray:
    """
    F(z*) = P(pivotal)*benefit - c_d*P(dissent).
    Vectorized over z_arr for a fixed y.
    Vote fire iff z_j < z*  →  p = P(z_j < z*).
    """
    nu_y      = 1.0 / sigma_y ** 2
    nu_z      = 1.0 / sigma_z ** 2
    alpha_hat = y / (1.0 + nu_y)
    ai_vec    = alpha_hat + (nu_z / (1.0 + nu_y + nu_z)) * (z_arr - alpha_hat)
    p_fire    = np.clip(norm.cdf((z_arr - ai_vec) / sigma_z), 0.01, 0.99)

    E0      = continuation_value(ALPHA_0, k, tau)
    benefit = np.array([
        (beta ** k) * (E0 - continuation_value(
            exp_ability_pivotal(y, float(z), N, nu_y, nu_z, sigma_z), k, tau
        )) - c_bar
        for z in z_arr
    ])

    if N == 1:
        ppiv  = np.ones_like(p_fire)
        pdiss = np.zeros_like(p_fire)
    else:
        no, nf = N - 1, (N - 1) // 2
        ppiv   = binom.pmf(nf, no, p_fire)
        pdiss  = binom.cdf(nf - 1, no, p_fire)

    return ppiv * benefit - c_d * pdiss


def compute_threshold_curve(c_d: float, c_bar: float, sigma_y: float, sigma_z: float,
                              N: int, beta: float, k: int, tau: float) -> list:
    """
    For each y on Z_SPACE, find z*(y) = largest root of F(z*) = 0.
    Returns a list of floats (NaN where no threshold exists).
    """
    out      = np.full(len(Z_SPACE), np.nan)
    z_min    = float(Z_SPACE[0])
    replaced = False   # replace first -inf (near bottom of y grid) with z_min

    for i, y in enumerate(Z_SPACE):
        fv = F_vectorized(Z_SPACE, float(y), c_d, N, c_bar, sigma_y, sigma_z, beta, k, tau)

        if np.nanmax(fv) < 0:
            z_s = -np.inf
        elif np.nanmin(fv) > 0:
            z_s = np.inf
        else:
            roots = list(Z_SPACE[fv == 0])
            for j in np.where(np.diff(np.sign(fv)) != 0)[0]:
                a, b = float(Z_SPACE[j]), float(Z_SPACE[j + 1])
                try:
                    r = root_scalar(
                        lambda z, _y=float(y), _cd=c_d, _N=N, _cb=c_bar,
                               _sy=sigma_y, _sz=sigma_z, _bt=beta, _k=k, _t=tau:
                            float(F_vectorized(np.array([z]), _y, _cd, _N, _cb,
                                               _sy, _sz, _bt, _k, _t)[0]),
                        bracket=[a, b], method="brentq", xtol=1e-10,
                    ).root
                    roots.append(r)
                except Exception:
                    roots.append(0.5 * (a + b))
            z_s = max(roots) if roots else -np.inf

        out[i] = z_s
        if z_s == -np.inf and float(y) > float(Z_SPACE[0]) and not replaced:
            out[i]   = z_min
            replaced = True

    pos_inf = np.isposinf(out)
    if np.any(pos_inf):
        out[pos_inf] = float(np.max(Z_SPACE))
    out[np.isneginf(out)] = np.nan

    return out.tolist()


# ─── Quintile turnover simulation ────────────────────────────────────────────

def run_simulation(thresholds: dict, tau: float, sigma_alpha: float,
                    sigma_y: float, sigma_z: float) -> dict:
    """
    Draw N_SIM × N_SAMPLE samples, compute mean turnover by quintile for each N.
    thresholds: {str(N): [z_star values aligned to Z_SPACE]}
    Returns:    {str(N): [mean_turnover_q1, ..., mean_turnover_q5]}
    """
    rng       = np.random.default_rng(SEED)
    alpha_all = rng.normal(ALPHA_0, sigma_alpha, (N_SIM, N_SAMPLE))
    y_raw_all = alpha_all + rng.normal(0.0, sigma_y, (N_SIM, N_SAMPLE))

    results = {}

    for N in N_LIST:
        z_noise  = rng.normal(0.0, sigma_z, (N_SIM, N_SAMPLE, N))
        uniform  = rng.random((N_SIM, N_SAMPLE))
        # NaN means threshold = -inf (director never votes fire).
        # Replace with -inf so the board always keeps the CEO for those y values,
        # letting exogenous separation (tau) still produce turnover.
        z_star_raw = np.array(thresholds[str(N)], dtype=float)
        z_star     = np.where(np.isnan(z_star_raw), -np.inf, z_star_raw)

        sum_turn = np.zeros(N_DECILES)
        count    = 0

        for s in range(N_SIM):
            y_raw = y_raw_all[s]

            # Nearest-neighbour snap to Z_SPACE grid
            idx  = np.clip(np.searchsorted(Z_SPACE, y_raw), 1, len(Z_SPACE) - 1)
            i_nn = np.where(
                np.abs(y_raw - Z_SPACE[idx]) <= np.abs(y_raw - Z_SPACE[idx - 1]),
                idx, idx - 1,
            )

            thresh = z_star[i_nn]   # may contain -inf (never fire) but not NaN

            y_v = y_raw
            zt  = thresh
            av  = alpha_all[s]
            zn  = z_noise[s]
            uv  = uniform[s]

            # Keep if majority of N directors observe z > z*
            # z_mat > -inf is always True → board keeps when threshold is -inf
            board_keep = ((av[:, None] + zn) > zt[:, None]).sum(axis=1) > (N / 2)
            final_turn = np.maximum(
                (~board_keep).astype(float),
                (uv < tau).astype(float),   # exogenous separation
            )

            # Quintile assignment by y performance
            nv    = N_SAMPLE
            order = np.argsort(y_v, kind="mergesort")
            ranks = np.empty(nv, dtype=np.int64)
            ranks[order] = np.arange(1, nv + 1)
            dec = np.clip((ranks * N_DECILES - 1) // nv, 0, N_DECILES - 1)

            for d in range(N_DECILES):
                m = dec == d
                if np.any(m):
                    sum_turn[d] += float(final_turn[m].mean())
            count += 1

        results[str(N)] = (sum_turn / count if count > 0 else np.zeros(N_DECILES)).tolist()

    return results


# ─── Figure builders ──────────────────────────────────────────────────────────

_BASE_LAYOUT = dict(
    plot_bgcolor  = "white",
    paper_bgcolor = "white",
    margin        = dict(l=62, r=20, t=52, b=50),
    hovermode     = "x unified",
    legend        = dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
        font=dict(size=12), bgcolor="rgba(255,255,255,0.8)",
    ),
)

_AXIS_STYLE = dict(gridcolor="#f0f0f0", zeroline=True,
                   zerolinecolor="#ddd", zerolinewidth=1)

_EMPTY_HINT = dict(
    text="Click  ▶ Run  to compute",
    xref="paper", yref="paper", x=0.5, y=0.5,
    showarrow=False, font=dict(size=15, color="#ccc"),
)


def _ghost_opacity(idx: int, n: int) -> float:
    """Single ghost = 0.55."""
    if n <= 1:
        return 0.55
    t = idx / (n - 1)
    return 0.35 + 0.20 * t


def _add_y_distribution_overlay(fig: go.Figure, sa: float, sy: float) -> None:
    """
    Overlay the y ~ N(0, σ_α² + σ_y²) density on the threshold plot.
    Density is scaled to occupy the bottom 20% of the z* axis range [-15, 15].
    Quintile boundary lines (20/40/60/80 pct) are drawn as vertical guides.
    """
    y_std  = float(np.sqrt(sa**2 + sy**2))
    y_grid = np.linspace(Z_MIN, Z_MAX, 1000)
    dens   = norm.pdf(y_grid, 0.0, y_std)

    # Scale density to bottom strip: z* ∈ [Z_MIN, Z_MIN + 5]
    z_lo   = float(Z_MIN)           # -15
    z_hi   = z_lo + 5.0             # -10  (5-unit strip at the bottom)
    dens_s = z_lo + (dens / dens.max()) * (z_hi - z_lo)

    # Filled density shape (polygon closed at z_lo baseline)
    x_poly = y_grid.tolist() + y_grid[::-1].tolist()
    y_poly = dens_s.tolist() + [z_lo] * len(y_grid)
    fig.add_trace(go.Scatter(
        x=x_poly, y=y_poly,
        mode="none", fill="toself",
        fillcolor="rgba(160,160,220,0.18)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False, hoverinfo="skip",
    ))
    # Density outline
    fig.add_trace(go.Scatter(
        x=y_grid.tolist(), y=dens_s.tolist(),
        mode="lines",
        line=dict(color="rgba(120,120,200,0.55)", width=1.5, dash="dot"),
        name=f"y ~ N(0, {y_std:.2f}²)",
        showlegend=True, hoverinfo="skip",
    ))

    # Quintile boundary lines (20th, 40th, 60th, 80th pct)
    pcts   = [0.20, 0.40, 0.60, 0.80]
    labels = ["Q1|Q2", "Q2|Q3", "Q3|Q4", "Q4|Q5"]
    for p, lbl in zip(pcts, labels):
        xq = float(norm.ppf(p, 0.0, y_std))
        if Z_MIN <= xq <= Z_MAX:
            fig.add_vline(
                x=xq, line_dash="dot", line_color="#bbb", line_width=1.2,
                annotation_text=lbl, annotation_position="top right",
                annotation_font=dict(size=9, color="#aaa"),
            )

    # Annotation: distribution parameters
    fig.add_annotation(
        text=f"y ~ N(0, {y_std:.2f}²)  (σ_α={sa:.2f}, σ_y={sy:.2f})",
        xref="paper", yref="paper", x=0.02, y=0.04,
        xanchor="left", yanchor="bottom",
        showarrow=False,
        font=dict(size=10, color="#999"),
        bgcolor="rgba(255,255,255,0.7)",
    )


def make_threshold_fig(history: list, current: dict) -> go.Figure:
    fig    = go.Figure()
    y_vals = Z_SPACE.tolist()
    ng     = len(history)

    for ri, run in enumerate(history):
        op = _ghost_opacity(ri, ng)
        for N in N_LIST:
            zv = run.get("thresholds", {}).get(str(N))
            if not zv:
                continue
            fig.add_trace(go.Scatter(
                x=y_vals, y=zv, mode="lines",
                line=dict(color=N_COLORS[N], width=2.5, dash="dot"),
                opacity=op, showlegend=False, hoverinfo="skip",
            ))

    has_current = current and "thresholds" in current
    if has_current:
        for N in N_LIST:
            zv = current["thresholds"].get(str(N))
            if not zv:
                continue
            fig.add_trace(go.Scatter(
                x=y_vals, y=zv, mode="lines",
                line=dict(color=N_COLORS[N], width=2.5),
                name=f"N = {N}",
            ))

        # Overlay y-distribution density + quintile lines
        p = current.get("params", {})
        _add_y_distribution_overlay(fig, float(p.get("sa", 1.0)), float(p.get("sy", 1.0)))

    if not history and not has_current:
        fig.add_annotation(**_EMPTY_HINT)

    fig.update_layout(
        **_BASE_LAYOUT,
        title=dict(text="Firing Threshold  z*(y)", font=dict(size=14, color="#222"), x=0.03),
        xaxis=dict(title="y  (public signal)", **_AXIS_STYLE),
        yaxis=dict(title="z*  (threshold)", **_AXIS_STYLE),
    )
    return fig


def make_turnover_fig(history: list, current: dict) -> go.Figure:
    fig = go.Figure()
    qs  = list(range(1, N_DECILES + 1))
    ng  = len(history)

    for ri, run in enumerate(history):
        op = _ghost_opacity(ri, ng)
        for N in N_LIST:
            tv = run.get("turnover", {}).get(str(N))
            if not tv:
                continue
            fig.add_trace(go.Scatter(
                x=qs, y=tv, mode="lines+markers",
                line=dict(color=N_COLORS[N], width=2.5, dash="dot"),
                marker=dict(size=5, color=N_COLORS[N]),
                opacity=op, showlegend=False, hoverinfo="skip",
            ))

    has_current = current and "turnover" in current
    if has_current:
        for N in N_LIST:
            tv = current["turnover"].get(str(N))
            if not tv:
                continue
            fig.add_trace(go.Scatter(
                x=qs, y=tv, mode="lines+markers",
                line=dict(color=N_COLORS[N], width=2.5),
                marker=dict(size=7, color=N_COLORS[N]),
                name=f"N = {N}",
            ))

    if not history and not has_current:
        fig.add_annotation(**_EMPTY_HINT)

    fig.update_layout(
        **_BASE_LAYOUT,
        title=dict(text="Turnover Rate by Performance Quintile",
                   font=dict(size=14, color="#222"), x=0.03),
        xaxis=dict(
            title="Performance Quintile  (1 = worst,  5 = best)",
            tickvals=qs, ticktext=[str(q) for q in qs], **_AXIS_STYLE,
        ),
        yaxis=dict(title="Mean Turnover Rate", tickformat=".0%",
                   range=[0, 1], **_AXIS_STYLE),
    )
    return fig


# ─── Layout helpers ───────────────────────────────────────────────────────────

_CARD = {
    "background": "#ffffff", "borderRadius": "8px",
    "boxShadow": "0 1px 3px rgba(0,0,0,0.08)",
    "padding": "14px 16px", "marginBottom": "10px",
}
_SECTION_LABEL = {
    "margin": "0 0 10px", "fontSize": "10px", "fontWeight": "700",
    "color": "#aaa", "textTransform": "uppercase", "letterSpacing": "0.08em",
}


def _slider_block(label: str, sid: str, lo: float, hi: float,
                   step: float, val: float):
    return html.Div([
        html.Div([
            html.Span(label, style={
                "fontSize": "13px", "fontWeight": "600", "color": "#333",
            }),
            html.Span(str(val), id=f"lbl-{sid}", style={
                "fontSize": "12px", "color": "#4C72B0", "fontWeight": "700",
                "background": "#eef2ff", "borderRadius": "4px", "padding": "1px 7px",
            }),
        ], style={
            "display": "flex", "justifyContent": "space-between",
            "alignItems": "center", "marginBottom": "3px",
        }),
        dcc.Slider(
            id=f"sl-{sid}", min=lo, max=hi, step=step, value=val,
            marks=None, tooltip={"always_visible": False}, updatemode="drag",
        ),
    ], style={"marginBottom": "13px"})


# ─── App ──────────────────────────────────────────────────────────────────────

app = dash.Dash(__name__, title="Threshold & Turnover Explorer")
server = app.server   # for deployment (gunicorn, etc.)

_INIT_T = make_threshold_fig([], {})
_INIT_V = make_turnover_fig([], {})

app.layout = html.Div([
    dcc.Store(id="store-history", data=[]),

    # ── Header ──────────────────────────────────────────────────────────
    html.Div([
        html.Img(
            src="/assets/board_vote.png",
            style={
                "height": "90px", "width": "160px", "borderRadius": "8px",
                "objectFit": "cover", "objectPosition": "center top",
                "boxShadow": "0 2px 8px rgba(0,0,0,0.22)",
            },
        ),
        html.Div([
            html.H2("Threshold & Turnover Explorer", style={
                "margin": 0, "fontSize": "24px", "fontWeight": "700", "color": "#111",
            }),
            html.Span(
                "Adjust parameters → click ▶ Run.  Old runs stay visible, shaded.",
                style={"fontSize": "13px", "color": "#999"},
            ),
        ]),
    ], style={
        "display": "flex", "alignItems": "center", "gap": "18px",
        "padding": "12px 24px", "background": "#fafafa",
        "borderBottom": "1px solid #e0e0e0",
    }),

    # ── Body ────────────────────────────────────────────────────────────
    html.Div([

        # ── Sidebar ─────────────────────────────────────────────────────
        html.Div([

            # Noise parameters
            html.Div([
                html.P("Noise", style=_SECTION_LABEL),
                _slider_block("σ_α  (sigma_alpha)", "sa",  0.10, 10.0, 0.01, 1.00),
                _slider_block("σ_y  (sigma_y)",     "sy",  0.10, 10.0, 0.01, 1.00),
                _slider_block("σ_z  (sigma_z)",     "sz",  0.10, 10.0, 0.01, 1.00),
            ], style=_CARD),

            # Cost parameters
            html.Div([
                html.P("Costs", style=_SECTION_LABEL),
                _slider_block("c_bar",            "cb", 0.0, 40.0, 0.01, 5.0),
                _slider_block("c_d  (c_dissent)", "cd", 0.0, 40.0, 0.01, 5.0),
            ], style=_CARD),

            # Model parameters
            html.Div([
                html.P("Model", style=_SECTION_LABEL),
                html.Div([
                    html.Span("k  (horizon)", style={
                        "fontSize": "13px", "fontWeight": "600", "color": "#333",
                    }),
                    dcc.RadioItems(
                        id="r-k", value=1,
                        options=[{"label": f" {v}", "value": v} for v in [1, 3, 5]],
                        inline=True,
                        labelStyle={"marginLeft": "10px", "fontSize": "13px"},
                        inputStyle={"marginRight": "3px"},
                    ),
                ], style={
                    "display": "flex", "alignItems": "center",
                    "justifyContent": "space-between", "marginBottom": "13px",
                }),
                _slider_block("τ  (exog. sep.)", "tau", 0.00, 0.50, 0.01, 0.00),
                _slider_block("β  (discount)",   "bt",  0.50, 1.00, 0.01, 0.90),
            ], style=_CARD),

            # Run / Clear buttons
            html.Button("▶  Run", id="btn-run", n_clicks=0, style={
                "width": "100%", "padding": "11px", "fontSize": "14px",
                "fontWeight": "700", "color": "#fff",
                "background": "linear-gradient(135deg, #1a1a2e 0%, #16213e 100%)",
                "border": "none", "borderRadius": "7px", "cursor": "pointer",
                "boxShadow": "0 2px 8px rgba(0,0,0,0.20)", "marginBottom": "8px",
            }),
            html.Button("✕  Clear History", id="btn-clr", n_clicks=0, style={
                "width": "100%", "padding": "8px", "fontSize": "12px",
                "color": "#aaa", "background": "transparent",
                "border": "1px solid #ddd", "borderRadius": "7px", "cursor": "pointer",
            }),

            html.Div(id="status-msg", style={
                "marginTop": "10px", "fontSize": "12px",
                "color": "#888", "textAlign": "center", "minHeight": "18px",
            }),

            # Color legend
            html.Div([
                html.P("Board size:", style={
                    "margin": "0 0 6px", "fontSize": "11px",
                    "color": "#bbb", "fontWeight": "600",
                }),
                html.Div([
                    html.Span(f"N = {N}", style={
                        "display": "inline-block",
                        "background": N_COLORS[N], "color": "#fff",
                        "borderRadius": "4px", "padding": "3px 9px",
                        "fontSize": "11px", "fontWeight": "700",
                        "marginRight": "5px", "marginBottom": "4px",
                    }) for N in N_LIST
                ]),
            ], style={"marginTop": "18px"}),

        ], style={
            "width": "278px", "minWidth": "278px",
            "padding": "14px", "overflowY": "auto",
            "background": "#f5f6fa", "borderRight": "1px solid #e0e0e0",
        }),

        # ── Plots ───────────────────────────────────────────────────────
        html.Div([
            dcc.Loading(type="circle", color="#1a1a2e", children=[
                dcc.Graph(
                    id="fig-thresh", figure=_INIT_T,
                    style={"height": "calc(50vh - 30px)"},
                    config={"displayModeBar": True,
                            "toImageButtonOptions": {"format": "svg", "filename": "thresholds"}},
                ),
                html.Hr(style={"margin": 0, "border": "none",
                               "borderTop": "1px solid #eee"}),
                dcc.Graph(
                    id="fig-turn", figure=_INIT_V,
                    style={"height": "calc(50vh - 30px)"},
                    config={"displayModeBar": True,
                            "toImageButtonOptions": {"format": "svg", "filename": "turnover"}},
                ),
            ]),
        ], style={
            "flex": 1, "overflow": "hidden",
            "display": "flex", "flexDirection": "column",
        }),

    ], style={
        "display": "flex", "height": "calc(100vh - 118px)", "overflow": "hidden",
    }),

], style={
    "fontFamily": "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
    "height": "100vh", "overflow": "hidden", "background": "#f5f6fa",
})


# ─── Callbacks ────────────────────────────────────────────────────────────────

@app.callback(
    Output("lbl-sa",  "children"),
    Output("lbl-sy",  "children"),
    Output("lbl-sz",  "children"),
    Output("lbl-cb",  "children"),
    Output("lbl-cd",  "children"),
    Output("lbl-tau", "children"),
    Output("lbl-bt",  "children"),
    Input("sl-sa",  "value"),
    Input("sl-sy",  "value"),
    Input("sl-sz",  "value"),
    Input("sl-cb",  "value"),
    Input("sl-cd",  "value"),
    Input("sl-tau", "value"),
    Input("sl-bt",  "value"),
)
def update_labels(sa, sy, sz, cb, cd, tau, bt):
    return (
        f"{sa:.2f}", f"{sy:.2f}", f"{sz:.2f}",
        f"{cb:.2f}", f"{cd:.2f}",
        f"{tau:.2f}", f"{bt:.2f}",
    )


@app.callback(
    Output("store-history", "data"),
    Output("fig-thresh",    "figure"),
    Output("fig-turn",      "figure"),
    Output("status-msg",    "children"),
    Input("btn-run", "n_clicks"),
    Input("btn-clr", "n_clicks"),
    State("store-history", "data"),
    State("sl-sa",  "value"),
    State("sl-sy",  "value"),
    State("sl-sz",  "value"),
    State("sl-cb",  "value"),
    State("sl-cd",  "value"),
    State("r-k",    "value"),
    State("sl-tau", "value"),
    State("sl-bt",  "value"),
    prevent_initial_call=True,
)
def on_action(run_n, clr_n, history, sa, sy, sz, cb, cd, k, tau, bt):
    ctx     = dash.callback_context
    trigger = ctx.triggered[0]["prop_id"].split(".")[0]
    history = history or []

    if trigger == "btn-clr":
        return [], make_threshold_fig([], {}), make_turnover_fig([], {}), "History cleared."

    try:
        # 1. Compute threshold curve for each N
        thresholds = {
            str(N): compute_threshold_curve(cd, cb, sy, sz, N, bt, k, tau)
            for N in N_LIST
        }

        # 2. Run quintile turnover simulation
        turnover = run_simulation(thresholds, tau, sa, sy, sz)

        current = {
            "params":     dict(sa=sa, sy=sy, sz=sz, cb=cb, cd=cd, k=k, tau=tau, bt=bt),
            "thresholds": thresholds,
            "turnover":   turnover,
        }

        fig_t  = make_threshold_fig(history, current)
        fig_v  = make_turnover_fig(history, current)
        n_prev = len(history)
        history = [current]   # keep only the last run as the single ghost

        shade_note = f"  ({n_prev} previous run{'s' if n_prev != 1 else ''} shaded)" if n_prev else ""
        return history, fig_t, fig_v, f"Done ✓{shade_note}"

    except Exception as exc:
        return history, dash.no_update, dash.no_update, f"Error: {exc}"


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8050))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"Starting — open http://127.0.0.1:{port} in your browser")
    print("Note: each Run takes ~5-15 s depending on parameters.")
    app.run(debug=False, host=host, port=port)
