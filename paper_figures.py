"""Paper figures 1-5 for the prism-adaptation study.

Every figure is a function, every condition is an argument, and the paper's
figure number is part of the output name.

Data lives in PA_excel/ and is written by the MATLAB side (REPEAT_MAIN,
REPEAT_VALIDATION).  Every sheet has the same layout:

    A1 empty | B1..  session numbers
    A2..     simulation numbers
    B2..     the Nsim x Nsession matrix

so ``pd.read_excel(..., index_col=0, header=0)`` yields a frame indexed by
simulation number whose columns are session numbers.  Because of that header row
and label column, all selection here is by label (``.loc``), never by position
(``.iloc``), which would be off by one.

Usage:
    python3 paper_figures.py               # every figure
    python3 paper_figures.py fig2 fig4     # a subset
    python3 paper_figures.py --stats       # one table with every statistical test
"""

import math
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Configuration -- the only place file names and levels are written down
# ---------------------------------------------------------------------------

DATA_DIR = Path("PA_excel")
OUT_DIR = Path("PA_figure")

# Sweep levels.  Blur 0 and risk divisor 1 are the unswept run, so they resolve
# to the plain const file on their own.
MD_LEVELS = [1, 2, 3, 4, 5, 6]
NOISE_LEVELS = [0, 5, 10, 20, 30, 40, 50]
BY_LEVELS = [1, 2, 5, 10, 20, 50, 100]

FIG1_FILE = "fig1_simulation.xlsx"  # written by fig1_simulation.m, outside the registry


def _main_file(sim_type, positions, blur=0, risk_divisor=1):
    """Name of a main-experiment workbook, mirroring data_files.m.

    positions is the initial-position regime: "hand" for a fixed target with a
    randomised hand, "target" for both randomised, "MD{k}" for a target pair at
    Manhattan distance k.  Conditions left at their default are omitted from the
    name, so the unswept run is just result_main_{sim_type}_{positions}.xlsx.
    """
    stem = f"result_main_{sim_type}_{positions}"
    if blur != 0:
        stem += f"_blur{blur}"
    if risk_divisor != 1:
        stem += f"_riskRangeby{risk_divisor}"
    return stem + ".xlsx"


# The two target regimes, as the pairs of workbooks the panels compare.
# "proposed" is the transfer learner, "naive" the from-scratch one.
CONDITIONS = {
    "const": {  # target fixed at (7,5), hand randomised
        "proposed": _main_file("transfer", "hand"),
        "naive": _main_file("naive", "hand"),
    },
    "rnd": {  # target and hand randomised per simulation
        "proposed": _main_file("transfer", "target"),
        "naive": _main_file("naive", "target"),
    },
}
MD_FILE = "result_main_transfer_MD{k}.xlsx"
VALID_FILE = "result_valid.xlsx"

# Session schedule, matching default_config.m: the prism goes on at
# exposureStart and comes off at removalStart, so it is worn during 141..170.
EXPOSURE_START = 141
REMOVAL_START = 171
BAND_X = (EXPOSURE_START - 0.5, REMOVAL_START - 0.5)  # 140.5 .. 170.5

VALID_T = 10  # trials per validation session; the x axis counts trials, not sessions

# Sessions the summary panels are read out at.  The noise sweep and the
# learning-rate sweep deliberately differ by one, as the figures in the paper do.
NOISE_SUMMARY_SESSION = 170
BY_SUMMARY_SESSIONS = (150, 171)
SCATTER_SESSION = 142  # one session after the prism went on

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

FONTSIZE = 30
FONTSIZE_LAMBDA = 23  # the lambda panels are half height, so they use a smaller face

PROPOSED = "31, 119, 180"  # blue
NAIVE = "200, 25, 35"  # red
TRUE_LAMBDA = "45, 104, 45"  # green: weight on the exposure target's policy
SHIFTED_LAMBDA = "234, 129, 6"  # orange: weight on the baseline target's policy
SUMMARY = "0, 0, 0"  # black: the sweep summaries
BAND_FILL = "rgba(0, 0, 0, 0.07)"

MARGIN = dict(t=10, b=10, l=10, r=10)

# Grid styling is per panel, not global: some panels show dotted grey gridlines
# and some switch the grid off entirely.  Pass DOT_GRID plus showgrid=False to
# get the dotted style on the minor ticks only.
DOT_GRID = dict(gridcolor="gray", griddash="dot")

# The line width fig_med draws the central curve at.  Which curves are thickened
# differs per panel, so panels that leave it unset get plotly's default of 2.
THICK = 4


def axis(**overrides):
    """Base axis style shared by every panel: black frame, outside ticks."""
    style = dict(
        showline=True,
        linewidth=1,
        mirror=True,
        linecolor="Black",
        ticks="outside",
        tickfont=dict(size=FONTSIZE),
    )
    style.update(overrides)
    return style


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def fig_med(fig, x, lower, med, upper, color, showlegend=False, name="learner",
            row=1, col=1, width=None):
    """Draw a central line with a shaded band between lower and upper.

    The band is built out of three traces because plotly's ``fill='tonexty'``
    fills against the previous trace: invisible lower bound, the line itself,
    then the invisible upper bound.
    """
    line = dict(color="rgb(" + color + ")")
    if width is not None:
        line["width"] = width

    fig.add_trace(go.Scatter(
        name="lower", x=x, y=lower,
        line=dict(width=0), mode="lines", showlegend=False,
    ), row=row, col=col)
    fig.add_trace(go.Scatter(
        name=name, x=x, y=med,
        line=line, fillcolor="rgba(" + color + ",0.3)",
        mode="lines", fill="tonexty", showlegend=showlegend,
    ), row=row, col=col)
    fig.add_trace(go.Scatter(
        name="upper", x=x, y=upper,
        line=dict(width=0), fillcolor="rgba(" + color + ",0.3)",
        mode="lines", fill="tonexty", showlegend=False,
    ), row=row, col=col)


def exposure_band(fig, y0, y1, row=1, col=1):
    """Shade the sessions the prism was worn (140.5 .. 170.5)."""
    x = np.linspace(BAND_X[0], BAND_X[1], 31)
    fig.add_trace(go.Scatter(
        x=np.append(x, x[::-1]),
        y=np.append([y1] * len(x), [y0] * len(x)),
        mode="none", fill="toself", fillcolor=BAND_FILL, showlegend=False,
    ), row=row, col=col)


def vline(fig, x, y0, y1, row=1, col=1):
    """Dashed vertical marker drawn in data coordinates."""
    fig.add_trace(go.Scatter(
        x=[x, x], y=[y0, y1], mode="lines",
        line=dict(dash="dash", color="Grey"), showlegend=False,
    ), row=row, col=col)


def decompose_number(num):
    """Split num into its leading digit and its decimal exponent."""
    if num == 0:
        return 0, 0
    exponent = math.floor(math.log10(abs(num)))
    coefficient = math.floor(num / (10 ** exponent))
    return coefficient, exponent


def to_latex_exponent(formatted):
    """Turn an "…e-05" tick label into the LaTeX 10^-5 plotly renders."""
    if formatted == "":
        return formatted
    exponent = int(formatted.split("e")[1])
    return rf"$\mathsf {{\large 10^{{{exponent}}}}}$"


def make_logtick(low, high, fmt="{}"):
    """Tick positions for a log axis: 1..9 per decade, labelled only at 10^n.

    Passing fmt='power' labels the decades as LaTeX powers of ten.
    """
    latex = False
    base = np.arange(1, 10)
    val, text = [], []
    coef_min, exp_min = decompose_number(low)
    coef_max, exp_max = decompose_number(high)
    if fmt == "power":
        latex = True
        fmt = "{:e}"
    for i in range(exp_min, exp_max + 1):
        if i == exp_min:
            val.extend([float(Decimal(str(n)) * Decimal(str(10 ** i))) for n in base[coef_min - 1:]])
            if coef_min == 1:
                text.append(fmt.format(Decimal(str(10 ** i))))
                text.extend([""] * 8)
            else:
                text.extend([""] * (10 - coef_min))
        elif i == exp_max:
            val.extend([float(Decimal(str(n)) * Decimal(str(10 ** i))) for n in base[:coef_max]])
            text.append(fmt.format(Decimal(str(10 ** i))))
            if coef_max != 1:
                text.extend([""] * (coef_max - 1))
        else:
            val.extend([float(Decimal(str(n)) * Decimal(str(10 ** i))) for n in base])
            text.append(fmt.format(Decimal(str(10 ** i))))
            text.extend([""] * 8)
    if latex:
        text = list(map(to_latex_exponent, text))
    return val, text


def read_sheets(filename, sheets):
    """Read sheets from DATA_DIR/filename as {name: DataFrame}.

    Index is the simulation number, columns are session numbers.
    """
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"{path} not found (put the MATLAB output in {DATA_DIR}/)")
    frames = pd.read_excel(path, sheet_name=list(sheets), index_col=0, header=0)
    return {name: frames[name] for name in sheets}


def read_sheet(filename, sheet):
    """One sheet of a workbook, as a simulation x session frame."""
    return read_sheets(filename, [sheet])[sheet]


def mean_band(df):
    """(mean - sd, mean, mean + sd) across simulations, indexed by session."""
    m, s = df.mean(), df.std()
    return m - s, m, m + s


def quantile_band(df):
    """(q25, q50, q75) across simulations, indexed by session."""
    q = df.quantile([0.25, 0.5, 0.75])
    return q.loc[0.25], q.loc[0.50], q.loc[0.75]


def sessions(df):
    """The session labels of a sheet, as a plain array."""
    return np.asarray(df.columns)


def save(fig, name):
    """Write one figure to OUT_DIR as SVG, creating the directory if needed."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    pio.write_image(fig, str(path), engine="kaleido")
    print(f"  wrote {path}")


def _theoretical(x, a=0.0389, b=-3.0):
    """The analytic error decay the validation curves are compared against."""
    return np.exp(-(a * np.asarray(x, dtype=float) + b))


# ---------------------------------------------------------------------------
# Fig 1 -- analytic vs simulated error decay
# ---------------------------------------------------------------------------


def _fig1_base():
    """The traces both Fig1 panels share, before the axis type is chosen.

    Ten runs of each update rule (err1 dark red, err2 dark blue) with the two
    analytic curves (th1 red, th2 blue) drawn thick on top.  fig1() then saves
    the same figure twice, once log-log and once semi-log.
    """
    df = read_sheets(FIG1_FILE, ["err1", "err2", "th1", "th2"])
    x = sessions(df["err1"])  # step index 1..10000

    fig = make_subplots(rows=1, cols=1)
    for sim in df["err1"].index:
        fig.add_trace(go.Scatter(x=x, y=df["err1"].loc[sim], mode="lines",
                                 line=dict(color="rgb(0.5,0,0)")))
    for sim in df["err2"].index:
        fig.add_trace(go.Scatter(x=sessions(df["err2"]), y=df["err2"].loc[sim], mode="lines",
                                 line=dict(color="rgb(0,0,0.3)")))
    fig.add_trace(go.Scatter(x=sessions(df["th1"]), y=df["th1"].iloc[0], mode="lines",
                             line=dict(color="Red", width=5)))
    fig.add_trace(go.Scatter(x=sessions(df["th2"]), y=df["th2"].iloc[0], mode="lines",
                             line=dict(color="Blue", width=5)))

    fig.update_layout(go.Layout(
        height=500, width=600, plot_bgcolor="white",
        font=dict(size=FONTSIZE, color="Black"), showlegend=False,
        margin=MARGIN, xaxis=dict(tickangle=0),
    ))

    minor_y, _ = make_logtick(10 ** -10, 1, "power")
    fig.update_yaxes(**axis(
        **DOT_GRID, showgrid=False, mirror=False,
        type="log", range=(-10, 0),
        tickvals=[1, 10 ** -2, 10 ** -4, 10 ** -6, 10 ** -8, 10 ** -10],
        exponentformat="power", zeroline=True, zerolinecolor="teal", zerolinewidth=0.5,
        minor=dict(ticks="outside", tickvals=minor_y, ticklen=5, tickcolor="Black"),
    ))
    return fig


def fig1():
    """Fig1: the error curves against their analytic prediction, linear and log x."""
    print("fig1")
    fig = _fig1_base()
    fig.update_xaxes(**axis(**DOT_GRID, showgrid=False,
                            title="step", type="linear", range=(1, 100)))
    save(fig, "fig1a_semilog.svg")

    # Same traces, log x.
    minor_x, _ = make_logtick(1, 10000, "power")
    fig.update_xaxes(
        type="log", range=(0, 4), tickvals=[1, 10, 100, 1000, 10000],
        exponentformat="none",
        minor=dict(ticks="outside", tickvals=minor_x, ticklen=5, tickcolor="Black"),
    )
    save(fig, "fig1b_loglog.svg")


# ---------------------------------------------------------------------------
# Fig 2 / Fig 3 panels -- identical apart from the condition they read
# ---------------------------------------------------------------------------


def _panel_x_broken(cond):
    """Hand x position relative to the target, mean +/- sd, broken x axis.

    Sessions 1-18 and 130-200 are drawn side by side: nothing happens in
    between, and the interesting part is the shift and its after-effect.
    """
    df = read_sheet(CONDITIONS[cond]["proposed"], "relPos")
    lower, mean, upper = mean_band(df)

    early, late = slice(1, 18), slice(130, 200)
    x1 = np.asarray(mean.loc[early].index)
    x2 = np.asarray(mean.loc[late].index)

    fig = make_subplots(rows=1, cols=2, shared_yaxes=True,
                        column_widths=[0.18, 0.7], horizontal_spacing=0.02)

    vline(fig, BAND_X[0], -4, 4, row=1, col=2)
    vline(fig, BAND_X[1], -4, 4, row=1, col=2)
    fig_med(fig, x1, lower.loc[early], mean.loc[early], upper.loc[early],
            PROPOSED, col=1, width=THICK)
    fig_med(fig, x2, lower.loc[late], mean.loc[late], upper.loc[late],
            PROPOSED, col=2, width=THICK)
    exposure_band(fig, -4, 4, row=1, col=2)

    fig.update_layout(go.Layout(
        height=400, width=1750, plot_bgcolor="white",
        font=dict(size=FONTSIZE, color="Black"), margin=MARGIN, showlegend=False,
    ))
    # The narrow left panel keeps its dotted grid; the wide right one drops it.
    fig.update_xaxes(**axis(**DOT_GRID, tickvals=[0, 1]), row=1, col=1)
    fig.update_xaxes(**axis(**DOT_GRID, showgrid=False,
                            range=(130, 200), tickvals=[130, 140, 170, 200]), row=1, col=2)
    fig.update_yaxes(**axis(
        **DOT_GRID, showgrid=False, mirror=False,
        range=(-4, 4), tickvals=[-4, -2, 0, 2, 4],
        zeroline=True, zerolinecolor="gray", zerolinewidth=0.5,
    ), row=1, col=1)
    fig.update_yaxes(
        mirror=True, showgrid=False, gridcolor="gray", griddash="dot",
        zeroline=True, zerolinecolor="teal", zerolinewidth=0.5,
        row=1, col=2,
    )
    return fig


def _panel_lambda(cond):
    """Mixing weight on the exposure-target policy (green) and the baseline one (orange)."""
    df = read_sheets(CONDITIONS[cond]["proposed"], ["truelambdaList", "shiftedlambdaList"])
    x = sessions(df["truelambdaList"])

    fig = make_subplots(rows=1, cols=1)
    fig_med(fig, x, *quantile_band(df["truelambdaList"]), color=TRUE_LAMBDA, width=THICK)
    fig_med(fig, x, *quantile_band(df["shiftedlambdaList"]), color=SHIFTED_LAMBDA, width=THICK)
    exposure_band(fig, 0, 12)

    fig.update_layout(go.Layout(
        height=200, width=600, plot_bgcolor="white",
        font=dict(size=FONTSIZE_LAMBDA, color="Black"), margin=MARGIN, showlegend=False,
    ))
    fig.update_xaxes(**axis(
        **DOT_GRID, range=[0, 200], tickvals=[0, 140, 170, 200],
        ticktext=["0", "140", "170", "200"], tickangle=0,
        tickfont=dict(size=FONTSIZE_LAMBDA),
    ))
    # No grid keys here on purpose: this axis keeps plotly's default horizontal grid.
    fig.update_yaxes(**axis(
        mirror=False, range=[0, 1], tickfont=dict(size=FONTSIZE_LAMBDA),
        zeroline=True, zerolinecolor="teal", zerolinewidth=0.5,
    ))
    return fig


def _duration_axes(fig, row):
    """Axes shared by the two duration panels: sessions 1-200, ticked at the
    prism on/off boundaries, against 0-100 trials (the cfg.T cap)."""
    fig.update_xaxes(**axis(
        **DOT_GRID, showgrid=False,
        range=[1, 200], tickvals=[1, 140, 170, 200],
        ticktext=["1", "140", "170", "200"],
        zeroline=True, zerolinecolor="teal", zerolinewidth=0.25,
    ), row=row, col=1)
    fig.update_yaxes(**axis(
        **DOT_GRID, showgrid=False,
        range=[0, 101], zeroline=True, zerolinecolor="teal", zerolinewidth=0.25,
    ), row=row, col=1)


def _panel_duration(cond):
    """Trials needed per session: proposed on top, naive below, shared x axis."""
    proposed = read_sheet(CONDITIONS[cond]["proposed"], "t")
    naive = read_sheet(CONDITIONS[cond]["naive"], "t")
    x = sessions(proposed)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.2)
    fig_med(fig, x, *quantile_band(proposed), color=PROPOSED, row=1, width=THICK)
    fig_med(fig, x, *quantile_band(naive), color=NAIVE, row=2, width=THICK)
    exposure_band(fig, -5, 120, row=1, col=1)
    exposure_band(fig, -5, 120, row=2, col=1)

    # Paper coordinates, so one line spans the gap between the subplots and the
    # top margin.  y1 > 1 is deliberate: inside the axes the white plot
    # background hides the line, so only the margins show it.
    span = 200 - 1
    for session in BAND_X:
        xp = (session - 1) / span
        fig.add_shape(type="line", x0=xp, x1=xp, y0=0, y1=1.5,
                      line=dict(color="Grey", dash="dash"), xref="paper", yref="paper")

    fig.update_layout(go.Layout(
        height=400, width=700, plot_bgcolor="white",
        font=dict(size=FONTSIZE, color="Black"), showlegend=False,
        margin=dict(t=80, b=10, l=10, r=10),
    ))
    _duration_axes(fig, 1)
    _duration_axes(fig, 2)
    return fig


def _panel_duration_scatter(cond, session=SCATTER_SESSION):
    """Per-simulation duration, naive against proposed, one session.

    Session 142 is the first one after a full session of prism exposure.
    """
    proposed = read_sheet(CONDITIONS[cond]["proposed"], "t")
    naive = read_sheet(CONDITIONS[cond]["naive"], "t")

    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(x=np.arange(0, 101), y=np.arange(0, 101),
                             line=dict(color="black", dash="dash")))
    fig.add_trace(go.Scatter(x=naive.loc[:, session], y=proposed.loc[:, session],
                             mode="markers", marker=dict(color="Black", size=10)))

    fig.update_layout(go.Layout(
        height=500, width=500, plot_bgcolor="white",
        font=dict(size=FONTSIZE, color="Black"), showlegend=False, margin=MARGIN,
    ))
    fig.update_xaxes(**axis(**DOT_GRID, showgrid=False,
                            range=[0, 101], tickangle=0, zeroline=False))
    fig.update_yaxes(**axis(**DOT_GRID, showgrid=False, range=[0, 101], zeroline=False))
    return fig


def _panel_md_duration(session=NOISE_SUMMARY_SESSION):
    """Duration against the Manhattan distance between baseline and exposure target."""
    lower, med, upper = [], [], []
    for k in MD_LEVELS:
        q25, q50, q75 = quantile_band(read_sheet(MD_FILE.format(k=k), "t"))
        lower.append(q25.loc[session])
        med.append(q50.loc[session])
        upper.append(q75.loc[session])

    fig = make_subplots(rows=1, cols=1)
    fig_med(fig, MD_LEVELS, lower, med, upper, SUMMARY)
    fig.update_layout(go.Layout(
        height=500, width=800, plot_bgcolor="white",
        font=dict(size=FONTSIZE, color="Black"), margin=MARGIN, showlegend=False,
    ))
    fig.update_xaxes(**axis(**DOT_GRID, tickvals=MD_LEVELS,
                            ticktext=[str(k) for k in MD_LEVELS]))
    fig.update_yaxes(**axis(**DOT_GRID, showgrid=False, range=(0, 100),
                            zeroline=True, zerolinecolor="gray", zerolinewidth=0.5))
    return fig


def _condition_figures(cond, prefix):
    """Save the four panels a target regime gets: a, b, c, d under `prefix`.

    Fig2 and Fig3 are the same four figures for the two regimes, so `cond`
    selects the workbooks ("const" or "rnd") and `prefix` the output names.
    """
    save(_panel_x_broken(cond), f"{prefix}a_x_broken.svg")
    save(_panel_lambda(cond), f"{prefix}b_lambda.svg")
    save(_panel_duration(cond), f"{prefix}c_duration.svg")
    save(_panel_duration_scatter(cond), f"{prefix}d_duration_scatter.svg")


def fig2():
    """Fig2: fixed target -- reaching error, mixing weights and duration."""
    print("fig2 (const)")
    _condition_figures("const", "fig2")


def fig3():
    """Fig3: random target -- the Fig2 panels plus the random-shift duration sweep."""
    print("fig3 (rnd)")
    _condition_figures("rnd", "fig3")
    save(_panel_md_duration(), "fig3e_md_duration.svg")


# ---------------------------------------------------------------------------
# Fig 4 -- validation experiment and the main run's generalisation error
# ---------------------------------------------------------------------------


def _valid_layout():
    """Layout shared by the validation panels: single plot, no legend."""
    return go.Layout(
        height=500, width=700, plot_bgcolor="white",
        font=dict(size=FONTSIZE, color="Black"), margin=MARGIN,
        xaxis=dict(tickangle=0), showlegend=False,
    )


def _valid_x(df):
    """Validation sheets are indexed by session; the x axis counts trials."""
    return sessions(df) * VALID_T


def _panel_valid_lambda_error():
    """Validation: how far the mixing weights are from the correct policy.

    Median and quartiles of ||lambda - lambda*||^2 against the step count, on a
    log y axis, with the analytic decay dashed over it.  The weights converge
    far faster than C does, hence the 10^-50 axis range.
    """
    df = read_sheet(VALID_FILE, "lambda_Gerror")
    x = _valid_x(df)

    fig = make_subplots(rows=1, cols=1)
    fig_med(fig, x, *quantile_band(df), color=PROPOSED)
    x_th = np.arange(0, 300) * 10
    fig.add_trace(go.Scatter(name="theoretical", x=x_th, y=_theoretical(x_th), mode="lines",
                             line=dict(color="rgb(25, 100, 95)", dash="dash")))

    fig.update_layout(_valid_layout())
    fig.update_xaxes(**axis(**DOT_GRID, showgrid=False,
                            title="step", type="linear", range=(0, 4000)))
    fig.update_yaxes(**axis(
        **DOT_GRID, showgrid=False,
        title=r"$\large || \lambda -\lambda^*||^2$", type="log",
        tickvals=[10 ** -50, 10 ** -40, 10 ** -30, 10 ** -20, 10 ** -10, 1],
        range=(-50, 0), exponentformat="power", mirror=False,
        zeroline=True, zerolinecolor="teal", zerolinewidth=0.5,
    ))
    return fig


def _panel_valid_c_loglog():
    """Validation: the two learners' policy error, log-log.

    Median and quartiles of ||C - C*||^2 for the transfer and the naive learner,
    which the validation experiment updates along the same trajectory, so the
    two curves are directly comparable.  Log-log shows the naive learner's
    power-law decay as a straight line.
    """
    df = read_sheets(VALID_FILE, ["transfer", "naive"])
    x = _valid_x(df["transfer"])

    fig = make_subplots(rows=1, cols=1)
    fig_med(fig, x, *quantile_band(df["transfer"]), color=PROPOSED,
            name="transfer learner", width=THICK)
    fig_med(fig, x, *quantile_band(df["naive"]), color=NAIVE,
            name="naive learner", width=THICK)

    x_tickvals, x_ticktext = make_logtick(10, 10000)
    y_minor, _ = make_logtick(10 ** -7, 1, "power")
    fig.update_layout(_valid_layout())
    fig.update_xaxes(**axis(
        **DOT_GRID, showgrid=False,
        title="step", type="log", tickvals=x_tickvals, ticktext=x_ticktext,
        range=(0, 4), exponentformat="power",
    ))
    fig.update_yaxes(**axis(
        **DOT_GRID, showgrid=False,
        title=r"$|| C-C^*||^2$", type="log",
        tickvals=[10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 1],
        range=(-5, 0), exponentformat="power", mirror=False,
        zeroline=True, zerolinecolor="teal", zerolinewidth=0.5,
        minor=dict(ticks="outside", tickvals=y_minor, ticklen=5, tickcolor="Black"),
    ))
    return fig


def _panel_valid_c_semilog():
    """Validation: the transfer learner's policy error, semi-log.

    The same ||C - C*||^2 as the log-log panel but for the transfer learner
    alone and over the first 600 steps, where its decay is exponential and so
    falls on a straight line here.  The analytic curve is dashed over it.
    """
    df = read_sheet(VALID_FILE, "transfer")
    x = _valid_x(df)

    fig = make_subplots(rows=1, cols=1)
    fig_med(fig, x, *quantile_band(df), color=PROPOSED, name="transfer learner", width=THICK)
    x_th = np.arange(1.2, 4, 0.1) * 100
    fig.add_trace(go.Scatter(name="theoretical", x=x_th, y=_theoretical(x_th), mode="lines",
                             line=dict(color="rgb(25, 100, 35)", dash="dash", width=THICK)))

    y_minor, _ = make_logtick(10 ** -7, 0.1, "power")
    fig.update_layout(_valid_layout())
    fig.update_xaxes(**axis(**DOT_GRID, showgrid=False,
                            title="step", type="linear", range=(0, 600)))
    fig.update_yaxes(**axis(
        **DOT_GRID, showgrid=False,
        title=r"$\large || C-C^*||^2$", type="log",
        tickvals=[10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 1],
        range=(-5, -0.8), exponentformat="power", mirror=False,
        zeroline=True, zerolinecolor="teal", zerolinewidth=0.5,
        minor=dict(ticks="outside", tickvals=y_minor, ticklen=5, tickcolor="Black"),
    ))
    return fig


def _panel_main_error(sheet, title, y_range, tickvals, minor_low, x_range=True):
    """Generalisation error of the main run, over the exposure sessions.

    Reads the C_Gerror / lambda_Gerror sheets REPEAT_MAIN writes.
    """
    df = read_sheet(CONDITIONS["const"]["proposed"], sheet)
    x = sessions(df)

    fig = make_subplots(rows=1, cols=1)
    fig_med(fig, x, *quantile_band(df), color=PROPOSED)

    minor, _ = make_logtick(minor_low, 1)
    fig.update_layout(_valid_layout())
    # The C panel pins the x range to the exposure sessions; the lambda panel
    # leaves it to plotly.
    fig.update_xaxes(**axis(**DOT_GRID, showgrid=False, title="step", type="linear",
                            **(dict(range=(x[0], x[-1])) if x_range else {})))
    fig.update_yaxes(**axis(
        **DOT_GRID, showgrid=False,
        title=title, type="log", tickvals=tickvals, range=y_range,
        exponentformat="power", mirror=False,
        zeroline=True, zerolinecolor="teal", zerolinewidth=0.5,
        minor=dict(ticks="outside", tickvals=minor, ticklen=5, tickcolor="Black"),
    ))
    return fig


def fig4():
    """Fig4: validation learning curves plus the main run's generalisation error."""
    print("fig4")
    save(_panel_valid_lambda_error(), "fig4a_valid_lambda_error.svg")
    save(_panel_valid_c_loglog(), "fig4b_valid_C_loglog.svg")
    save(_panel_valid_c_semilog(), "fig4c_valid_C_semilog.svg")
    save(_panel_main_error("C_Gerror", r"$|| C  -C^*||^2$", (-3, -1),
                           [10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 1],
                           10 ** -5),
         "fig4d_main_C_error.svg")
    save(_panel_main_error("lambda_Gerror", r"$|| \lambda  -\lambda^*||^2$", (-8, -1),
                           [10 ** -n for n in range(8, -1, -1)],
                           10 ** -8, x_range=False),
         "fig4e_main_lambda_error.svg")


# ---------------------------------------------------------------------------
# Fig 5 -- sensory noise and learning-rate sweeps
# ---------------------------------------------------------------------------


def _noised_file(level):
    """Workbook of one blur level.  Level 0 is unblurred, i.e. the const run."""
    return _main_file("transfer", "hand", blur=level)


def _by_file(level):
    """Workbook of one risk-width level.  Level 1 divides by nothing, i.e. const."""
    return _main_file("transfer", "hand", risk_divisor=level)


def _panel_noised_lambda(levels=(0, 10, 50)):
    """One row per noise level, both mixing weights in each.

    Every curve keeps plotly's default width here -- unlike the single-panel
    lambda figure, this one never raised any of them to THICK.
    """
    fig = make_subplots(rows=len(levels), cols=1, shared_xaxes=True, vertical_spacing=0.09)
    for row, level in enumerate(levels, start=1):
        df = read_sheets(_noised_file(level), ["truelambdaList", "shiftedlambdaList"])
        x = sessions(df["truelambdaList"])
        fig_med(fig, x, *quantile_band(df["truelambdaList"]), color=TRUE_LAMBDA, row=row)
        fig_med(fig, x, *quantile_band(df["shiftedlambdaList"]), color=SHIFTED_LAMBDA, row=row)

    fig.update_layout(go.Layout(
        height=400, width=600, plot_bgcolor="white",
        font=dict(size=FONTSIZE_LAMBDA, color="Black"), showlegend=False,
        margin=dict(t=1, b=1, l=1, r=1),
    ))
    fig.update_xaxes(**axis(
        **DOT_GRID, showgrid=False,
        range=[1, 200], tickvals=[1, 140, 170, 200],
        ticktext=["1", "140", "170", "200"], tickfont=dict(size=FONTSIZE_LAMBDA),
        zeroline=True, zerolinecolor="teal", zerolinewidth=0.25,
    ))
    fig.update_yaxes(**axis(
        **DOT_GRID, showgrid=False,
        range=[0, 1], tickfont=dict(size=FONTSIZE_LAMBDA),
        zeroline=True, zerolinecolor="teal", zerolinewidth=0.25,
    ))
    return fig


def _panel_noise_summary(session=NOISE_SUMMARY_SESSION):
    """Weight on the exposure-target policy at one session, against noise level."""
    lower, med, upper = [], [], []
    for level in NOISE_LEVELS:
        q25, q50, q75 = quantile_band(read_sheet(_noised_file(level), "truelambdaList"))
        lower.append(q25.loc[session])
        med.append(q50.loc[session])
        upper.append(q75.loc[session])

    fig = make_subplots(rows=1, cols=1)
    fig_med(fig, NOISE_LEVELS, lower, med, upper, SUMMARY)
    fig.update_layout(go.Layout(
        height=400, width=600, plot_bgcolor="white",
        font=dict(size=FONTSIZE, color="Black"), margin=MARGIN, showlegend=False,
    ))
    fig.update_xaxes(**axis(**DOT_GRID, tickvals=NOISE_LEVELS,
                            ticktext=[str(k) for k in NOISE_LEVELS]))
    fig.update_yaxes(**axis(**DOT_GRID, showgrid=False, range=(0, 1),
                            zeroline=True, zerolinecolor="gray", zerolinewidth=0.5))
    return fig


def _panel_by_summary(session):
    """Relative hand x at one session, against the learning rate 1/k (log axis)."""
    x, lower, med, upper = [], [], [], []
    for level in BY_LEVELS:
        lo, m, up = mean_band(read_sheet(_by_file(level), "relPos"))
        x.append(1.0 / level)
        lower.append(lo.loc[session])
        med.append(m.loc[session])
        upper.append(up.loc[session])

    fig = make_subplots(rows=1, cols=1)
    fig_med(fig, x, lower, med, upper, SUMMARY)
    fig.update_layout(go.Layout(
        height=500, width=800, plot_bgcolor="white",
        font=dict(size=FONTSIZE, color="Black"), margin=MARGIN, showlegend=False,
    ))
    fig.update_xaxes(**axis(
        **DOT_GRID, type="log", tickvals=[1.0 / k for k in reversed(BY_LEVELS)],
        ticktext=[f"1/{k}" for k in reversed(BY_LEVELS)],
    ))
    fig.update_yaxes(**axis(**DOT_GRID, showgrid=False,
                            zeroline=True, zerolinecolor="gray", zerolinewidth=0.5))
    return fig


def fig5():
    """Fig5: robustness to sensory noise and to the learning rate."""
    print("fig5")
    save(_panel_noised_lambda(), "fig5a_noised_lambda.svg")
    save(_panel_noise_summary(), f"fig5b_noised_lambda_at{NOISE_SUMMARY_SESSION}.svg")
    for letter, session in zip("cd", BY_SUMMARY_SESSIONS):
        save(_panel_by_summary(session), f"fig5{letter}_by_x_at{session}.svg")


# ---------------------------------------------------------------------------
# Statistics (printed, not plotted)
# ---------------------------------------------------------------------------


# Sessions the duration comparison is read out at: one session after the prism
# goes on, and one after it comes off.
DURATION_TEST_SESSIONS = (EXPOSURE_START + 1, REMOVAL_START + 1)

# Which figure each test belongs to, so one table can carry all of them.
DURATION_TEST_FIGURES = {"const": "Fig2", "rnd": "Fig3"}

STAT_COLUMNS = ("figure", "quantity", "group A", "group B", "session",
                "U", "p", "r")


def _mannwhitney(a, b):
    """Two-sided Mann-Whitney U, with the rank-biserial effect size.

    U is the statistic for the first sample, and r = 2U/(n1 n2) - 1 is the
    matching rank-biserial correlation: it runs from -1 to +1 and is positive
    when the first sample tends to hold the larger values.  No multiple-
    comparison correction is applied -- p is the raw two-sided value.
    """
    from scipy import stats

    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return u, p, 2 * u / (a.size * b.size) - 1


def _stat_tests():
    """Every test reported in the paper, as one row per comparison.

    Yields (figure, quantity, group A label, group B label, session, sample A,
    sample B).  Fig2 and Fig3 compare the two learners at a fixed and at a
    random target; Fig5 compares the baseline learning rate 1/1 against every
    other rate in the sweep.
    """
    for cond, figure in DURATION_TEST_FIGURES.items():
        proposed = read_sheet(CONDITIONS[cond]["proposed"], "t")
        naive = read_sheet(CONDITIONS[cond]["naive"], "t")
        for session in DURATION_TEST_SESSIONS:
            yield (figure, f"duration ({cond})", "proposed", "naive", session,
                   proposed.loc[:, session], naive.loc[:, session])

    base = read_sheet(_by_file(BY_LEVELS[0]), "relPos")
    for level in BY_LEVELS[1:]:
        df = read_sheet(_by_file(level), "relPos")
        for session in BY_SUMMARY_SESSIONS:
            yield ("Fig5", "x", f"1/{BY_LEVELS[0]}", f"1/{level}", session,
                   base.loc[:, session], df.loc[:, session])


def print_stats():
    """Print every Mann-Whitney U test behind Fig2, Fig3 and Fig5 as one table."""
    rows, sizes = [], set()
    for figure, quantity, name_a, name_b, session, a, b in _stat_tests():
        u, p, r = _mannwhitney(a, b)
        rows.append((figure, quantity, name_a, name_b, str(session),
                     f"{u:.1f}", f"{p:.3e}", f"{r:+.3f}"))
        sizes.update((len(a), len(b)))

    widths = [max(len(str(cell)) for cell in column)
              for column in zip(STAT_COLUMNS, *rows)]
    line = "  ".join("{:<%d}" % w for w in widths)
    print(line.format(*STAT_COLUMNS))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(line.format(*row))
    n = str(sizes.pop()) if len(sizes) == 1 else "/".join(str(s) for s in sorted(sizes))
    print(f"\nMann-Whitney U, two-sided, n={n} simulations per group, "
          "no correction for multiple comparisons.\n"
          "U is the statistic for group A; r is the rank-biserial correlation "
          "(positive when group A is larger).")


# ---------------------------------------------------------------------------

FIGURES = {"fig1": fig1, "fig2": fig2, "fig3": fig3, "fig4": fig4, "fig5": fig5}


def main(argv):
    """Draw the figures named on the command line, or every figure by default.

    With --stats no figure is drawn; the statistical tests are printed instead.
    Returns the process exit status.
    """
    if "--stats" in argv:
        print_stats()
        return 0

    names = [a for a in argv if not a.startswith("-")] or list(FIGURES)
    unknown = [n for n in names if n not in FIGURES]
    if unknown:
        print(f"unknown figure(s): {', '.join(unknown)}", file=sys.stderr)
        print(f"available: {', '.join(FIGURES)}", file=sys.stderr)
        return 1

    for name in names:
        FIGURES[name]()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
