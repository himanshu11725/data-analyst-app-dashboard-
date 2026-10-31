"""Insights engine: works out what kind of data this is (sales, student marks, finance),
then builds KPI cards, plain-English findings and interactive Plotly charts.

Pure pandas + plotly, no Streamlit code, so it can be tested on its own.
Colour rules: one blue for single-series charts, light-to-dark blue for ordered bands,
and blue + orange only when two series must be told apart.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

BLUE, ORANGE = "#2a78d6", "#eb6834"
BLUE_BANDS = ["#bcd6f4", "#7fb0e8", "#2a78d6", "#1b4f93"]  # light -> dark, one hue

DOMAINS = ["sales", "student", "finance", "general"]

DOMAIN_WORDS = {
    "sales": ["sale", "revenue", "order", "customer", "product", "qty", "quantity", "price",
              "invoice", "region", "discount", "profit", "item", "units"],
    "student": ["student", "marks", "mark", "score", "subject", "grade", "roll", "class", "exam",
                "math", "science", "english", "hindi", "physics", "chemistry", "percentage",
                "attendance", "result"],
    "finance": ["expense", "income", "debit", "credit", "balance", "transaction", "budget",
                "account", "payment", "salary", "tax", "deposit", "withdraw", "narration"],
}

ID_WORDS = re.compile(r"(^|_)(id|no|number|roll|sr|serial|index|code|phone|mobile|pin|pincode|zip)($|_)")
NON_SCORE = re.compile(r"attend|age|year|class|roll|rank|total|percent|average|avg|grade|sem")
MEASURE_PRIORITY = ["revenue", "sales", "amount", "total", "profit", "income", "expense", "price",
                    "value", "marks", "score", "balance"]
GROUP_HINTS = ["product", "category", "region", "city", "state", "channel", "segment", "item",
               "customer", "department", "payment", "status", "type"]


# --------------------------------------------------------------------------- helpers
def fmt(x) -> str:
    """Readable number: 1,234 / 12.5 / 1.2M style for big values."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    ax = abs(x)
    if ax >= 1e9:
        return f"{x / 1e9:,.2f}B"
    if ax >= 1e6:
        return f"{x / 1e6:,.2f}M"
    if ax >= 1000:
        return f"{x:,.0f}"
    if float(x).is_integer():
        return f"{x:,.0f}"
    return f"{x:,.2f}"


def _num(df: pd.DataFrame, c: str) -> pd.Series:
    return pd.to_numeric(df[c], errors="coerce").astype("float64")


def _nice(c: str) -> str:
    return str(c).replace("_", " ").strip().title()


def _style(fig: go.Figure, title: str, height: int = 340) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0, font=dict(size=15)),
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", y=-0.2, x=0),
        bargap=0.35,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.2)")
    return fig


def _pad_hbar(fig: go.Figure) -> go.Figure:
    """Leave room on the right so the value label of the longest bar is not clipped."""
    m = max((float(np.nanmax(t.x)) for t in fig.data if getattr(t, "x", None) is not None and len(t.x)), default=0)
    if m > 0:
        fig.update_xaxes(range=[0, m * 1.22])
    return fig


# --------------------------------------------------------------------------- column roles
@dataclass
class Roles:
    dates: list = field(default_factory=list)
    nums: list = field(default_factory=list)
    cats: list = field(default_factory=list)    # few repeated values (city, status)
    labels: list = field(default_factory=list)  # mostly-unique text (names)
    ids: list = field(default_factory=list)


def column_roles(df: pd.DataFrame) -> Roles:
    r = Roles()
    n = len(df)
    for c in df.columns:
        s = df[c]
        nn = s.dropna()
        if pd.api.types.is_datetime64_any_dtype(s):
            r.dates.append(c)
        elif pd.api.types.is_numeric_dtype(s):
            seq = len(nn) > 5 and (nn % 1 == 0).all() and (nn.astype("float64").diff().dropna() == 1).all()
            if ID_WORDS.search(str(c).lower()) or seq:
                r.ids.append(c)
            else:
                r.nums.append(c)
        elif not nn.empty:
            u = nn.nunique()
            if u <= 30 and (u / max(n, 1) <= 0.6 or u <= 6):
                r.cats.append(c)
            else:
                r.labels.append(c)
    return r


def detect_domain(df: pd.DataFrame) -> str:
    names = [str(c).lower() for c in df.columns]
    scores = {d: sum(any(w in n for w in words) for n in names) for d, words in DOMAIN_WORDS.items()}
    # A Type/Dr-Cr column full of income/expense values is a strong finance signal
    flow_vals = {"income", "expense", "credit", "debit", "cr", "dr", "deposit", "withdrawal"}
    for c in df.columns:
        s = df[c]
        if not pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_datetime64_any_dtype(s):
            vals = s.dropna().astype(str).str.lower().str.strip()
            if len(vals) and vals.isin(flow_vals).mean() >= 0.7:
                scores["finance"] += 3
    best = max(scores, key=lambda d: (scores[d], -list(scores).index(d)))
    return best if scores[best] >= 1 else "general"


def pick_measure(roles: Roles) -> str | None:
    for p in MEASURE_PRIORITY:
        for c in roles.nums:
            if p in str(c).lower():
                return c
    return roles.nums[0] if roles.nums else None


def pick_date(roles: Roles) -> str | None:
    for c in roles.dates:
        if "date" in str(c).lower():
            return c
    return roles.dates[0] if roles.dates else None


def group_candidates(roles: Roles) -> list[str]:
    pool = roles.cats + [c for c in roles.labels if c not in roles.cats]
    def rank(c):
        low = str(c).lower()
        for i, h in enumerate(GROUP_HINTS):
            if h in low:
                return i
        return len(GROUP_HINTS)
    return sorted(pool, key=rank)


# --------------------------------------------------------------------------- generic charts
def trend_chart(df, date, measure, agg="sum"):
    s = df[[date, measure]].dropna().copy()
    s[measure] = _num(s, measure)
    if len(s) < 3:
        return None, None
    span = (s[date].max() - s[date].min()).days
    freq, label = ("MS", "month") if span > 75 else ("D", "day")
    ts = s.set_index(date)[measure].resample(freq).agg(agg)
    if freq == "MS":
        ts = ts[ts.index >= ts.index.min()]
    if len(ts) < 2:
        return None, None
    fig = go.Figure(go.Scatter(x=ts.index, y=ts.values, mode="lines+markers",
                               line=dict(color=BLUE, width=2), marker=dict(size=8),
                               hovertemplate="%{x|%d %b %Y}<br>%{y:,.2f}<extra></extra>"))
    _style(fig, f"{_nice(measure)} by {label}")
    return fig, ts


def top_chart(df, group, measure, n=10, agg="sum", title=None):
    g = df.assign(_v=_num(df, measure)).groupby(group)["_v"].agg(agg).dropna()
    g = g.sort_values(ascending=False).head(n).sort_values()
    if len(g) < 2:
        return None
    fig = go.Figure(go.Bar(x=g.values, y=g.index.astype(str), orientation="h", marker_color=BLUE,
                           text=[fmt(v) for v in g.values], textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}<br>%{x:,.2f}<extra></extra>"))
    _style(fig, title or f"Top {_nice(group)} by {_nice(measure)}", height=max(280, 40 * len(g) + 90))
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.2)")
    fig.update_yaxes(showgrid=False)
    return _pad_hbar(fig)


def hist_chart(values: pd.Series, title: str):
    v = values.dropna()
    if len(v) < 5:
        return None
    fig = go.Figure(go.Histogram(x=v, marker_color=BLUE, nbinsx=20,
                                 hovertemplate="%{x}<br>count %{y}<extra></extra>"))
    _style(fig, title)
    fig.update_layout(bargap=0.08)
    return fig


def outlier_bounds(s: pd.Series, k: float = 3.0):
    """Wide fence (3 x IQR) so mixed-price data is not flagged for normal variation."""
    s = s.dropna()
    if len(s) < 8:
        return None
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


# --------------------------------------------------------------------------- student
def student_scores(df, roles):
    subj = [c for c in roles.nums if not NON_SCORE.search(str(c).lower())]
    pct = next((c for c in roles.nums if re.search(r"percent|average|avg", str(c).lower())), None)
    if pct:
        return _num(df, pct), subj
    if len(subj) >= 2:
        return df[subj].apply(pd.to_numeric, errors="coerce").astype("float64").mean(axis=1), subj
    if subj:
        return _num(df, subj[0]), subj
    return None, subj


def name_column(roles):
    for c in roles.labels + roles.cats:
        if "name" in str(c).lower():
            return c
    return roles.labels[0] if roles.labels else None


# --------------------------------------------------------------------------- finance
IN_WORDS = ("income", "credit", "deposit", "salary", "received", "inflow", "cr")
OUT_WORDS = ("expense", "debit", "withdraw", "spent", "payment", "outflow", "dr")


def finance_flows(df, roles):
    """Return a frame with per-row 'in' and 'out' amounts (both positive), or None."""
    cols = {str(c).lower(): c for c in roles.nums}
    in_c = next((c for k, c in cols.items() if any(w in k for w in ("income", "credit", "deposit", "inflow"))), None)
    out_c = next((c for k, c in cols.items() if any(w in k for w in ("expense", "debit", "withdraw", "outflow"))), None)
    if in_c or out_c:
        zero = pd.Series(0.0, index=df.index)
        i = _num(df, in_c).fillna(0) if in_c else zero
        o = _num(df, out_c).fillna(0).abs() if out_c else zero
        return pd.DataFrame({"in": i, "out": o}, index=df.index)
    amt = next((c for c in roles.nums if "amount" in str(c).lower() or "value" in str(c).lower()), None) or pick_measure(roles)
    if amt is None:
        return None
    a = _num(df, amt)
    for c in roles.cats:
        vals = df[c].astype(str).str.lower().str.strip()
        is_in = vals.isin({"income", "credit", "cr", "deposit", "received", "salary"})
        is_out = vals.isin({"expense", "debit", "dr", "withdrawal", "withdraw", "spent", "payment"})
        if (is_in | is_out).mean() >= 0.7:
            return pd.DataFrame({"in": a.where(is_in, 0).fillna(0).abs(), "out": a.where(is_out, 0).fillna(0).abs()}, index=df.index)
    if (a < 0).any():
        return pd.DataFrame({"in": a.clip(lower=0).fillna(0), "out": (-a).clip(lower=0).fillna(0)}, index=df.index)
    return None


# --------------------------------------------------------------------------- dashboard
@dataclass
class Dashboard:
    domain: str
    measure: str | None
    kpis: list = field(default_factory=list)       # (label, value, help)
    findings: list = field(default_factory=list)   # plain-English sentences
    charts: list = field(default_factory=list)     # (key, figure)
    notes: list = field(default_factory=list)      # why something was skipped
    period: str | None = None                      # "01 Jan 2024 to 28 Jun 2024"


def build_dashboard(df: pd.DataFrame, domain: str = "auto", pass_mark: float = 40.0) -> Dashboard:
    roles = column_roles(df)
    dom = detect_domain(df) if domain == "auto" else domain
    measure = pick_measure(roles)
    date = pick_date(roles)
    d = Dashboard(domain=dom, measure=measure)
    if date and df[date].notna().any():
        d.period = f"{df[date].min():%d %b %Y} to {df[date].max():%d %b %Y}"

    if dom == "student":
        _student(df, roles, d, pass_mark)
    elif dom == "finance":
        _finance(df, roles, d, date)
    else:
        _sales_general(df, roles, d, measure, date)

    # data-quality findings common to all
    for c in df.columns:
        miss = df[c].isna().mean()
        if miss >= 0.2:
            d.findings.append(f"**{_nice(c)}** is blank in {miss:.0%} of rows, so results using it may be incomplete.")
    if not d.charts:
        d.notes.append("Not enough numeric or date columns to draw automatic charts. Use the chart builder below.")
    return d


def _sales_general(df, roles, d, measure, date):
    if measure is None:
        d.notes.append("No numeric column found, so no totals or averages can be shown.")
        return
    v = _num(df, measure)
    label = _nice(measure)
    d.kpis += [
        (f"Total {label}", fmt(v.sum()), f"Sum of {measure}"),
        (f"Average {label}", fmt(v.mean()), "Mean per row"),
        (f"Largest {label}", fmt(v.max()), "Biggest single value"),
    ]
    groups = group_candidates(roles)
    for g in groups[:2]:
        grp = df.assign(_v=v).groupby(g)["_v"].sum().dropna().sort_values(ascending=False)
        if len(grp) >= 2 and grp.sum() != 0:
            share = grp.iloc[0] / grp.sum()
            d.findings.append(f"**{grp.index[0]}** is the top {_nice(g).lower()} with {fmt(grp.iloc[0])} ({share:.0%} of total {label.lower()}).")
            if g == groups[0]:
                d.kpis.append((f"Top {_nice(g)}", str(grp.index[0]), f"{share:.0%} of total"))
        fig = top_chart(df, g, measure)
        if fig:
            d.charts.append((f"top_{g}", fig))

    if date:
        fig, ts = trend_chart(df, date, measure)
        if fig:
            d.charts.insert(0, ("trend", fig))
            if len(ts) >= 2 and ts.iloc[-2] != 0:
                ch = (ts.iloc[-1] - ts.iloc[-2]) / abs(ts.iloc[-2]) * 100
                word = "up" if ch >= 0 else "down"
                partial = ts.index[-1].days_in_month > df[date].max().day and ts.index[-1].month == df[date].max().month
                d.findings.append(f"Latest month ({ts.index[-1]:%b %Y}) {label.lower()} is {word} {abs(ch):.0f}% vs the previous month."
                                  + (" Data stops before the month ends, so this comparison is partial." if partial else ""))
            peak = ts.idxmax()
            d.findings.append(f"Peak {label.lower()} was in **{peak:%b %Y}** ({fmt(ts.max())}).")

    fig = hist_chart(v, f"Spread of {label}")
    if fig:
        d.charts.append(("hist", fig))
    b = outlier_bounds(v)
    if b:
        hi_n, lo_n = int((v > b[1]).sum()), int((v < b[0]).sum())
        parts = ([f"{hi_n} above {fmt(b[1])}"] if hi_n else []) + ([f"{lo_n} below {fmt(b[0])}"] if lo_n else [])
        if parts:
            d.findings.append(f"Extreme value(s) in **{label}**: " + " and ".join(parts) + ". Check they are not typos.")


def _student(df, roles, d, pass_mark):
    score, subj = student_scores(df, roles)
    if score is None:
        d.notes.append("Could not find marks columns for student analysis.")
        return
    name = name_column(roles)
    sc = score.dropna()
    d.kpis += [
        ("Students", fmt(len(sc)), "Rows with a score"),
        ("Class average", fmt(sc.mean()), "Mean score"),
        ("Highest", fmt(sc.max()), None),
        ("Lowest", fmt(sc.min()), None),
        ("Pass rate", f"{(sc >= pass_mark).mean():.0%}", f"Score of {pass_mark:g} or more"),
    ]
    if name:
        top = df.loc[sc.idxmax(), name]
        d.kpis.append(("Top scorer", str(top), None))
        d.findings.append(f"**{top}** scored highest ({fmt(sc.max())}); class average is {fmt(sc.mean())}.")
    fails = int((sc < pass_mark).sum())
    if fails:
        d.findings.append(f"{fails} student(s) are below the pass mark of {pass_mark:g}.")

    # subject averages (wide format) or by subject column (long format)
    subj_cat = next((c for c in roles.cats if "subject" in str(c).lower()), None)
    if subj_cat and len(roles.nums):
        avg = df.assign(_v=score).groupby(subj_cat)["_v"].mean().dropna().sort_values()
    elif len(subj) >= 2:
        avg = df[subj].apply(pd.to_numeric, errors="coerce").mean().sort_values()
    else:
        avg = None
    if avg is not None and len(avg) >= 2:
        fig = go.Figure(go.Bar(x=avg.values, y=[_nice(i) for i in avg.index], orientation="h", marker_color=BLUE,
                               text=[fmt(x) for x in avg.values], textposition="outside", cliponaxis=False))
        _style(fig, "Average score by subject", height=max(280, 45 * len(avg) + 90))
        fig.update_yaxes(showgrid=False)
        _pad_hbar(fig)
        d.charts.append(("subjects", fig))
        d.findings.append(f"Strongest subject: **{_nice(avg.index[-1])}** ({fmt(avg.iloc[-1])}). Weakest: **{_nice(avg.index[0])}** ({fmt(avg.iloc[0])}).")

    # grade bands (one hue, light -> dark)
    # Bands scale with the pass mark: below pass, then three equal bands from pass mark to the top score.
    top = max(float(sc.max()), float(pass_mark) + 3)
    step = (top - pass_mark) / 3
    e1, e2 = pass_mark + step, pass_mark + 2 * step
    bands = pd.cut(sc, [-np.inf, pass_mark, e1, e2, np.inf], right=False,
                   labels=[f"Below {pass_mark:g}", f"{pass_mark:g}-{e1:.0f}", f"{e1:.0f}-{e2:.0f}", f"{e2:.0f}+"])
    cnt = bands.value_counts().reindex(bands.cat.categories).fillna(0)
    fig = go.Figure(go.Bar(x=list(cnt.index), y=cnt.values, marker_color=BLUE_BANDS, text=cnt.values, textposition="outside"))
    _style(fig, "Students by score band")
    d.charts.append(("bands", fig))

    fig = hist_chart(sc, "Score distribution")
    if fig:
        d.charts.append(("hist", fig))

    if name:
        r = pd.DataFrame({"n": df[name].astype(str), "s": score}).dropna().sort_values("s", ascending=False)
        top10 = r.head(10).sort_values("s")
        if len(top10) >= 2:
            fig = go.Figure(go.Bar(x=top10["s"], y=top10["n"], orientation="h", marker_color=BLUE,
                                   text=[fmt(x) for x in top10["s"]], textposition="outside", cliponaxis=False))
            _style(fig, "Top 10 students", height=max(280, 40 * len(top10) + 90))
            fig.update_yaxes(showgrid=False)
            _pad_hbar(fig)
            d.charts.append(("top_students", fig))
        low = r[r["s"] < pass_mark]
        if len(low):
            d.findings.append("Needs support: " + ", ".join(low["n"].head(5)) + (" and others." if len(low) > 5 else "."))


def _finance(df, roles, d, date):
    flows = finance_flows(df, roles)
    if flows is None:
        d.notes.append("Could not tell income from expenses. Add a column such as Type (income/expense) or Debit/Credit, or use signed amounts.")
        _sales_general(df, roles, d, pick_measure(roles), date)
        return
    tin, tout = flows["in"].sum(), flows["out"].sum()
    net = tin - tout
    d.kpis += [
        ("Total in", fmt(tin), "Income / credits"),
        ("Total out", fmt(tout), "Expenses / debits"),
        ("Net", fmt(net), "In minus out"),
    ]
    if tin > 0:
        rate = net / tin
        d.kpis.append(("Savings rate", f"{rate:.0%}", "Net as a share of income"))
        d.findings.append(f"You kept {rate:.0%} of income (net {fmt(net)})." if net >= 0 else f"Spending exceeds income by {fmt(-net)}.")

    cat = next((c for c in group_candidates(roles) if not df[c].astype(str).str.lower().isin({"income", "expense", "credit", "debit"}).mean() > 0.7), None)
    if cat:
        spend = flows["out"].groupby(df[cat]).sum().sort_values(ascending=False)
        spend = spend[spend > 0]
        if len(spend) >= 2:
            d.findings.append(f"Biggest spending category: **{spend.index[0]}** ({fmt(spend.iloc[0])}, {spend.iloc[0] / spend.sum():.0%} of expenses).")
            top = spend.head(10).sort_values()
            fig = go.Figure(go.Bar(x=top.values, y=top.index.astype(str), orientation="h", marker_color=BLUE,
                                   text=[fmt(x) for x in top.values], textposition="outside", cliponaxis=False))
            _style(fig, f"Spending by {_nice(cat)}", height=max(280, 40 * len(top) + 90))
            fig.update_yaxes(showgrid=False)
            _pad_hbar(fig)
            d.charts.append(("spend_cat", fig))

    if date:
        tmp = flows.assign(_d=df[date]).dropna(subset=["_d"])
        if len(tmp) >= 3:
            span = (tmp["_d"].max() - tmp["_d"].min()).days
            freq = "MS" if span > 75 else "D"
            m = tmp.set_index("_d")[["in", "out"]].resample(freq).sum()
            if len(m) >= 2:
                fig = go.Figure()
                fig.add_bar(x=m.index, y=m["in"], name="In", marker_color=BLUE)
                fig.add_bar(x=m.index, y=m["out"], name="Out", marker_color=ORANGE)
                fig.update_layout(barmode="group")
                _style(fig, "Money in vs out by " + ("month" if freq == "MS" else "day"))
                d.charts.insert(0, ("in_out", fig))
                net_m = m["in"] - m["out"]
                worst = net_m.idxmin()
                d.findings.append(f"Weakest period: **{worst:%b %Y}** (net {fmt(net_m.min())}).")


# --------------------------------------------------------------------------- custom chart builder
def custom_chart(df: pd.DataFrame, x: str, y: str | None, kind: str, agg: str = "sum"):
    """kind: Bar, Line, Scatter, Histogram, Box. Returns a figure or raises ValueError with a friendly message."""
    work = df.copy()
    if y is not None:
        y_is_num = pd.api.types.is_numeric_dtype(work[y])
        if not y_is_num and kind != "Bar":
            raise ValueError("Pick a numeric column for Y.")
        if y_is_num:
            work[y] = _num(work, y)
    if kind == "Histogram":
        if not pd.api.types.is_numeric_dtype(work[x]):
            raise ValueError("Histogram needs a numeric X column.")
        fig = px.histogram(work, x=x, color_discrete_sequence=[BLUE], nbins=20)
        return _style(fig, f"Distribution of {_nice(x)}")
    if y is None:
        raise ValueError("Pick a Y column.")
    if kind in ("Bar", "Line"):
        g = work.groupby(x)[y].agg(agg if pd.api.types.is_numeric_dtype(work[y]) else "count").reset_index()
        if len(g) > 40 and kind == "Bar":
            g = g.sort_values(y, ascending=False).head(40)
        if kind == "Line":
            fig = px.line(g.sort_values(x), x=x, y=y, markers=True, color_discrete_sequence=[BLUE])
        else:
            fig = px.bar(g.sort_values(y, ascending=False), x=x, y=y, color_discrete_sequence=[BLUE])
        return _style(fig, f"{agg.title() if pd.api.types.is_numeric_dtype(work[y]) else 'Count'} of {_nice(y)} by {_nice(x)}")
    if kind == "Scatter":
        if not pd.api.types.is_numeric_dtype(work[x]):
            raise ValueError("Scatter needs numeric X and Y columns.")
        fig = px.scatter(work, x=x, y=y, color_discrete_sequence=[BLUE])
        fig.update_traces(marker=dict(size=9, line=dict(width=2, color="rgba(255,255,255,0.9)")))
        return _style(fig, f"{_nice(y)} vs {_nice(x)}")
    if kind == "Box":
        fig = px.box(work, x=x if not pd.api.types.is_numeric_dtype(work[x]) else None, y=y, color_discrete_sequence=[BLUE])
        return _style(fig, f"Spread of {_nice(y)}" + (f" by {_nice(x)}" if not pd.api.types.is_numeric_dtype(work[x]) else ""))
    raise ValueError("Unknown chart type.")
