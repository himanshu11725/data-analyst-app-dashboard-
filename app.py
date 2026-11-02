"""Personal Data Analyst - Step 1: upload, clean, profile, download."""
import io

import pandas as pd
import streamlit as st

from cleaner import MAX_FILE_MB, CleanOptions, FileError, clean, load_file, profile
from insights import DOMAINS, build_dashboard, column_roles, custom_chart, detect_domain

st.set_page_config(page_title="Tidy Data", page_icon="📊", layout="wide")

st.title("📊 Tidy Data ")
st.caption(
    "Upload a messy CSV or Excel file. Get a cleaned version, a health check, and a download. "
    "Your file is processed in memory and is not stored."
)

# ---------------- sidebar options ----------------
with st.sidebar:
    st.header("Cleaning options")
    opts = CleanOptions(
        drop_blank_rows=st.checkbox("Remove blank rows", True),
        drop_blank_cols=st.checkbox("Remove blank columns", True),
        drop_duplicates=st.checkbox("Remove duplicate rows", True),
        trim_text=st.checkbox("Trim extra spaces", True),
        fix_nulls=st.checkbox("Turn 'N/A', '-', 'null' into blanks", True),
        fix_numbers=st.checkbox("Convert text to numbers (₹1,200 → 1200)", True),
        fix_dates=st.checkbox("Standardize dates", True),
        day_first=st.checkbox("Dates are day-first (05/03 = 5 March)", True),
        unify_text_case=st.checkbox("Unify spelling/case (delhi, DELHI → Delhi)", True),
        snake_case_columns=st.checkbox("Tidy column names", True),
    )
    st.divider()
    st.header("Dashboard settings")
    domain_choice = st.selectbox(
        "Type of data", ["auto"] + DOMAINS,
        format_func=lambda x: {"auto": "Auto-detect", "sales": "Sales", "student": "Student marks",
                               "finance": "Finance", "general": "General"}[x],
    )
    pass_slot = st.container()  # filled later, only for student data
    st.divider()
    st.caption(f"Limits: {MAX_FILE_MB} MB, .csv / .xlsx / .xls")

uploaded = st.file_uploader("Upload your data file", type=["csv", "xlsx", "xls"])

if uploaded is None:
    st.info("👆 Upload a file to begin. No file handy? Download the sample from the README folder.")
    st.stop()


@st.cache_data(show_spinner="Reading file...")
def _load(raw: bytes, name: str) -> pd.DataFrame:
    return load_file(raw, name)


try:
    raw_df = _load(uploaded.getvalue(), uploaded.name)
except FileError as err:
    st.error(str(err))
    st.stop()

result = clean(raw_df, opts)
cleaned = result.df

# Settings that only apply to one data type appear only for that type.
effective_domain = detect_domain(cleaned) if domain_choice == "auto" else domain_choice
pass_mark = 40.0
if effective_domain == "student":
    with pass_slot:
        pass_mark = st.number_input("Pass mark", 0.0, 1000.0, 40.0, step=1.0,
                                    help="Students scoring below this are counted as not passing.")
before, after = profile(raw_df), profile(cleaned)

# ---------------- headline numbers ----------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{after['rows']:,}", f"{after['rows'] - before['rows']:,}" if after["rows"] != before["rows"] else None)
c2.metric("Columns", after["columns"])
c3.metric("Missing cells", f"{after['missing_cells']:,}", f"{after['missing_cells'] - before['missing_cells']:,}" if after["missing_cells"] != before["missing_cells"] else None, delta_color="inverse")
c4.metric("Duplicate rows", after["duplicate_rows"])

tab_dash, tab_clean, tab_profile, tab_data, tab_download = st.tabs(
    ["📈 Dashboard", "🧹 What was fixed", "🔍 Data health", "📄 Cleaned data", "⬇️ Download"]
)

with tab_dash:
    try:
        dash = build_dashboard(cleaned, domain_choice, pass_mark)
    except Exception as exc:  # noqa: BLE001 - never show a stack trace to a visitor
        st.error("Could not build the dashboard for this file. Try the chart builder below or change the data type in the sidebar.")
        st.caption(f"Technical detail: {type(exc).__name__}")
        dash = None

    if dash is not None:
        names = {"sales": "Sales", "student": "Student marks", "finance": "Finance", "general": "General"}
        st.caption(
            f"Detected data type: **{names[dash.domain]}**. Wrong? Change it in the sidebar."
            + (f"  \nData covers **{dash.period}**." if dash.period else "")
        )

        for i in range(0, len(dash.kpis), 4):
            row = dash.kpis[i:i + 4]
            for col, (label, value, hint) in zip(st.columns(4), row):
                col.metric(label, value, help=hint)

        if dash.findings:
            st.subheader("Key findings")
            for line in dash.findings:
                st.markdown(f"- {line}")
        for note in dash.notes:
            st.info(note)

        for i in range(0, len(dash.charts), 2):
            for col, (key, fig) in zip(st.columns(2), dash.charts[i:i + 2]):
                col.plotly_chart(fig, key=f"auto_{key}")

    with st.expander("🛠️ Build your own chart"):
        roles = column_roles(cleaned)
        all_cols = list(cleaned.columns)
        b1, b2, b3, b4 = st.columns(4)
        kind = b1.selectbox("Chart type", ["Bar", "Line", "Scatter", "Histogram", "Box"])
        x = b2.selectbox("X (group / horizontal)", all_cols)
        y_options = [None] + all_cols
        y_default = y_options.index(roles.nums[0]) if roles.nums else 0
        y = b3.selectbox("Y (value)", y_options, index=y_default,
                         format_func=lambda v: "(none)" if v is None else v,
                         disabled=(kind == "Histogram"))
        agg = b4.selectbox("Combine values by", ["sum", "mean", "median", "max", "min", "count"],
                           disabled=(kind not in ("Bar", "Line")))
        try:
            st.plotly_chart(custom_chart(cleaned, x, None if kind == "Histogram" else y, kind, agg), key="custom")
        except ValueError as err:
            st.warning(str(err))
        except Exception as exc:  # noqa: BLE001
            st.warning(f"That combination can't be charted ({type(exc).__name__}). Try different columns.")
        st.caption("Tip: hover a chart and use the camera icon to save it as an image.")

with tab_clean:
    for line in result.log:
        st.markdown(f"- {line}")

with tab_profile:
    st.subheader("Columns")
    st.dataframe(after["columns_table"], width="stretch", hide_index=True)

    miss = after["columns_table"].set_index("column")["missing"]
    if miss.sum() > 0:
        st.subheader("Missing values by column")
        st.bar_chart(miss[miss > 0])

    if after["numeric_summary"] is not None:
        st.subheader("Numeric summary")
        st.dataframe(after["numeric_summary"], width="stretch")
    if after["correlation"] is not None:
        st.subheader("Correlation between numeric columns")
        try:
            # Colored heatmap needs matplotlib; fall back to a plain table if it is missing.
            styled = after["correlation"].style.background_gradient(cmap="RdBu", vmin=-1, vmax=1)
            st.dataframe(styled, width="stretch")
        except ImportError:
            st.dataframe(after["correlation"], width="stretch")

with tab_data:
    st.caption("Showing the first 500 rows.")
    st.dataframe(cleaned.head(500), width="stretch")

with tab_download:
    base = uploaded.name.rsplit(".", 1)[0]
    csv_bytes = cleaned.to_csv(index=False).encode("utf-8-sig")
    xl_buf = io.BytesIO()
    with pd.ExcelWriter(xl_buf, engine="openpyxl") as writer:
        cleaned.to_excel(writer, index=False, sheet_name="cleaned")
        pd.DataFrame({"What was fixed": result.log}).to_excel(writer, index=False, sheet_name="cleaning_log")
    st.download_button("⬇️ Download cleaned CSV", csv_bytes, f"{base}_cleaned.csv", "text/csv")
    st.download_button(
        "⬇️ Download cleaned Excel (with cleaning log)",
        xl_buf.getvalue(),
        f"{base}_cleaned.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
