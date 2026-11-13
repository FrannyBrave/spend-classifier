import io
import pandas as pd
import streamlit as st

from classifier import classify_line
from config import HAIKU_MODEL, SONNET_MODEL

st.set_page_config(page_title="Spend Classifier", layout="wide")

st.info(
    "Demo with synthetic data. Not real company data.",
    icon="ℹ️",
)

st.title("Spend Classifier")

tab_classify, tab_review, tab_results = st.tabs(
    ["Classify", "Review Queue", "Results"]
)

# ── Classify tab ─────────────────────────────────────────────────────────────
with tab_classify:
    st.header("Classify purchase-order lines")

    col_upload, col_model = st.columns([3, 1])

    with col_upload:
        uploaded = st.file_uploader(
            "Upload a CSV (supplier, description, amount_usd)",
            type="csv",
        )

    with col_model:
        model_label = st.selectbox(
            "Model",
            options=["Haiku (fast, cheap)", "Sonnet (accurate, slower)"],
        )
    model_id = HAIKU_MODEL if model_label.startswith("Haiku") else SONNET_MODEL

    # Load data: uploaded file or default sample
    if uploaded is not None:
        df_input = pd.read_csv(uploaded)
        source_label = uploaded.name
    else:
        df_input = pd.read_csv("data/sample_po_lines.csv")
        source_label = "data/sample_po_lines.csv (default)"

    required_cols = {"supplier", "description", "amount_usd"}
    missing = required_cols - set(df_input.columns)

    st.caption(f"Loaded **{len(df_input)} rows** from `{source_label}`")

    if missing:
        st.error(f"CSV is missing required columns: {', '.join(sorted(missing))}")
        st.stop()

    run_clicked = st.button("Run classification", type="primary")

    if run_clicked:
        progress_bar = st.progress(0, text="Starting…")
        rows = []
        total = len(df_input)

        for i, row in df_input.iterrows():
            pct = i / total
            progress_bar.progress(
                pct,
                text=f"Classifying {i + 1}/{total} — {row['supplier'][:40]}",
            )
            result = classify_line(
                supplier=str(row["supplier"]),
                description=str(row["description"]),
                amount=row["amount_usd"],
                model=model_id,
            )
            rows.append({
                "line_id":    row.get("line_id", f"row-{i+1}"),
                "supplier":   row["supplier"],
                "description": row["description"],
                "amount_usd": row["amount_usd"],
                "category":   result["category"],
                "confidence": result["confidence"],
                "reason":     result["reason"],
                "input_tokens":  result["input_tokens"],
                "output_tokens": result["output_tokens"],
                "latency_s":  result["latency_seconds"],
                "model":      model_id,
            })

        progress_bar.progress(1.0, text="Done.")
        results_df = pd.DataFrame(rows)
        st.session_state["results"] = results_df
        st.session_state["results_model"] = model_label

    # Show results table if we have data in session state
    if "results" in st.session_state:
        results_df = st.session_state["results"]
        model_used = st.session_state.get("results_model", "")

        n_error = (results_df["category"] == "error").sum()
        n_ok = len(results_df) - n_error

        summary_cols = st.columns(4)
        summary_cols[0].metric("Rows classified", n_ok)
        summary_cols[1].metric("Errors", n_error)
        summary_cols[2].metric(
            "Avg confidence",
            f"{results_df.loc[results_df['category'] != 'error', 'confidence'].mean():.0%}",
        )
        summary_cols[3].metric("Model", model_used)

        display_df = results_df[
            ["line_id", "supplier", "description", "amount_usd",
             "category", "confidence", "reason"]
        ].copy()
        display_df["confidence"] = display_df["confidence"] * 100

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "line_id":     st.column_config.TextColumn("PO Line", width="small"),
                "supplier":    st.column_config.TextColumn("Supplier", width="medium"),
                "description": st.column_config.TextColumn("Description", width="large"),
                "amount_usd":  st.column_config.NumberColumn(
                    "Amount (USD)", format="$%.2f", width="medium"
                ),
                "category":    st.column_config.TextColumn("Category", width="medium"),
                "confidence":  st.column_config.ProgressColumn(
                    "Confidence",
                    min_value=0.0,
                    max_value=1.0,
                    format="%.0f%%",
                    width="small",
                ),
                "reason":      st.column_config.TextColumn("Reason", width="large"),
            },
        )

        csv_bytes = results_df.to_csv(index=False).encode()
        st.download_button(
            "Download results CSV",
            data=csv_bytes,
            file_name="classified_po_lines.csv",
            mime="text/csv",
        )

# ── Review Queue tab ──────────────────────────────────────────────────────────
with tab_review:
    st.header("Review Queue")
    st.info("Coming soon — low-confidence and error rows will appear here for manual review.")

# ── Results tab ───────────────────────────────────────────────────────────────
with tab_results:
    st.header("Results")
    st.info("Coming soon — accuracy scoring against the answer key will appear here.")
