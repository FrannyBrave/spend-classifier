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
                    min_value=0,
                    max_value=100,
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

    if "results" not in st.session_state:
        st.info("Run a classification on the Classify tab first.")
    else:
        results_df = st.session_state["results"]

        threshold = st.slider(
            "Confidence threshold — lines below this are sent to review",
            min_value=0.0, max_value=1.0, value=0.75, step=0.01,
            format="%.2f",
        )

        # Partition rows
        is_error = results_df["category"] == "error"
        is_low   = results_df["confidence"] < threshold
        mask_queue = is_error | is_low

        df_queue = results_df[mask_queue].copy()
        df_auto  = results_df[~mask_queue].copy()

        # Initialise reviews dict (line_id → {final_category, human_reviewed})
        # Only add rows not yet tracked; don't reset existing decisions.
        if "reviews" not in st.session_state:
            st.session_state["reviews"] = {}
        reviews = st.session_state["reviews"]

        for _, row in results_df.iterrows():
            lid = row["line_id"]
            if lid not in reviews:
                reviews[lid] = {
                    "final_category": row["category"],
                    "human_reviewed": False,
                }

        # Counts
        n_auto     = len(df_auto)
        n_queue    = len(df_queue)
        n_reviewed = sum(
            1 for lid in df_queue["line_id"]
            if reviews.get(lid, {}).get("human_reviewed")
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("Auto-accepted", n_auto)
        m2.metric("Sent to review", n_queue)
        m3.metric("Reviewed", f"{n_reviewed} / {n_queue}")

        st.divider()

        if df_queue.empty:
            st.success("All lines are above the threshold — nothing to review.")
        else:
            from classifier import CATEGORIES  # already imported at module level but safe to re-reference

            for _, row in df_queue.iterrows():
                lid       = row["line_id"]
                review    = reviews[lid]
                reviewed  = review["human_reviewed"]

                with st.container():
                    # Header row: PO id + supplier + amount
                    h1, h2 = st.columns([5, 1])
                    with h1:
                        st.markdown(
                            f"**{lid}** &nbsp;·&nbsp; {row['supplier']} &nbsp;·&nbsp; "
                            f"_{row['description']}_"
                        )
                    with h2:
                        st.markdown(f"**${row['amount_usd']:,.2f}**")

                    # AI prediction row
                    conf_pct = int(round(row["confidence"] * 100))
                    conf_color = (
                        "🟢" if row["confidence"] >= 0.75
                        else "🟡" if row["confidence"] >= 0.50
                        else "🔴"
                    )
                    st.caption(
                        f"AI: **{row['category']}** &nbsp; {conf_color} {conf_pct}% confidence &nbsp;·&nbsp; {row['reason']}"
                    )

                    if reviewed:
                        rc1, rc2 = st.columns([5, 1])
                        with rc1:
                            st.success(
                                f"✓ Reviewed — final category: **{review['final_category']}**",
                                icon="✅",
                            )
                        with rc2:
                            if st.button("Undo", key=f"undo_{lid}"):
                                reviews[lid]["human_reviewed"] = False
                                st.rerun()
                    else:
                        a1, a2, a3 = st.columns([1, 2, 3])

                        with a1:
                            if st.button("Approve AI category", key=f"approve_{lid}"):
                                reviews[lid] = {
                                    "final_category": row["category"],
                                    "human_reviewed": True,
                                }
                                st.rerun()

                        with a2:
                            # Initialise the selectbox state once; never pass
                            # index= after that, or Streamlit resets the value
                            # on every rerun (the Save bug).
                            sel_key = f"sel_{lid}"
                            if sel_key not in st.session_state:
                                ai_cat = row["category"]
                                st.session_state[sel_key] = (
                                    ai_cat if ai_cat in CATEGORIES else CATEGORIES[0]
                                )
                            st.selectbox(
                                "Override category",
                                options=CATEGORIES,
                                key=sel_key,
                                label_visibility="collapsed",
                            )

                        with a3:
                            if st.button("Save override", key=f"override_{lid}"):
                                reviews[lid] = {
                                    "final_category": st.session_state[f"sel_{lid}"],
                                    "human_reviewed": True,
                                }
                                st.rerun()

                    st.divider()

# ── Results tab ───────────────────────────────────────────────────────────────
with tab_results:
    st.header("Results")
    st.info("Coming soon — accuracy scoring against the answer key will appear here.")
