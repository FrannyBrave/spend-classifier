import os
import pandas as pd
import streamlit as st

from classifier import classify_line, CATEGORIES
from config import (
    HAIKU_MODEL, SONNET_MODEL,
    HAIKU_INPUT_PRICE_PER_M, HAIKU_OUTPUT_PRICE_PER_M,
    SONNET_INPUT_PRICE_PER_M, SONNET_OUTPUT_PRICE_PER_M,
)

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
            format="%.2f", key="threshold",
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

    ANSWER_KEY_PATH = "data/answer_key.csv"
    _MODEL_META = {
        HAIKU_MODEL:  ("Haiku",  HAIKU_INPUT_PRICE_PER_M,  HAIKU_OUTPUT_PRICE_PER_M),
        SONNET_MODEL: ("Sonnet", SONNET_INPUT_PRICE_PER_M, SONNET_OUTPUT_PRICE_PER_M),
    }

    if not os.path.exists(ANSWER_KEY_PATH):
        st.info("No answer key found at data/answer_key.csv — accuracy scoring unavailable.")
    elif "results" not in st.session_state:
        st.info("Run a classification on the Classify tab first.")
    else:
        results_df = st.session_state["results"]
        reviews    = st.session_state.get("reviews", {})
        threshold  = st.session_state.get("threshold", 0.75)
        answer_key = pd.read_csv(ANSWER_KEY_PATH)

        # Merge AI results with answer key; skip unmatched lines
        merged = results_df.merge(answer_key[["line_id", "correct_category"]], on="line_id", how="inner")

        # Attach final (post-review) category
        merged["final_category"] = merged.apply(
            lambda r: reviews.get(r["line_id"], {}).get("final_category", r["category"]),
            axis=1,
        )

        # Score only non-error rows
        scored = merged[merged["category"] != "error"].copy()
        n = len(scored)

        # ── 1. Overall accuracy ───────────────────────────────────────────────
        st.subheader("Overall accuracy")

        ai_acc     = (scored["category"]       == scored["correct_category"]).mean() if n else 0
        review_acc = (scored["final_category"] == scored["correct_category"]).mean() if n else 0
        ai_n     = int((scored["category"]       == scored["correct_category"]).sum())
        review_n = int((scored["final_category"] == scored["correct_category"]).sum())

        c1, c2, c3 = st.columns(3)
        c1.metric("AI alone",     f"{ai_acc:.1%}",     f"{ai_n} / {n} correct")
        c2.metric("After review", f"{review_acc:.1%}", f"{review_n} / {n} correct")
        c3.metric("Confidence threshold used", f"{threshold:.0%}")

        # ── 2. Accuracy above vs below threshold ─────────────────────────────
        st.subheader("Accuracy by confidence band")

        above = scored[scored["confidence"] >= threshold]
        below = scored[scored["confidence"] <  threshold]

        def _band_row(label, df):
            if df.empty:
                return {"Band": label, "Lines": 0, "AI correct": 0, "AI accuracy": "—",
                        "After review": 0, "Review accuracy": "—"}
            ai_ok  = (df["category"]       == df["correct_category"]).sum()
            rev_ok = (df["final_category"] == df["correct_category"]).sum()
            return {
                "Band":             label,
                "Lines":            len(df),
                "AI correct":       int(ai_ok),
                "AI accuracy":      f"{ai_ok / len(df):.0%}",
                "After review":     int(rev_ok),
                "Review accuracy":  f"{rev_ok / len(df):.0%}",
            }

        band_df = pd.DataFrame([
            _band_row(f"Above {threshold:.0%}", above),
            _band_row(f"Below {threshold:.0%}", below),
        ])
        st.dataframe(band_df, hide_index=True, use_container_width=True)

        # ── 3. Confusion matrix ───────────────────────────────────────────────
        st.subheader("Confusion matrix — AI predictions")
        st.caption("Rows = correct category · Columns = AI predicted · Diagonal = correct")

        cm = pd.crosstab(
            scored["correct_category"],
            scored["category"],
        )
        cm = cm.reindex(index=CATEGORIES, columns=CATEGORIES, fill_value=0)
        cm.index.name   = "Correct ↓ / Predicted →"
        cm.columns.name = None

        # Shorten labels so columns aren't squeezed
        short = {
            "Direct Materials":    "Direct Mat.",
            "MRO":                 "MRO",
            "Logistics & Freight": "Logistics",
            "IT & Software":       "IT & SW",
            "Professional Services": "Prof. Svcs",
            "Facilities":          "Facilities",
        }
        cm_display = cm.copy()
        cm_display.columns = [short.get(c, c) for c in cm_display.columns]
        cm_display.index   = [short.get(i, i) for i in cm_display.index]

        def _style_cm(df):
            """Green bold for diagonal (correct); light red for non-zero errors."""
            out = pd.DataFrame("", index=df.index, columns=df.columns)
            for i in range(len(df.index)):
                for j in range(len(df.columns)):
                    if i == j:
                        out.iloc[i, j] = "background-color: #d4edda; font-weight: bold"
                    elif df.iloc[i, j] > 0:
                        out.iloc[i, j] = "background-color: #fce8e8"
            return out

        st.dataframe(
            cm_display.style.apply(_style_cm, axis=None).format("{:d}"),
            width="stretch",
        )

        # ── 4. Model comparison ───────────────────────────────────────────────
        st.subheader("Model comparison")

        model_slug = {m: m.replace("/", "-") for m in _MODEL_META}
        saved = {
            m: pd.read_csv(f"results/{model_slug[m]}.csv")
            for m in _MODEL_META
            if os.path.exists(f"results/{model_slug[m]}.csv")
        }

        if len(saved) < 2:
            missing_names = [_MODEL_META[m][0] for m in _MODEL_META if m not in saved]
            st.info(
                f"Run the batch script for **{', '.join(missing_names)}** "
                f"to enable side-by-side comparison."
            )
        else:
            comp_cols = st.columns(len(saved))
            for col, (model_id, df_saved) in zip(comp_cols, saved.items()):
                label, price_in, price_out = _MODEL_META[model_id]

                # Accuracy
                df_m = df_saved.merge(
                    answer_key[["line_id", "correct_category"]], on="line_id", how="inner"
                )
                df_m_scored = df_m[df_m["predicted_category"] != "error"]
                acc = (df_m_scored["predicted_category"] == df_m_scored["correct_category"]).mean()

                # Cost
                total_cost = (
                    df_saved["input_tokens"].sum()  * price_in +
                    df_saved["output_tokens"].sum() * price_out
                ) / 1_000_000

                # Avg latency
                avg_latency = df_saved["latency_seconds"].mean()

                # Lines sent to review (below threshold)
                n_review = (df_saved["confidence"] < threshold).sum()

                col.markdown(f"### {label}")
                col.metric("Accuracy",           f"{acc:.1%}")
                col.metric("Total cost",         f"${total_cost:.4f}")
                col.metric("Avg sec / line",     f"{avg_latency:.2f}s")
                col.metric("Lines to review",    int(n_review),
                           help=f"Confidence < {threshold:.0%}")
