import os
from pathlib import Path

import pandas as pd
import streamlit as st

from classifier import classify_line, CATEGORIES
from config import (
    HAIKU_MODEL, SONNET_MODEL,
    HAIKU_INPUT_PRICE_PER_M, HAIKU_OUTPUT_PRICE_PER_M,
    SONNET_INPUT_PRICE_PER_M, SONNET_OUTPUT_PRICE_PER_M,
)

# Resolve data paths relative to this file so the app works from any directory.
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
SAMPLE_PATH = DATA_DIR / "sample_po_lines.csv"
ANSWER_KEY_PATH = DATA_DIR / "answer_key.csv"

# Guards for public deployments (they spend the owner's API credits):
#   SPEND_CLASSIFIER_MAX_ROWS     max rows accepted from an uploaded CSV (default 100)
#   SPEND_CLASSIFIER_DISABLE_LIVE set to 1 to disable live API calls entirely
try:
    MAX_UPLOAD_ROWS = int(os.getenv("SPEND_CLASSIFIER_MAX_ROWS", "100"))
except ValueError:
    MAX_UPLOAD_ROWS = 100
LIVE_DISABLED = os.getenv("SPEND_CLASSIFIER_DISABLE_LIVE", "").strip() == "1"

_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")

st.set_page_config(page_title="Spend Classifier", layout="wide")


def _short(value, n: int) -> str:
    """First n characters of a cell as text; blank/NaN cells become ''."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value)[:n]


def _coerce_amount(series: pd.Series) -> pd.Series:
    """Parse amounts like '$1,200.50' to numbers; unreadable values become NaN."""
    cleaned = series.astype(str).str.replace(r"[$,\s]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def _money(value) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if pd.isna(number) else f"${number:,.2f}"


def _csv_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Neutralise spreadsheet formula injection in text cells of a download.

    Text starting with = + - @ would run as a formula when opened in Excel or
    Sheets, so it gets a leading apostrophe.
    """
    out = df.copy()
    for col in out.columns:
        if not pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].map(
                lambda v: "'" + v if isinstance(v, str) and v.startswith(_FORMULA_PREFIXES) else v
            )
    return out


def _set_results(df: pd.DataFrame, label: str) -> None:
    """Store new results and reset anything derived from the previous results."""
    st.session_state["results"] = df
    st.session_state["results_model"] = label
    st.session_state["reviews"] = {}
    for key in [k for k in st.session_state if str(k).startswith("sel_")]:
        del st.session_state[key]


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
            help=f"Up to {MAX_UPLOAD_ROWS} rows. Include a unique line_id column to use saved results.",
        )

    with col_model:
        model_label = st.selectbox(
            "Model",
            options=["Haiku (fast, cheap)", "Sonnet (accurate, slower)"],
        )
    model_id = HAIKU_MODEL if model_label.startswith("Haiku") else SONNET_MODEL

    # Load data: uploaded file or default sample
    if uploaded is not None:
        try:
            df_input = pd.read_csv(uploaded)
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
            st.error(f"Could not read that CSV: {exc}")
            st.stop()
        source_label = uploaded.name
    else:
        df_input = pd.read_csv(SAMPLE_PATH)
        source_label = "data/sample_po_lines.csv (default)"

    required_cols = {"supplier", "description", "amount_usd"}
    missing = required_cols - set(df_input.columns)

    st.caption(f"Loaded **{len(df_input)} rows** from `{source_label}`")

    if missing:
        st.error(f"CSV is missing required columns: {', '.join(sorted(missing))}")
        st.stop()

    # Public-demo guard: uploads are what spend API credits.
    if uploaded is not None and len(df_input) > MAX_UPLOAD_ROWS:
        st.error(
            f"Uploads are limited to {MAX_UPLOAD_ROWS} rows; this file has {len(df_input)}. "
            "Trim the file and upload it again."
        )
        st.stop()

    # Duplicate line_ids would collide in the review queue and in saved-result merges.
    if "line_id" in df_input.columns:
        dupes = df_input.loc[df_input["line_id"].duplicated(keep=False), "line_id"].astype(str).unique()
        if len(dupes):
            st.error(
                "line_id values must be unique. Duplicates: "
                + ", ".join(dupes[:5]) + (" …" if len(dupes) > 5 else "")
            )
            st.stop()

    df_input = df_input.reset_index(drop=True)
    df_input["amount_usd"] = _coerce_amount(df_input["amount_usd"])
    n_bad_amount = int(df_input["amount_usd"].isna().sum())
    if n_bad_amount:
        st.warning(f"{n_bad_amount} row(s) have an unreadable amount; they will be classified with the amount shown as unknown.")

    btn_col1, btn_col2 = st.columns([2, 3])
    run_clicked  = btn_col1.button(
        "Run classification", type="primary", disabled=LIVE_DISABLED,
        help="Live API calls are turned off in this deployment. Use 'Load saved results'." if LIVE_DISABLED else None,
    )
    load_clicked = btn_col2.button("Load saved results", help="Loads the saved batch CSV for the selected model from results/")

    if load_clicked:
        saved_path = RESULTS_DIR / f"{model_id.replace('/', '-')}.csv"
        shown_path = f"results/{saved_path.name}"
        if not saved_path.exists():
            st.error(f"No saved results found at `{shown_path}`. Run `python run_batch.py --model {'haiku' if model_id == HAIKU_MODEL else 'sonnet'}` first.")
        elif "line_id" not in df_input.columns:
            st.error("Saved results are matched by `line_id`, but this CSV has no `line_id` column. Remove the upload to use the sample data.")
        else:
            saved_df = pd.read_csv(saved_path)
            # Align column names with what the Classify tab produces
            saved_df = saved_df.rename(columns={
                "predicted_category": "category",
                "latency_seconds":    "latency_s",
            })
            # Merge in supplier/description/amount from the input CSV
            saved_df = saved_df.merge(
                df_input[["line_id", "supplier", "description", "amount_usd"]],
                on="line_id", how="left",
            )
            saved_df["model"] = model_id
            _set_results(saved_df, model_label)
            st.success(f"Loaded {len(saved_df)} rows from `{shown_path}`.")
            n_unmatched = int(saved_df["supplier"].isna().sum())
            if n_unmatched:
                st.warning(
                    f"{n_unmatched} saved row(s) have no matching line_id in the loaded CSV, "
                    "so their supplier and description are blank."
                )

    if run_clicked:
        progress_bar = st.progress(0, text="Starting…")
        rows = []
        total = len(df_input)

        for i, row in df_input.iterrows():
            pct = i / total
            progress_bar.progress(
                pct,
                text=f"Classifying {i + 1}/{total} — {_short(row['supplier'], 40)}",
            )
            try:
                # Raw cell values: classify_line handles blanks/NaN itself.
                result = classify_line(
                    supplier=row["supplier"],
                    description=row["description"],
                    amount=row["amount_usd"],
                    model=model_id,
                )
            except RuntimeError as exc:  # e.g. ANTHROPIC_API_KEY not set
                progress_bar.empty()
                st.error(str(exc))
                st.stop()
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
        _set_results(pd.DataFrame(rows), model_label)

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
            width="stretch",
            hide_index=True,
            column_config={
                "line_id":     st.column_config.TextColumn("PO Line", width="small"),
                "supplier":    st.column_config.TextColumn("Supplier", width="medium"),
                "description": st.column_config.TextColumn("Description", width="large"),
                "amount_usd":  st.column_config.NumberColumn(
                    "Amount (USD)", format="$%.2f", width="small"
                ),
                "category":    st.column_config.TextColumn("Category", width="medium"),
                "confidence":  st.column_config.ProgressColumn(
                    "Confidence",
                    min_value=0,
                    max_value=100,
                    format="%.0f%%",
                    width="small",
                ),
                "reason":      st.column_config.TextColumn("Reason", width="large", help="One-sentence explanation from the model"),
            },
        )

        csv_bytes = _csv_safe(results_df).to_csv(index=False).encode()
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
                        st.markdown(f"**{_money(row['amount_usd'])}**")

                    # AI prediction row
                    is_err_row = row["category"] == "error"
                    conf_pct = int(round(row["confidence"] * 100))
                    conf_color = (
                        "⚠️" if is_err_row
                        else "🟢" if row["confidence"] >= 0.75
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
                            if is_err_row:
                                # There is no AI category to approve; a human must choose one.
                                st.caption("Classification failed — choose a category →")
                            elif st.button("Approve AI category", key=f"approve_{lid}"):
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
        # Only an actual human decision overrides the AI's category; unreviewed
        # lines always use the current results, never a stale stored value.
        def _final_category(r):
            review = reviews.get(r["line_id"])
            if review and review.get("human_reviewed"):
                return review["final_category"]
            return r["category"]

        merged["final_category"] = merged.apply(_final_category, axis=1)

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
        st.dataframe(band_df, hide_index=True, width="stretch")

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
        st.subheader("Model comparison — from saved batch runs (results/ folder)")

        model_slug = {m: m.replace("/", "-") for m in _MODEL_META}
        saved = {
            m: pd.read_csv(RESULTS_DIR / f"{model_slug[m]}.csv")
            for m in _MODEL_META
            if (RESULTS_DIR / f"{model_slug[m]}.csv").exists()
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
