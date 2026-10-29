
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

from rag import answer_question, ingest_pdf

from analytics import (
    load_dataset,
    profile_dataset,
    clean_dataset,
    numeric_summary,
    create_chart,
    analyze_data_question,
    execute_analysis_plan,
    calculate_kpis,
    calculate_monthly_trends,
    detect_anomalies,
    generate_business_insights,
    generate_ai_recommendations,
)

st.set_page_config(
    page_title="InsightIQ",
    page_icon="📊",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

st.title("InsightIQ")
st.caption("AI-powered document intelligence and business analytics")

with st.sidebar:
    st.header("Workspace")
    workspace = st.radio(
        "Choose a workspace",
        ["Document Q&A", "Business Analytics"],
    )

# ==================================================
# DOCUMENT Q&A
# ==================================================

if workspace == "Document Q&A":
    st.header("Document Q&A")
    st.write("Upload a PDF and ask questions about its contents.")

    uploaded_pdf = st.file_uploader(
        "Upload a PDF document",
        type=["pdf"],
        key="document_pdf",
    )

    if uploaded_pdf is not None:
        if st.button("Index PDF", key="index_pdf"):
            saved_path = UPLOAD_DIR / Path(uploaded_pdf.name).name

            try:
                with st.spinner("Extracting and indexing document..."):
                    saved_path.write_bytes(uploaded_pdf.getvalue())
                    result = ingest_pdf(saved_path)

                if result.startswith("Successfully indexed"):
                    st.success(result)
                else:
                    st.error(result)

            except Exception as exc:
                st.error(f"Could not index the PDF: {exc}")

    st.divider()
    st.subheader("Ask a question")

    question = st.text_input(
        "Your question",
        placeholder="e.g., Summarize this document",
        key="document_question",
    )

    if st.button("Ask InsightIQ", key="ask_document"):
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            with st.spinner("Searching documents and generating an answer..."):
                try:
                    answer = answer_question(question)
                    st.markdown("### Answer")
                    st.write(answer)
                except Exception as exc:
                    st.error(f"Could not answer the question: {exc}")

# ==================================================
# BUSINESS ANALYTICS
# ==================================================

elif workspace == "Business Analytics":
    st.header("Business Analytics")
    st.write(
        "Explore datasets, calculate KPIs, identify trends, "
        "detect anomalies, and generate business recommendations."
    )

    uploaded_file = st.file_uploader(
        "Upload a CSV or Excel file",
        type=["csv", "xlsx", "xls"],
        key="analytics_file",
    )

    if uploaded_file is not None:
        try:
            df = load_dataset(uploaded_file)

            if df.empty or len(df.columns) == 0:
                st.warning("The uploaded dataset is empty.")
                st.stop()

            st.session_state["analytics_df"] = df
            st.session_state["analytics_filename"] = uploaded_file.name

        except Exception as exc:
            st.error(f"Could not load the dataset: {exc}")
            st.stop()

    df = st.session_state.get("analytics_df")

    if df is None:
        st.info("Upload a CSV or Excel file to begin.")
        st.stop()

    st.subheader(
        f"Dataset Preview — "
        f"{st.session_state.get('analytics_filename', 'Dataset')}"
    )
    st.dataframe(df.head(20), use_container_width=True)

    # --------------------------------------------------
    # DATASET PROFILE
    # --------------------------------------------------

    profile = profile_dataset(df)

    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("Rows", f"{profile['rows']:,}")
    metric2.metric("Columns", f"{profile['columns']:,}")
    metric3.metric("Missing Values", f"{profile['missing_values']:,}")
    metric4.metric("Duplicate Rows", f"{profile['duplicate_rows']:,}")

    with st.expander("Data types and missing values"):
        details = pd.DataFrame({
            "Data Type": profile["data_types"],
            "Missing Values": profile["missing_by_column"],
        })
        st.dataframe(details, use_container_width=True)

    with st.expander("Dataset summary"):
        st.dataframe(profile["summary"], use_container_width=True)

    # --------------------------------------------------
    # NUMERIC SUMMARY
    # --------------------------------------------------

    st.subheader("Numeric Summary")
    summary = numeric_summary(df)

    if summary.empty:
        st.info("No numeric columns were found.")
    else:
        st.dataframe(summary, use_container_width=True)

    # --------------------------------------------------
    # KPI DASHBOARD
    # --------------------------------------------------

    st.divider()
    st.subheader("Key Performance Indicators")

    kpis = calculate_kpis(df)

    if "Notice" in kpis:
        st.info(kpis["Notice"])
    else:
        kpi_items = list(kpis.items())

        for start in range(0, len(kpi_items), 4):
            cols = st.columns(4)

            for col, (label, value) in zip(
                cols, kpi_items[start:start + 4]
            ):
                if value is None:
                    display_value = "N/A"
                elif isinstance(value, (int, float)):
                    display_value = (
                        f"{value:,.2f}%"
                        if "Margin" in label
                        else f"{value:,.2f}"
                    )
                else:
                    display_value = str(value)

                col.metric(label, display_value)

    # --------------------------------------------------
    # MONTHLY TREND ANALYSIS
    # --------------------------------------------------

    st.divider()
    st.subheader("Monthly Trend Analysis")

    date_columns = [
        column for column in df.columns
        if "date" in str(column).lower()
    ]
    numeric_columns = list(
        df.select_dtypes(include="number").columns
    )

    if date_columns and numeric_columns:
        trend_col1, trend_col2 = st.columns(2)

        with trend_col1:
            date_column = st.selectbox(
                "Date column",
                date_columns,
                key="trend_date",
            )

        with trend_col2:
            trend_metric = st.selectbox(
                "Metric to analyze",
                numeric_columns,
                key="trend_metric",
            )

        monthly, error = calculate_monthly_trends(
            df,
            date_column=date_column,
            metric_column=trend_metric,
        )

        if error:
            st.warning(error)
        elif not monthly.empty:
            st.dataframe(monthly, use_container_width=True)

            fig = px.line(
                monthly,
                x="Month",
                y=trend_metric,
                markers=True,
                title=f"Monthly {trend_metric}",
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(
            "Monthly trends require a date column and a numeric column."
        )

    # --------------------------------------------------
    # ANOMALY DETECTION
    # --------------------------------------------------

    st.divider()
    st.subheader("Anomaly Detection")

    if numeric_columns:
        anomaly_column = st.selectbox(
            "Choose a metric",
            numeric_columns,
            key="anomaly_metric",
        )

        if st.button("Detect Anomalies"):
            anomalies, error = detect_anomalies(df, anomaly_column)

            if error:
                st.info(error)
            elif anomalies.empty:
                st.success(
                    "No unusual values were detected using the IQR method."
                )
            else:
                st.warning(
                    f"Found {len(anomalies)} unusual value(s). "
                    "These statistical flags are not necessarily errors."
                )
                st.dataframe(anomalies, use_container_width=True)
    else:
        st.info("Anomaly detection requires numeric columns.")

    # --------------------------------------------------
    # RULE-BASED BUSINESS INSIGHTS
    # --------------------------------------------------

    st.divider()
    st.subheader("Business Insights")

    if st.button("Generate Business Insights"):
        insights = generate_business_insights(df)

        for insight in insights:
            st.markdown(f"- {insight}")

    # --------------------------------------------------
    # AI BUSINESS RECOMMENDATIONS
    # --------------------------------------------------

    st.divider()
    st.subheader("AI Business Recommendations")

    st.caption(
        "Generate actionable recommendations based on calculated "
        "KPIs, monthly trends, and detected anomalies."
    )

    if st.button(
        "Generate AI Recommendations",
        key="ai_recommendations",
    ):
        with st.spinner("Analyzing business performance..."):
            try:
                recommendations = generate_ai_recommendations(df)

                if (
                    recommendations.startswith(
                        "Could not generate business recommendations"
                    )
                    or recommendations.startswith(
                        "Recommendations unavailable"
                    )
                ):
                    st.error(recommendations)
                else:
                    st.markdown(recommendations)

            except Exception as exc:
                st.error(
                    f"Could not generate recommendations: {exc}"
                )

    # --------------------------------------------------
    # INTERACTIVE CHARTS
    # --------------------------------------------------

    st.divider()
    st.subheader("Interactive Charts")

    chart_type = st.selectbox(
        "Chart type",
        ["Bar", "Line", "Scatter", "Histogram"],
    )

    columns = list(df.columns)
    x_column = st.selectbox("X-axis / Category", columns)

    y_column = None

    if chart_type in ["Bar", "Line", "Scatter"]:
        y_column = st.selectbox(
            "Y-axis / Value",
            columns,
            index=1 if len(columns) > 1 else 0,
        )

    if st.button("Generate Chart"):
        try:
            figure = create_chart(
                df,
                chart_type,
                x_column,
                y_column,
            )
            st.plotly_chart(figure, use_container_width=True)
        except Exception as exc:
            st.error(f"Could not create chart: {exc}")

    # --------------------------------------------------
    # NATURAL-LANGUAGE DATA ANALYSIS
    # --------------------------------------------------

    st.divider()
    st.subheader("Ask InsightIQ About Your Data")

    st.caption(
        "Ask questions about totals, averages, rankings, grouped "
        "metrics, missing values, duplicates, or correlations."
    )

    data_question = st.text_input(
        "Business question",
        placeholder="e.g., Which category has the highest sales?",
        key="data_question",
    )

    if st.button("Analyze Data", key="analyze_dataset"):
        if not data_question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Analyzing your question..."):
                try:
                    plan = analyze_data_question(df, data_question)

                    if plan.startswith("Analysis planning failed:"):
                        st.error(plan)
                    else:
                        result = execute_analysis_plan(df, plan)
                        st.markdown("### Analysis Result")
                        st.write(result)

                except Exception as exc:
                    st.error(f"Analysis failed: {exc}")

    # --------------------------------------------------
    # DATA CLEANING AND DOWNLOAD
    # --------------------------------------------------

    st.divider()
    st.subheader("Clean Dataset")

    cleaned_df = clean_dataset(df)
    removed_rows = len(df) - len(cleaned_df)

    st.write(
        f"Duplicate rows that will be removed: **{removed_rows:,}**"
    )
    st.dataframe(cleaned_df.head(20), use_container_width=True)

    csv_data = cleaned_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Cleaned CSV",
        data=csv_data,
        file_name="insightiq_cleaned.csv",
        mime="text/csv",
    )
