
import json
import os
import re
import math
import time

import pandas as pd
import plotly.express as px


# ==================================================
# DATASET LOADING
# ==================================================

def load_dataset(file):
    filename = file.name.lower()

    if filename.endswith(".csv"):
        return pd.read_csv(file)

    if filename.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)

    raise ValueError("Please upload a CSV or Excel file.")


# ==================================================
# DATASET PROFILING
# ==================================================

def profile_dataset(df):
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": list(df.columns),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "data_types": df.dtypes.astype(str).to_dict(),
        "missing_by_column": df.isna().sum().to_dict(),
        "summary": df.describe(include="all").transpose(),
    }


# ==================================================
# DATA CLEANING
# ==================================================

def clean_dataset(df):
    cleaned = df.copy()
    cleaned.columns = [str(column).strip() for column in cleaned.columns]
    cleaned = cleaned.drop_duplicates()
    return cleaned


# ==================================================
# NUMERIC SUMMARY
# ==================================================

def numeric_summary(df):
    numeric = df.select_dtypes(include="number")

    if numeric.empty:
        return pd.DataFrame()

    return numeric.describe().transpose()


# ==================================================
# INTERACTIVE CHARTS
# ==================================================

def create_chart(df, chart_type, x_column, y_column=None):
    if x_column not in df.columns:
        raise ValueError("The selected X column does not exist.")

    if y_column is not None and y_column not in df.columns:
        raise ValueError("The selected Y column does not exist.")

    if chart_type == "Bar":
        return px.bar(df, x=x_column, y=y_column)

    if chart_type == "Line":
        return px.line(df, x=x_column, y=y_column)

    if chart_type == "Scatter":
        if y_column is None:
            raise ValueError("Scatter charts require X and Y columns.")
        return px.scatter(df, x=x_column, y=y_column)

    if chart_type == "Histogram":
        return px.histogram(df, x=x_column)

    raise ValueError("Unsupported chart type.")


# ==================================================
# NATURAL-LANGUAGE ANALYSIS PLANNER
# ==================================================

def analyze_data_question(df, question):
    from dotenv import load_dotenv
    from google import genai

    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return "Analysis planning failed: GEMINI_API_KEY is missing."

    client = genai.Client(api_key=api_key)

    schema = {
        "columns": list(df.columns),
        "data_types": df.dtypes.astype(str).to_dict(),
        "sample_rows": df.head(5).fillna("").to_dict(orient="records"),
    }

    prompt = f"""
You are InsightIQ, a business data analyst.

Dataset schema and sample:
{json.dumps(schema, default=str)}

User question:
{question}

Return ONLY valid JSON without Markdown fences.

Choose one operation:
row_count, column_summary, numeric_summary, group_by,
correlation, missing_values, duplicates, highest_group,
lowest_group, total, average, minimum, maximum,
value_counts, top_values, filter_rows, date_trend

Use this exact structure:
{{
  "operation": "operation_name",
  "column": null,
  "group_column": null,
  "metric": null,
  "filter_column": null,
  "filter_value": null,
  "top_n": 5
}}

Rules:
- Use only existing column names.
- Highest/lowest category questions use highest_group/lowest_group.
- Sum questions use total.
- Average questions use average.
- Category counts use value_counts.
- Top N categories use top_values.
- Filtering uses filter_rows only when the condition is clear.
- Time trends use date_trend when a date column exists.
- Group calculations use group_column for the category and metric
  for the numeric column.
- Never invent column names.
- If the question cannot be mapped safely, use numeric_summary.
"""

    models = ["gemini-3.5-flash", "gemini-3.8-flash"]
    last_error = None

    for model in models:
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                )

                if response.text:
                    return response.text.strip()

                last_error = "The AI returned an empty response."

            except Exception as exc:
                last_error = str(exc)
                message = last_error.lower()

                transient = any(term in message for term in [
                    "503",
                    "unavailable",
                    "429",
                    "resource_exhausted",
                    "500",
                    "internal server error",
                    "deadline exceeded",
                ])

                if transient and attempt == 0:
                    time.sleep(2)
                    continue

                break

    return f"Analysis planning failed: {last_error}"


# ==================================================
# SAFE PANDAS CALCULATION ENGINE
# ==================================================

def execute_analysis_plan(df, plan_text):
    try:
        text = plan_text.strip()

        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

        plan = json.loads(text)

        operation = plan.get("operation")
        column = plan.get("column")
        group_column = plan.get("group_column")
        metric = plan.get("metric")
        filter_column = plan.get("filter_column")
        filter_value = plan.get("filter_value")

        try:
            top_n = max(1, min(int(plan.get("top_n", 5)), 50))
        except (TypeError, ValueError):
            top_n = 5

        def require_column(name):
            if not isinstance(name, str) or name not in df.columns:
                raise ValueError(
                    f"Column {name!r} does not exist in the dataset."
                )

        def require_numeric(name):
            require_column(name)
            if not pd.api.types.is_numeric_dtype(df[name]):
                raise ValueError(f"Column {name!r} must be numeric.")

        def format_number(value):
            if pd.isna(value):
                return "Not available"

            if isinstance(value, (int, float)):
                return f"{value:,.2f}"

            return str(value)

        if operation == "row_count":
            return f"The dataset contains {len(df):,} rows."

        if operation == "missing_values":
            result = df.isna().sum()
            return (
                f"Total missing values: {int(result.sum()):,}\n\n"
                + result.to_string()
            )

        if operation == "duplicates":
            return f"Duplicate rows: {int(df.duplicated().sum()):,}"

        if operation == "numeric_summary":
            result = numeric_summary(df)
            return (
                "No numeric columns were found."
                if result.empty
                else result.round(2).to_string()
            )

        if operation == "column_summary":
            require_column(column)
            return df[column].describe().to_string()

        if operation in {"total", "average", "minimum", "maximum"}:
            require_numeric(column)
            series = pd.to_numeric(df[column], errors="coerce").dropna()

            if series.empty:
                return f"Column {column!r} has no valid numeric values."

            calculations = {
                "total": ("Total", series.sum()),
                "average": ("Average", series.mean()),
                "minimum": ("Minimum", series.min()),
                "maximum": ("Maximum", series.max()),
            }

            label, value = calculations[operation]
            return f"{label} for {column}: {format_number(value)}"

        if operation == "correlation":
            numeric = df.select_dtypes(include="number")

            if numeric.shape[1] < 2:
                return "At least two numeric columns are required."

            return numeric.corr().round(3).to_string()

        if operation == "value_counts":
            require_column(column)
            return (
                df[column]
                .value_counts(dropna=False)
                .head(top_n)
                .to_string()
            )

        if operation == "top_values":
            require_column(group_column)
            require_numeric(metric)

            result = (
                df.groupby(group_column, dropna=False)[metric]
                .sum()
                .sort_values(ascending=False)
                .head(top_n)
            )

            return (
                f"Top {top_n} {group_column} values by total {metric}:\n\n"
                + result.to_string()
            )

        if operation in {"group_by", "highest_group", "lowest_group"}:
            require_column(group_column)
            require_numeric(metric)

            grouped = (
                df.groupby(group_column, dropna=False)[metric]
                .sum()
                .sort_values(ascending=False)
            )

            if grouped.empty:
                return "No groups were available for comparison."

            if operation == "highest_group":
                return (
                    f"Highest {group_column} by total {metric}: "
                    f"{grouped.index[0]} "
                    f"({format_number(grouped.iloc[0])})"
                )

            if operation == "lowest_group":
                return (
                    f"Lowest {group_column} by total {metric}: "
                    f"{grouped.index[-1]} "
                    f"({format_number(grouped.iloc[-1])})"
                )

            result = df.groupby(
                group_column, dropna=False
            )[metric].agg(["count", "sum", "mean", "min", "max"])

            return result.round(2).to_string()

        if operation == "filter_rows":
            require_column(filter_column)

            if filter_value is None:
                return "The filter value was not specified."

            matches = df[
                df[filter_column].astype(str).str.casefold()
                == str(filter_value).casefold()
            ]

            if matches.empty:
                return "No rows matched the requested filter."

            return (
                f"Matching rows: {len(matches):,}\n\n"
                + matches.head(50).to_string(index=False)
            )

        if operation == "date_trend":
            require_column(group_column)
            require_numeric(metric)

            dates = pd.to_datetime(df[group_column], errors="coerce")
            values = pd.to_numeric(df[metric], errors="coerce")

            working = pd.DataFrame({
                "Date": dates,
                "Value": values,
            }).dropna()

            if working.empty:
                return "No valid date and numeric values were found."

            working["Month"] = working["Date"].dt.to_period("M").astype(str)

            return (
                working.groupby("Month")["Value"]
                .sum()
                .to_string()
            )

        return (
            "I couldn't map that question to a supported calculation. "
            "Try asking for a total, average, highest category, "
            "missing values, duplicates, or correlation."
        )

    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        return f"Could not execute the analysis: {exc}"


# ==================================================
# KPI ANALYSIS
# ==================================================

def calculate_kpis(df):
    kpis = {}

    normalized = {
        str(column).strip().lower().replace("_", " "): column
        for column in df.columns
    }

    sales_column = next(
        (
            normalized[name]
            for name in ["sales", "revenue", "total sales", "amount"]
            if name in normalized
        ),
        None,
    )

    profit_column = next(
        (
            normalized[name]
            for name in ["profit", "net profit", "gross profit"]
            if name in normalized
        ),
        None,
    )

    quantity_column = next(
        (
            normalized[name]
            for name in ["quantity", "units sold", "units"]
            if name in normalized
        ),
        None,
    )

    if sales_column:
        sales = pd.to_numeric(df[sales_column], errors="coerce").dropna()

        if not sales.empty:
            kpis["Total Revenue"] = float(sales.sum())
            kpis["Average Sales"] = float(sales.mean())

    if profit_column:
        profit = pd.to_numeric(df[profit_column], errors="coerce").dropna()

        if not profit.empty:
            kpis["Total Profit"] = float(profit.sum())
            kpis["Average Profit"] = float(profit.mean())

    if sales_column and profit_column:
        sales = pd.to_numeric(df[sales_column], errors="coerce")
        profit = pd.to_numeric(df[profit_column], errors="coerce")

        valid = sales.notna() & profit.notna()
        total_sales = sales[valid].sum()
        total_profit = profit[valid].sum()

        kpis["Profit Margin (%)"] = (
            float(total_profit / total_sales * 100)
            if total_sales != 0
            else None
        )

    if quantity_column:
        quantity = pd.to_numeric(
            df[quantity_column], errors="coerce"
        ).dropna()

        if not quantity.empty:
            kpis["Total Quantity"] = float(quantity.sum())

    if not kpis:
        kpis["Notice"] = (
            "No standard sales, revenue, profit, or quantity "
            "columns were detected."
        )

    return kpis


# ==================================================
# MONTHLY TREND ANALYSIS
# ==================================================

def calculate_monthly_trends(df, date_column=None, metric_column=None):
    if date_column is None:
        date_column = next(
            (
                column for column in df.columns
                if str(column).strip().lower()
                in ["date", "order date", "transaction date"]
            ),
            None,
        )

    if metric_column is None:
        metric_column = next(
            (
                column for column in df.columns
                if str(column).strip().lower()
                in ["sales", "revenue", "total sales", "profit"]
            ),
            None,
        )

    if date_column not in df.columns or metric_column not in df.columns:
        return pd.DataFrame(), (
            "A date column and a numeric metric column are required."
        )

    dates = pd.to_datetime(df[date_column], errors="coerce")
    values = pd.to_numeric(df[metric_column], errors="coerce")

    working = pd.DataFrame({
        "Date": dates,
        "Value": values,
    }).dropna()

    if working.empty:
        return pd.DataFrame(), "No valid date and metric values were found."

    working["Month"] = working["Date"].dt.to_period("M").astype(str)

    monthly = (
        working.groupby("Month", as_index=False)["Value"]
        .sum()
        .rename(columns={"Value": metric_column})
    )

    monthly["Growth (%)"] = (
        monthly[metric_column].pct_change() * 100
    ).replace([float("inf"), float("-inf")], float("nan")).round(2)

    return monthly, None


# ==================================================
# ANOMALY DETECTION
# ==================================================

def detect_anomalies(df, column=None):
    if column is None:
        column = next(
            (
                name for name in df.select_dtypes(include="number").columns
                if str(name).strip().lower()
                in ["sales", "revenue", "profit", "amount"]
            ),
            None,
        )

    if column is None or column not in df.columns:
        return pd.DataFrame(), "No suitable numeric metric column was found."

    values = pd.to_numeric(df[column], errors="coerce")
    valid = values.dropna()

    if len(valid) < 4:
        return pd.DataFrame(), "At least four valid numeric values are needed."

    q1 = valid.quantile(0.25)
    q3 = valid.quantile(0.75)
    iqr = q3 - q1

    if iqr == 0:
        return pd.DataFrame(), (
            "Anomalies could not be identified reliably because the IQR is zero."
        )

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    mask = (values < lower_bound) | (values > upper_bound)
    anomalies = df.loc[mask].copy()
    anomalies["Anomaly Value"] = values.loc[mask]
    anomalies["Anomaly Type"] = values.loc[mask].apply(
        lambda value: (
            "Unusually high" if value > upper_bound else "Unusually low"
        )
    )

    return anomalies, None


# ==================================================
# RULE-BASED BUSINESS INSIGHTS
# ==================================================

def generate_business_insights(df):
    insights = []
    kpis = calculate_kpis(df)

    if "Total Revenue" in kpis:
        insights.append(
            f"Total revenue across the dataset is "
            f"{kpis['Total Revenue']:,.2f}."
        )

    if "Total Profit" in kpis:
        insights.append(
            f"Total profit across the dataset is "
            f"{kpis['Total Profit']:,.2f}."
        )

    if kpis.get("Profit Margin (%)") is not None:
        insights.append(
            f"Overall profit margin is "
            f"{kpis['Profit Margin (%)']:.2f}%."
        )

    normalized = {
        str(column).strip().lower().replace("_", " "): column
        for column in df.columns
    }

    sales_column = next(
        (
            normalized[name]
            for name in ["sales", "revenue", "total sales", "amount"]
            if name in normalized
        ),
        None,
    )

    profit_column = next(
        (
            normalized[name]
            for name in ["profit", "net profit", "gross profit"]
            if name in normalized
        ),
        None,
    )

    for metric_column, metric_label in [
        (sales_column, "sales"),
        (profit_column, "profit"),
    ]:
        if metric_column is None:
            continue

        values = pd.to_numeric(df[metric_column], errors="coerce").dropna()

        if values.empty:
            continue

        insights.append(
            f"Highest individual {metric_label} value is "
            f"{values.max():,.2f}; lowest is {values.min():,.2f}."
        )

    if not insights:
        insights.append(
            "No standard business metrics were detected. "
            "Review the dataset's column names."
        )

    return insights


# ==================================================
# JSON-SAFE CONVERSION
# ==================================================

def make_json_safe(value):
    import numpy as np

    if isinstance(value, dict):
        return {
            str(key): make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(item) for item in value]

    if value is None or value is pd.NA or value is pd.NaT:
        return None

    if isinstance(value, np.ndarray):
        return make_json_safe(value.tolist())

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None

    if isinstance(value, (int, str, bool)):
        return value

    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()

    if isinstance(value, (pd.Period,)):
        return str(value)

    if isinstance(value, (pd.Series,)):
        return make_json_safe(value.to_dict())

    if isinstance(value, (pd.DataFrame,)):
        return make_json_safe(value.to_dict(orient="records"))

    try:
        missing = pd.isna(value)
        if isinstance(missing, bool) and missing:
            return None
    except (TypeError, ValueError):
        pass

    return str(value)


# ==================================================
# AI-POWERED BUSINESS RECOMMENDATIONS
# ==================================================

def generate_ai_recommendations(df):
    from dotenv import load_dotenv
    from google import genai

    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return "Recommendations unavailable: GEMINI_API_KEY is missing."

    kpis = calculate_kpis(df)
    insights = generate_business_insights(df)

    monthly, trend_error = calculate_monthly_trends(df)

    if trend_error is None and not monthly.empty:
        trend_data = monthly.to_dict(orient="records")
    else:
        trend_data = {
            "notice": trend_error or "No monthly trend data available."
        }

    numeric_columns = list(df.select_dtypes(include="number").columns)
    anomaly_data = {}

    for column in numeric_columns[:8]:
        anomalies, error = detect_anomalies(df, column)

        if error is None and not anomalies.empty:
            anomaly_data[column] = {
                "count": int(len(anomalies)),
                "examples": anomalies["Anomaly Value"].head(5).tolist(),
                "types": anomalies["Anomaly Type"].head(5).tolist(),
            }

    evidence = make_json_safe({
        "row_count": len(df),
        "columns": list(df.columns),
        "calculated_kpis": kpis,
        "calculated_insights": insights,
        "monthly_trends": trend_data,
        "detected_anomalies": anomaly_data,
    })

    # allow_nan=False verifies that no NaN or Infinity reaches Gemini.
    evidence_json = json.dumps(
        evidence,
        default=str,
        allow_nan=False,
    )

    client = genai.Client(api_key=api_key)

    prompt = f"""
You are InsightIQ, a business intelligence consultant.

Analyze the following calculated dataset evidence and provide
practical, evidence-based business recommendations.

EVIDENCE:
{evidence_json}

Requirements:
1. Explain the most important findings in plain language.
2. Provide 3 to 5 prioritized recommendations when evidence supports them.
3. Explain the evidence and suggested action for each recommendation.
4. Distinguish observed facts from possible explanations.
5. Do not invent metrics, trends, causes, targets, or business context.
6. If evidence is insufficient, explain what additional data is needed.
7. Treat statistical anomalies as items to investigate, not automatically errors.
8. Do not claim growth unless the trend data supports it.
9. Do not repeat every KPI unnecessarily.
10. Keep the answer concise and practical.

Structure:
## Executive Summary
## Key Findings
## Recommended Actions
## Data Limitations
"""

    models = ["gemini-3.5-flash", "gemini-3.8-flash"]
    last_error = None

    for model in models:
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt,
                )

                if response.text:
                    return response.text.strip()

                last_error = "The AI returned an empty response."

            except Exception as exc:
                last_error = str(exc)
                message = last_error.lower()

                transient = any(term in message for term in [
                    "503",
                    "unavailable",
                    "429",
                    "resource_exhausted",
                    "500",
                    "internal server error",
                    "deadline exceeded",
                ])

                if transient and attempt == 0:
                    time.sleep(2)
                    continue

                break

    return (
        "Could not generate business recommendations at this time. "
        f"Please try again later. Details: {last_error}"
    )
