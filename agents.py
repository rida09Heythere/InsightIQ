
import json
import os
import pandas as pd

from dotenv import load_dotenv
from google import genai
from google.genai import types
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

from analytics import (
    calculate_kpis,
    calculate_monthly_trends,
    generate_business_insights,
    detect_anomalies,
    make_json_safe,
)

load_dotenv()

# Preferred model first, followed by fallbacks.
MODELS = [
    "gemini-3.8-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
]


class InsightState(TypedDict, total=False):
    dataset_json: str
    question: str
    evidence: dict
    analysis: str
    recommendations: str
    review: str
    error: str


def get_client():
    """Create Gemini client with a finite request timeout."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing from .env")

    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=15000),
    )


def call_gemini(prompt, max_output_tokens=1800):
    """Try each configured model once, in priority order."""
    client = get_client()
    errors = []

    for model in MODELS:
        try:
            print(f"Trying {model}...", flush=True)

            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=max_output_tokens,
                ),
            )

            if response.text and response.text.strip():
                print(f"Success: {model}", flush=True)
                return response.text.strip()

            errors.append(f"{model}: empty response")

        except Exception as exc:
            errors.append(f"{model}: {exc}")
            print(f"Failed: {model} — {exc}", flush=True)

    raise RuntimeError(
        "All configured Gemini models failed:\n" + "\n".join(errors)
    )


def data_analyst(state: InsightState):
    """Calculate evidence locally using pandas."""
    try:
        records = json.loads(state["dataset_json"])
        df = pd.DataFrame(records)

        if df.empty:
            return {"error": "The supplied dataset is empty."}

        trends_result = calculate_monthly_trends(df)

        # Support either a DataFrame or a tuple from analytics.py.
        if isinstance(trends_result, tuple):
            trends_df = next(
                (
                    item for item in trends_result
                    if isinstance(item, pd.DataFrame)
                ),
                None,
            )
        else:
            trends_df = trends_result

        monthly_trends = (
            trends_df.to_dict(orient="records")
            if isinstance(trends_df, pd.DataFrame)
            else []
        )

        evidence = {
            "row_count": int(len(df)),
            "columns": list(df.columns),
            "kpis": calculate_kpis(df),
            "business_insights": generate_business_insights(df),
            "monthly_trends": monthly_trends,
            "anomaly_count": int(len(detect_anomalies(df))),
            "missing_values": int(df.isna().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum()),
            "question": state.get("question", ""),
        }

        return {"evidence": make_json_safe(evidence)}

    except Exception as exc:
        return {"error": f"Data analysis failed: {exc}"}


def insight_generator(state: InsightState):
    """Generate a complete business report using Gemini."""
    if state.get("error"):
        return {
            "analysis": "",
            "recommendations": "",
        }

    prompt = f"""
You are a business intelligence analyst.

USER REQUEST:
{state.get("question", "Analyze business performance.")}

CALCULATED DATA EVIDENCE:
{json.dumps(state.get("evidence", {}), indent=2, allow_nan=False)}

Write a complete, concise business report with these headings:

## Executive Summary
Summarize performance using verified metrics.

## Key Findings
Provide 3-5 findings supported by the evidence.

## Recommended Actions
Provide 3-5 practical, prioritized actions.

## Data Limitations
Mention relevant data limitations.

Rules:
- Complete all four sections.
- Do not invent figures or claim unsupported causation.
- Do not assume a currency unless established by the dataset.
- Distinguish measured facts from hypotheses.
"""

    try:
        report = call_gemini(prompt, max_output_tokens=2200)
        return {
            "analysis": report,
            "recommendations": report,
        }

    except Exception as exc:
        return {
            "error": f"Insight generation failed: {exc}",
            "analysis": "",
            "recommendations": "",
        }


def report_reviewer(state: InsightState):
    """Retain the generated report without an extra API request."""
    if state.get("error"):
        return {"review": state["error"]}

    draft = state.get("analysis", "")

    if not draft:
        return {"review": "No report was generated."}

    return {
        "review": draft,
        "recommendations": draft,
    }


# Build the LangGraph workflow.
builder = StateGraph(InsightState)

builder.add_node("data_analyst", data_analyst)
builder.add_node("insight_generator", insight_generator)
builder.add_node("report_reviewer", report_reviewer)

builder.add_edge(START, "data_analyst")
builder.add_edge("data_analyst", "insight_generator")
builder.add_edge("insight_generator", "report_reviewer")
builder.add_edge("report_reviewer", END)

insight_graph = builder.compile()


def run_insight_workflow(
    df: pd.DataFrame,
    question="Analyze business performance and recommend improvements",
):
    """Run the three-stage LangGraph workflow."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "run_insight_workflow expects a pandas DataFrame."
        )

    dataset_json = df.to_json(
        orient="records",
        date_format="iso",
    )

    result = insight_graph.invoke({
        "dataset_json": dataset_json,
        "question": question,
    })

    error = result.get("error", "")
    report = result.get("review", "")

    return {
        "evidence": result.get("evidence", {}),
        "analysis": result.get("analysis", ""),
        "report": report,
        "recommendations": result.get("recommendations", ""),
        "error": error,
    }
