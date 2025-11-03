"""
AI Dashboard Assistant (Offline + Generative AI)

Description:
A Streamlit app that accepts CSV/Excel uploads, performs automatic
data analysis and visualization, generates rule-based OR LLM-based
natural-language insights, and supports simple natural-language-style queries.
Fully offline — no external API calls.

Requirements:
pip install streamlit pandas numpy plotly scikit-learn transformers accelerate torch

Run:
streamlit run ai_dashboard_assistant_streamlit.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
from sklearn.impute import SimpleImputer
import plotly.express as px
import plotly.graph_objects as go
import torch
from transformers import pipeline

# ------------------------------------------------------------
# Streamlit setup
# ------------------------------------------------------------
st.set_page_config(page_title="AI Dashboard Assistant (Offline + GenAI)", layout="wide")
st.title("🧠 AI Dashboard Assistant (Offline + Generative AI)")
st.write("Upload a CSV/XLSX file to analyze it automatically. The assistant can generate insights using either rule-based logic or a local LLM (Phi-2).")

# ------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------

def read_file(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            return pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file type. Please upload a CSV or Excel file.")
            return None
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        return None


def safe_describe(df):
    num = df.select_dtypes(include=[np.number])
    cat = df.select_dtypes(include=['object', 'category', 'bool'])
    desc = {
        'shape': df.shape,
        'num_columns': list(num.columns),
        'cat_columns': list(cat.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'head': df.head(5).to_dict(orient='records')
    }
    return desc


def generate_rule_based_insights(df, max_insights=10):
    insights = []
    n, m = df.shape
    insights.append(f"The dataset contains {n} rows and {m} columns.")

    # Missing values
    miss = df.isnull().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    for col, val in miss.items():
        insights.append(f"Column '{col}' has {val} missing values ({val/n:.1%} of rows).")
        if len(insights) >= max_insights: return insights

    # Numeric columns
    numeric = df.select_dtypes(include=[np.number])
    if not numeric.empty:
        desc = numeric.describe().T
        high_var = desc[desc['std'] > desc['mean'] * 0.5]
        for idx in high_var.index[:3]:
            insights.append(f"Numeric column '{idx}' shows high variability (std={desc.loc[idx,'std']:.2f}).")
            if len(insights) >= max_insights: return insights

        # Correlations
        corr = numeric.corr()
        strong_pairs = []
        for i, c1 in enumerate(corr.columns):
            for j, c2 in enumerate(corr.columns):
                if j <= i: continue
                val = corr.iloc[i, j]
                if abs(val) >= 0.7:
                    strong_pairs.append((c1, c2, val))
        strong_pairs = sorted(strong_pairs, key=lambda x: -abs(x[2]))
        for c1, c2, val in strong_pairs[:3]:
            insights.append(f"Strong correlation ({val:.2f}) found between '{c1}' and '{c2}'.")
            if len(insights) >= max_insights: return insights

    # Categorical columns
    cat = df.select_dtypes(include=['object', 'category', 'bool'])
    for col in cat.columns[:3]:
        top = df[col].value_counts(dropna=True).nlargest(3)
        top_list = ", ".join([f"{idx}({cnt})" for idx, cnt in top.items()])
        insights.append(f"Column '{col}' top values: {top_list}.")
        if len(insights) >= max_insights: return insights

    if len(insights) == 1:
        insights.append("No major issues detected from quick heuristics.")
    return insights


def df_to_markdown_report(df, insights):
    md = ["# AI Dashboard Assistant Report\n"]
    md.append(f"**Rows:** {df.shape[0]}  ")
    md.append(f"**Columns:** {df.shape[1]}  \n")

    md.append("## Insights\n")
    for i, ins in enumerate(insights, 1):
        md.append(f"{i}. {ins}  \n")

    return "\n".join(md)


# ------------------------------------------------------------
# Load Local LLM (Phi-2)
# ------------------------------------------------------------

@st.cache_resource
def load_local_llm():
    st.info("🔄 Loading local Phi-2 model (this may take a few minutes on first run)...")
    generator = pipeline(
        "text-generation",
        model="microsoft/phi-2",
        torch_dtype=torch.float32,
        device_map="auto"
    )
    return generator

# ------------------------------------------------------------
# File Upload
# ------------------------------------------------------------
uploaded_file = st.file_uploader("Upload CSV or Excel file", type=['csv', 'xls', 'xlsx'])

if uploaded_file is not None:
    df = read_file(uploaded_file)
    if df is None:
        st.stop()

    st.sidebar.header("Data Overview")
    desc = safe_describe(df)
    st.sidebar.write(f"Shape: {desc['shape']}")
    st.sidebar.write(f"Numeric columns ({len(desc['num_columns'])}): {', '.join(desc['num_columns'][:10])}")
    st.sidebar.write(f"Categorical columns ({len(desc['cat_columns'])}): {', '.join(desc['cat_columns'][:10])}")

    # Show sample data
    st.subheader("Data Sample")
    st.dataframe(df.head(10))

    # ------------------------------------------------------------
    # Cleaning options
    # ------------------------------------------------------------
    st.sidebar.subheader("Quick Cleaning")
    fill_method = st.sidebar.selectbox("Fill numeric missing with", ['none', 'mean', 'median', 'zero'])
    drop_na_thresh = st.sidebar.slider("Drop columns with >% missing", 0, 100, 100)

    working = df.copy()
    if drop_na_thresh < 100:
        thresh = int((drop_na_thresh / 100.0) * len(working))
        working = working.dropna(axis=1, thresh=thresh)
    if fill_method != 'none':
        num_cols = working.select_dtypes(include=[np.number]).columns
        if fill_method == 'mean':
            working[num_cols] = working[num_cols].fillna(working[num_cols].mean())
        elif fill_method == 'median':
            working[num_cols] = working[num_cols].fillna(working[num_cols].median())
        elif fill_method == 'zero':
            working[num_cols] = working[num_cols].fillna(0)

    # ------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------
    st.subheader("Auto Visualizations")
    num_cols = working.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = working.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    col1, col2 = st.columns([2, 1])
    with col1:
        st.write("### Numeric Distributions")
        if num_cols:
            sel = st.selectbox("Choose numeric column for histogram", num_cols)
            fig = px.histogram(working, x=sel, nbins=30, marginal='box')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric columns found.")

    with col2:
        st.write("### Categorical Summary")
        if cat_cols:
            scel = st.selectbox("Choose categorical column for bar chart", cat_cols)
            top = working[scel].value_counts().nlargest(10).reset_index()
            top.columns = [scel, 'count']
            fig2 = px.bar(top, x=scel, y='count')
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No categorical columns found.")

    st.write("### Correlation Heatmap")
    if len(num_cols) >= 2:
        corr = working[num_cols].corr()
        fig3 = px.imshow(corr, text_auto=True)
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Need at least 2 numeric columns for correlation heatmap.")

    # ------------------------------------------------------------
    # Insights Section (Rule-based or Generative)
    # ------------------------------------------------------------
    st.subheader("AI Insights")

    mode = st.radio("Choose AI mode:", ["Rule-based (fast, offline)", "Generative (local LLM - Phi-2)"])

    if mode == "Rule-based (fast, offline)":
        insights = generate_rule_based_insights(working, max_insights=12)
        for i, ins in enumerate(insights, 1):
            st.markdown(f"**{i}.** {ins}")

    else:
        generator = load_local_llm()
        with st.spinner("Generating insights using local LLM (Phi-2)..."):
            rb_insights = generate_rule_based_insights(working, max_insights=8)
            prompt = f"""
            You are a data analyst AI. Given the dataset summary and rule-based insights below,
            produce a concise, natural-language report with 5-8 key insights.

            Dataset shape: {working.shape}
            Columns: {list(working.columns)}
            Sample rows:
            {working.head(5).to_string(index=False)}

            Rule-based insights:
            {', '.join(rb_insights)}
            """
            result = generator(prompt, max_new_tokens=300, temperature=0.7)
            text_output = result[0]['generated_text']
            st.markdown("### 🤖 Generated Insights (Phi-2)")
            st.write(text_output)

    # ------------------------------------------------------------
    # Report Export
    # ------------------------------------------------------------
    st.subheader("Export Report")
    insights_text = generate_rule_based_insights(working, max_insights=12)
    md = df_to_markdown_report(working, insights_text)
    st.download_button("Download Markdown Report", md, file_name='ai_dashboard_report.md')

    buf = io.BytesIO()
    working.to_csv(buf, index=False)
    st.download_button("Download Cleaned CSV", buf.getvalue(), file_name='cleaned_data.csv')

else:
    st.info("Upload a CSV or Excel file to begin.")
