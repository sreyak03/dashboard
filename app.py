import streamlit as st
import pandas as pd
import numpy as np
from transformers import pipeline
import torch

# -------------------------------
# Page Setup
# -------------------------------
st.set_page_config(page_title="AI Dashboard Assistant", layout="wide")
st.title("📊 AI Dashboard Assistant (Offline + Generative)")

st.info("💡 Upload a dataset and choose between Rule-based or Generative AI mode for insights.")

# -------------------------------
# Load Local LLM (Phi-2)
# -------------------------------
@st.cache_resource
def load_local_llm():
    model_name = "microsoft/phi-2"
    generator = pipeline(
        "text-generation",
        model=model_name,
        torch_dtype=torch.float32,
        device_map="auto"
    )
    return generator

local_generator = load_local_llm()

# -------------------------------
# Utility: Generate Rule-Based Insights
# -------------------------------
def generate_rule_based_insights(df, max_insights=10):
    insights = []

    if df.empty:
        return ["No data available for analysis."]

    # Basic shape
    insights.append(f"The dataset contains **{df.shape[0]} rows** and **{df.shape[1]} columns**.")

    # Missing values
    missing = df.isnull().sum().sum()
    if missing > 0:
        insights.append(f"There are **{missing} missing values** in the dataset.")
    else:
        insights.append("There are **no missing values** in the dataset.")

    # Numeric summary
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        mean_vals = df[num_cols].mean().round(2)
        for col in mean_vals.head(max_insights).index:
            insights.append(f"The average value of **{col}** is approximately **{mean_vals[col]}**.")
    else:
        insights.append("The dataset has no numeric columns for statistical summary.")

    # Categorical columns
    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    if len(cat_cols) > 0:
        for col in cat_cols[:max_insights]:
            top_value = df[col].mode()[0]
            insights.append(f"The most common value in **{col}** is **{top_value}**.")
    else:
        insights.append("The dataset has no categorical columns.")

    return insights[:max_insights]

# -------------------------------
# File Upload Section
# -------------------------------
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("🧠 AI Insights Mode")
    mode = st.radio(
        "Choose AI mode:",
        ["Rule-based (fast, offline)", "Generative (local LLM - Phi-2)"]
    )

    # -------------------------------
    # Rule-based Mode
    # -------------------------------
    if mode == "Rule-based (fast, offline)":
        with st.spinner("Generating rule-based insights..."):
            insights = generate_rule_based_insights(df, max_insights=12)
            st.markdown("### ⚙️ Rule-based Insights")
            for i, ins in enumerate(insights, 1):
                st.markdown(f"**{i}.** {ins}")

    # -------------------------------
    # Generative Mode (Local LLM)
    # -------------------------------
    elif mode == "Generative (local LLM - Phi-2)":
        with st.spinner("Generating insights using Phi-2..."):
            # Optionally ground the model with rule-based facts
            base_insights = generate_rule_based_insights(df)
            prompt = f"""
            You are an expert AI data analyst.
            Analyze the following dataset and summarize insights naturally.

            Dataset shape: {df.shape}
            Columns: {list(df.columns)}

            Quick facts:
            {chr(10).join(base_insights)}

            Sample data:
            {df.head(5).to_string(index=False)}
            """

            result = local_generator(prompt, max_new_tokens=300, temperature=0.7)
            text_output = result[0]['generated_text']

            st.markdown("### 🤖 Generated Insights (Phi-2)")
            st.write(text_output)

else:
    st.warning("⬆️ Please upload a CSV file to get started.")

