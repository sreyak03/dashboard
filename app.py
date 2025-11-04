import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import pipeline
import torch

# -------------------------------
# Page Setup
# -------------------------------
st.set_page_config(page_title="AI Dashboard Assistant", layout="wide")
st.title("📊 AI Dashboard Assistant (Offline + Generative + Charts)")

st.info("💡 Upload a dataset, view automatic charts, and get AI-generated insights — fully offline.")

# -------------------------------
# Load Local LLM (Phi-2)
# -------------------------------
from transformers import pipeline
import torch

@st.cache_resource
def load_local_llm():
    model_name = "distilgpt2"  # ✅ very small, safe for Streamlit Cloud

    try:
        st.info(f"🚀 Loading model: {model_name} ...")
        generator = pipeline(
            "text-generation",
            model=model_name,
            device_map="auto"
        )
        st.success("✅ Model loaded successfully!")
        return generator
    except Exception as e:
        st.warning(f"⚠️ Generative AI disabled: {e}")
        return None

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

    # Dataset preview
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    # Column analysis layout
    st.subheader("📊 Basic Charts and Visual Analysis")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    col1, col2 = st.columns(2)

    # -------------------------------
    # Numeric column histograms
    # -------------------------------
    with col1:
        if numeric_cols:
            selected_num = st.selectbox("Select a numeric column for histogram:", numeric_cols)
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.hist(df[selected_num].dropna(), bins=20, color="#1f77b4", alpha=0.8)
            ax.set_title(f"Distribution of {selected_num}")
            st.pyplot(fig)
        else:
            st.info("No numeric columns available for histograms.")

    # -------------------------------
    # Categorical column pie charts
    # -------------------------------
    with col2:
        if categorical_cols:
            selected_cat = st.selectbox("Select a categorical column for pie chart:", categorical_cols)
            fig, ax = plt.subplots(figsize=(4, 4))
            df[selected_cat].value_counts().head(6).plot.pie(
                autopct="%1.1f%%", ax=ax, startangle=90, colors=plt.cm.Paired.colors
            )
            ax.set_ylabel("")
            ax.set_title(f"Distribution of {selected_cat}")
            st.pyplot(fig)
        else:
            st.info("No categorical columns available for pie charts.")

    # -------------------------------
    # AI Insights Mode
    # -------------------------------
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
    elif mode == "Generative (local LLM - Phi-2)" or mode == "Generative Insights":
    if local_generator is None:
        st.error("❌ Generative Mode not available in this environment. Please run locally.")
    else:
        with st.spinner("Generating smart insights..."):
            base_insights = generate_rule_based_insights(df)
            prompt = (
                "Analyze this dataset and describe key insights:\n\n" +
                "\n".join(base_insights) +
                "\n\nSample Data:\n" +
                df.head().to_string()
            )
            result = local_generator(prompt, max_new_tokens=200)
            text_output = result[0]["generated_text"]
            st.markdown("### 🤖 Generated Insights")
            st.write(text_output)
else:
    st.warning("⬆️ Please upload a CSV file to get started.")
