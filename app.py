import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
import torch

# -------------------------------
# Page Setup
# -------------------------------
st.set_page_config(page_title="AI Dashboard Assistant", layout="wide")
st.title("📊 AI Dashboard Assistant (Offline + Generative + Charts)")
st.info("💡 Upload a dataset, view automatic charts, and get AI-generated insights — fully offline.")

# -------------------------------
# Load Local LLM (DistilGPT-2)
# -------------------------------
@st.cache_resource
def load_local_llm():
    model_name = "distilgpt2"
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
# Utility: Rule-Based Insights
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
# File Upload
# -------------------------------
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")

    # Dataset preview
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    # Columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # -------------------------------
    # Basic Visualizations
    # -------------------------------
    st.subheader("📊 Visual Analysis Section")
    plot_type = st.selectbox(
        "Select the type of plot:",
        [
            "Histogram (Numeric)",
            "Pie Chart (Categorical)",
            "Scatter Plot",
            "Box Plot",
            "Correlation Heatmap"
        ]
    )

    # 1️⃣ Histogram
    if plot_type == "Histogram (Numeric)":
        if numeric_cols:
            col = st.selectbox("Select a numeric column:", numeric_cols)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(df[col].dropna(), bins=20, color="#1f77b4", alpha=0.8)
            ax.set_title(f"Distribution of {col}")
            st.pyplot(fig)
        else:
            st.warning("No numeric columns available.")

    # 2️⃣ Pie Chart
    elif plot_type == "Pie Chart (Categorical)":
        if categorical_cols:
            col = st.selectbox("Select a categorical column:", categorical_cols)
            fig, ax = plt.subplots(figsize=(5, 5))
            df[col].value_counts().head(6).plot.pie(
                autopct="%1.1f%%", ax=ax, startangle=90, colors=plt.cm.Paired.colors
            )
            ax.set_ylabel("")
            ax.set_title(f"Distribution of {col}")
            st.pyplot(fig)
        else:
            st.warning("No categorical columns available.")

    # 3️⃣ Scatter Plot
    elif plot_type == "Scatter Plot":
        if len(numeric_cols) >= 2:
            x_col = st.selectbox("Select X-axis:", numeric_cols, key="x")
            y_col = st.selectbox("Select Y-axis:", numeric_cols, key="y")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(df[x_col], df[y_col], alpha=0.7, color="#ff7f0e")
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f"Scatter Plot: {x_col} vs {y_col}")
            st.pyplot(fig)
        else:
            st.warning("Need at least two numeric columns for scatter plot.")

    # 4️⃣ Box Plot
    elif plot_type == "Box Plot":
        if numeric_cols and categorical_cols:
            num_col = st.selectbox("Select numeric column:", numeric_cols, key="num_box")
            cat_col = st.selectbox("Select categorical column:", categorical_cols, key="cat_box")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(x=cat_col, y=num_col, data=df, ax=ax)
            ax.set_title(f"Box Plot of {num_col} by {cat_col}")
            st.pyplot(fig)
        else:
            st.warning("Need both numeric and categorical columns for box plot.")

    # 5️⃣ Correlation Heatmap
    elif plot_type == "Correlation Heatmap":
        if len(numeric_cols) > 1:
            fig, ax = plt.subplots(figsize=(6, 4))
            corr = df[numeric_cols].corr()
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            ax.set_title("Correlation Heatmap")
            st.pyplot(fig)
        else:
            st.warning("Not enough numeric columns for correlation heatmap.")

    # -------------------------------
    # AI Insights Mode
    # -------------------------------
    st.subheader("🧠 AI Insights Mode")
    mode = st.radio(
        "Choose AI mode:",
        ["Rule-based (fast, offline)", "Generative (local LLM - DistilGPT2)"]
    )

    # Rule-based mode
    if mode == "Rule-based (fast, offline)":
        with st.spinner("Generating rule-based insights..."):
            insights = generate_rule_based_insights(df, max_insights=12)
            st.markdown("### ⚙️ Rule-based Insights")
            for i, ins in enumerate(insights, 1):
                st.markdown(f"**{i}.** {ins}")

    # Generative mode
    elif "Generative" in mode:
        if local_generator is None:
            st.error("❌ Generative Mode not available in this environment. Please run locally.")
        else:
            with st.spinner("Generating smart insights..."):
                try:
                    base_insights = generate_rule_based_insights(df)
                    prompt = (
                        "Summarize key patterns and relationships in this dataset:\n\n"
                        + "\n".join(base_insights)
                        + "\n\nHere’s a small sample of the data:\n"
                        + df.head(3).to_string()
                    )

                    prompt = prompt[:800]
                    result = local_generator(
                        prompt,
                        max_new_tokens=120,
                        pad_token_id=50256,
                        temperature=0.7,
                        do_sample=True,
                    )

                    text_output = result[0]["generated_text"]
                    st.markdown("### 🤖 Generated Insights")
                    st.write(text_output)

                except IndexError:
                    st.error("⚠️ Generation failed: prompt too long or model input overflow.")
                except Exception as e:
                    st.error(f"⚠️ Something went wrong: {e}")

else:
    st.warning("⬆️ Please upload a CSV file to get started.")

