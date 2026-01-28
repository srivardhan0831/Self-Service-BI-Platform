import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Self-Service BI Platform",
    layout="wide"
)

st.title("📊 Self-Service Business Intelligence Platform")

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload CSV or Excel File",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:

    # ---------------- LOAD DATA ----------------
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.head())

    # ---------------- BASIC CLEANING ----------------
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ---------------- DATETIME CONVERSION (FIRST) ----------------
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                df[col] = pd.to_datetime(df[col])
            except:
                pass

    # ---------------- COLUMN TYPE DETECTION (AFTER CONVERSION) ----------------
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64[ns]"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # ---------------- KPIs ----------------
    st.subheader("📌 Key Performance Indicators")

    if numeric_cols:
        kpi_cols = st.columns(len(numeric_cols))
        for i, col in enumerate(numeric_cols):
            kpi_cols[i].metric(
                label=f"{col} Total",
                value=f"{df[col].sum():,.2f}"
            )

    # ---------------- DATE RANGE INFO ----------------
    if datetime_cols:
        st.subheader("📅 Date Ranges")
        for col in datetime_cols:
            st.write(f"**{col}** : {df[col].min()} → {df[col].max()}")

    # ---------------- FILTERS ----------------
    st.subheader("🎛️ Filters")
    filtered_df = df.copy()

    for col in categorical_cols:
        selected_values = st.multiselect(
            f"Filter by {col}",
            options=df[col].unique(),
            default=df[col].unique()
        )
        filtered_df = filtered_df[filtered_df[col].isin(selected_values)]

    # ---------------- VISUALIZATION ----------------
    st.subheader("📈 Data Visualization")

    chart_type = st.selectbox(
        "Select Chart Type",
        ["Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart", "Histogram"]
    )

    if chart_type in ["Bar Chart", "Line Chart", "Scatter Plot"]:
        x_axis = st.selectbox("Select X-axis", filtered_df.columns)
        y_axis = st.selectbox("Select Y-axis (Numeric)", numeric_cols)

    elif chart_type == "Pie Chart":
        x_axis = st.selectbox("Select Category Column", categorical_cols)
        y_axis = st.selectbox("Select Numeric Column", numeric_cols)

    elif chart_type == "Histogram":
        y_axis = st.selectbox("Select Numeric Column", numeric_cols)

    try:
        if chart_type == "Bar Chart":
            fig = px.bar(filtered_df, x=x_axis, y=y_axis)

        elif chart_type == "Line Chart":
            fig = px.line(filtered_df, x=x_axis, y=y_axis)

        elif chart_type == "Scatter Plot":
            fig = px.scatter(filtered_df, x=x_axis, y=y_axis)

        elif chart_type == "Pie Chart":
            fig = px.pie(filtered_df, names=x_axis, values=y_axis)

        elif chart_type == "Histogram":
            fig = px.histogram(filtered_df, x=y_axis)

        st.plotly_chart(fig, use_container_width=True)

    except:
        st.error("Invalid chart configuration. Please change selections.")

    # ================= ML PREDICTION =================
    st.subheader("🤖 Basic ML Prediction (Optional)")

    enable_ml = st.checkbox("Enable ML Prediction")

    if enable_ml:
        if not numeric_cols or not datetime_cols:
            st.warning("ML requires at least one numeric and one date column.")
        else:
            date_col = st.selectbox("Select Date Column", datetime_cols)
            target_col = st.selectbox("Select Target Column", numeric_cols)

            ml_df = df[[date_col, target_col]].dropna()
            ml_df = ml_df.sort_values(by=date_col)

            # Convert date to ordinal
            ml_df["date_ordinal"] = ml_df[date_col].map(pd.Timestamp.toordinal)

            X = ml_df[["date_ordinal"]]
            y = ml_df[target_col]

            model = LinearRegression()
            model.fit(X, y)

            days_ahead = st.slider("Predict Days Ahead", 1, 30, 7)

            last_day = ml_df["date_ordinal"].max()
            future_days = np.array([[last_day + i] for i in range(1, days_ahead + 1)])

            predictions = model.predict(future_days)

            pred_df = pd.DataFrame({
                "Date": [pd.Timestamp.fromordinal(int(d[0])) for d in future_days],
                "Predicted Value": predictions
            })

            st.subheader("📊 Prediction Results")
            st.dataframe(pred_df)

            fig_pred = px.line(
                pred_df,
                x="Date",
                y="Predicted Value",
                title="Future Trend Prediction"
            )

            st.plotly_chart(fig_pred, use_container_width=True)
