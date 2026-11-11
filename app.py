

# ============================================================
# MARKETING INTELLIGENCE & BUDGET OPTIMIZER
# Marketing Mix Modeling + Incrementality + Budget Optimization
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_absolute_error

from scipy.optimize import minimize

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Marketing Intelligence & Budget Optimizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Marketing Intelligence & Budget Optimizer")

st.markdown("""
**An intelligent decision-support platform for marketing
campaign effectiveness, incremental revenue estimation,
and dynamic advertising budget allocation.**
""")

st.divider()

# ============================================================
# 2. GLOBAL CONFIGURATION
# ============================================================

CHANNELS = [
    "google_spend",
    "meta_spend",
    "email_spend",
    "influencer_spend"
]

CHANNEL_NAMES = {
    "google_spend": "Google Ads",
    "meta_spend": "Meta Ads",
    "email_spend": "Email Marketing",
    "influencer_spend": "Influencer Marketing"
}

# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def adstock(series, decay):
    """Calculate advertising carryover."""

    series = np.asarray(series, dtype=float)

    result = np.zeros(len(series))

    for i in range(len(series)):

        previous = result[i-1] if i > 0 else 0

        result[i] = series[i] + decay * previous

    return result


def hill_transform(x, half_saturation):
    """Apply diminishing-return saturation."""

    x = np.maximum(np.asarray(x, dtype=float), 0)

    return (
        x ** 1.5
    ) / (
        x ** 1.5 + half_saturation ** 1.5 + 1e-9
    )


def generate_data(n=156, seed=42):
    """Generate illustrative weekly marketing data."""

    rng = np.random.default_rng(seed)

    data = pd.DataFrame({
        "date": pd.date_range(
            end=pd.Timestamp.today().normalize(),
            periods=n,
            freq="W"
        )
    })

    data["google_spend"] = rng.uniform(30000, 90000, n)
    data["meta_spend"] = rng.uniform(20000, 70000, n)
    data["email_spend"] = rng.uniform(5000, 25000, n)
    data["influencer_spend"] = rng.uniform(10000, 50000, n)

    data["promo"] = rng.integers(0, 2, n)

    data["price_index"] = rng.uniform(0.9, 1.1, n)

    decay_values = {
        "google_spend": 0.45,
        "meta_spend": 0.50,
        "email_spend": 0.30,
        "influencer_spend": 0.65
    }

    effect_values = {
        "google_spend": 0.18,
        "meta_spend": 0.15,
        "email_spend": 0.10,
        "influencer_spend": 0.12
    }

    revenue = np.full(n, 500000.0)

    for channel in CHANNELS:

        accumulated = adstock(
            data[channel].values,
            decay_values[channel]
        )

        saturation = hill_transform(
            accumulated,
            np.median(data[channel]) * 2
        )

        revenue += (
            effect_values[channel]
            * 10000000
            * saturation
        )

    trend = np.linspace(0, 150000, n)

    revenue += trend

    revenue += data["promo"] * 100000

    revenue -= (
        data["price_index"] - 1
    ) * 1500000

    revenue += rng.normal(0, 90000, n)

    data["sales"] = np.maximum(revenue, 100000)

    return data


def format_currency(value):
    return f"₹{value:,.0f}"


def make_features(data, decay, saturation_factor):

    features = pd.DataFrame(index=data.index)

    for channel in CHANNELS:

        accumulated = adstock(
            data[channel].values,
            decay
        )

        half_saturation = max(
            np.median(data[channel]) *
            saturation_factor / (1 - decay + 1e-9),
            1.0
        )

        features[channel] = hill_transform(
            accumulated,
            half_saturation
        )

    features["promo"] = data["promo"].values

    features["price_index"] = data["price_index"].values

    features["trend"] = np.linspace(
        0, 1, len(data)
    )

    return features


# ============================================================
# 4. SIDEBAR - DATA CONFIGURATION
# ============================================================

st.sidebar.title("⚙️ Configuration")

st.sidebar.subheader("Data Source")

data_source = st.sidebar.radio(
    "Select data source",
    [
        "Simulated Dataset",
        "Upload CSV Dataset"
    ]
)

if data_source == "Upload CSV Dataset":

    uploaded_file = st.sidebar.file_uploader(
        "Upload marketing CSV",
        type=["csv"]
    )

    if uploaded_file is None:

        st.info(
            "Please upload a CSV dataset to proceed."
        )

        st.stop()

    try:
        df = pd.read_csv(uploaded_file)

    except Exception as error:

        st.error(f"Unable to read CSV: {error}")

        st.stop()

else:

    df = generate_data()

# ============================================================
# 5. DATA VALIDATION AND PREPROCESSING
# ============================================================

st.sidebar.subheader("Dataset Information")

required_columns = [
    "date",
    "sales"
] + CHANNELS

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:

    st.error(
        f"Missing required columns: {missing_columns}"
    )

    st.stop()

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

numeric_columns = [
    "sales"
] + CHANNELS

for col in numeric_columns:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

if "promo" not in df.columns:

    df["promo"] = 0

if "price_index" not in df.columns:

    df["price_index"] = 1.0

df["promo"] = pd.to_numeric(
    df["promo"],
    errors="coerce"
)

df["price_index"] = pd.to_numeric(
    df["price_index"],
    errors="coerce"
)

df = df.dropna(
    subset=[
        "date",
        "sales",
        "price_index",
        "promo"
    ] + CHANNELS
).copy()

df = df.sort_values(
    "date"
).reset_index(drop=True)

if len(df) < 30:

    st.error(
        "At least 30 valid observations are required."
    )

    st.stop()

if (df[CHANNELS] < 0).any().any():

    st.error(
        "Advertising expenditure cannot be negative."
    )

    st.stop()

if (df["sales"] <= 0).any():

    st.error(
        "Sales values must be positive."
    )

    st.stop()

if (df["price_index"] <= 0).any():

    st.error(
        "Price index must be positive."
    )

    st.stop()

if not np.isfinite(
    df[["sales", "price_index", "promo"] + CHANNELS].values
).all():

    st.error(
        "The dataset contains invalid numerical values."
    )

    st.stop()

df["promo"] = (df["promo"] > 0).astype(int)

st.sidebar.metric(
    "Observations",
    len(df)
)

st.sidebar.metric(
    "Total Revenue",
    format_currency(df["sales"].sum())
)

# ============================================================
# 6. MODEL PARAMETERS
# ============================================================

st.sidebar.divider()

st.sidebar.subheader("Model Parameters")

decay = st.sidebar.slider(
    "Adstock Decay Factor",
    min_value=0.0,
    max_value=0.9,
    value=0.5,
    step=0.05
)

ridge_alpha = st.sidebar.slider(
    "Ridge Regularization",
    min_value=0.1,
    max_value=100.0,
    value=10.0,
    step=0.1
)

saturation_factor = st.sidebar.slider(
    "Saturation Multiplier",
    min_value=0.5,
    max_value=5.0,
    value=2.0,
    step=0.1
)

# ============================================================
# 7. MARKETING MIX MODEL
# ============================================================

X = make_features(
    df,
    decay,
    saturation_factor
)

y = df["sales"].values

model = make_pipeline(
    StandardScaler(),
    Ridge(alpha=ridge_alpha)
)

model.fit(X, y)

predictions = model.predict(X)

residuals = y - predictions

r2 = r2_score(y, predictions)

mae = mean_absolute_error(y, predictions)

rmse = np.sqrt(np.mean(residuals ** 2))

# ============================================================
# 8. COUNTERFACTUAL INCREMENTALITY ANALYSIS
# ============================================================

channel_results = []

for channel in CHANNELS:

    counterfactual = X.copy()

    counterfactual[channel] = 0

    baseline_prediction = model.predict(X)

    counterfactual_prediction = model.predict(
        counterfactual
    )

    incremental_revenue = np.sum(
        baseline_prediction -
        counterfactual_prediction
    )

    total_spend = df[channel].sum()

    incremental_roas = (
        incremental_revenue / total_spend
        if total_spend > 0 else 0
    )

    channel_results.append({

        "Channel": CHANNEL_NAMES[channel],

        "Total Spend": total_spend,

        "Estimated Incremental Revenue":
            incremental_revenue,

        "Incremental ROAS":
            incremental_roas

    })

results = pd.DataFrame(channel_results)

# ============================================================
# 9. EXECUTIVE DASHBOARD
# ============================================================

st.header("1. Executive Marketing Dashboard")

total_sales = df["sales"].sum()

total_spend = df[CHANNELS].sum().sum()

total_incremental = results[
    "Estimated Incremental Revenue"
].sum()

overall_roas = (
    total_sales / total_spend
    if total_spend > 0 else 0
)

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Total Revenue",
    format_currency(total_sales)
)

c2.metric(
    "Marketing Expenditure",
    format_currency(total_spend)
)

c3.metric(
    "Estimated Incremental Revenue",
    format_currency(total_incremental)
)

c4.metric(
    "Overall ROAS",
    f"{overall_roas:.2f}x"
)

st.caption(
    "Monetary values use the currency units supplied in the dataset. "
    "The synthetic dataset uses INR."
)

st.divider()

# ============================================================
# 10. MODEL PERFORMANCE
# ============================================================

st.header("2. Model Performance")

m1, m2, m3 = st.columns(3)

m1.metric(
    "R-squared",
    f"{r2:.3f}"
)

m2.metric(
    "Mean Absolute Error",
    format_currency(mae)
)

m3.metric(
    "Root Mean Squared Error",
    format_currency(rmse)
)

performance_df = pd.DataFrame({

    "Date": df["date"],

    "Actual Revenue": y,

    "Predicted Revenue": predictions

})

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=performance_df["Date"],
    y=performance_df["Actual Revenue"],
    name="Actual Revenue",
    mode="lines"
))

fig.add_trace(go.Scatter(
    x=performance_df["Date"],
    y=performance_df["Predicted Revenue"],
    name="Predicted Revenue",
    mode="lines"
))

fig.update_layout(
    title="Actual vs Predicted Revenue",
    xaxis_title="Date",
    yaxis_title="Revenue",
    hovermode="x unified"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ============================================================
# 11. CHANNEL EFFECTIVENESS
# ============================================================

st.header("3. Marketing Channel Effectiveness")

display_results = results.copy()

display_results["Total Spend"] = (
    display_results["Total Spend"].map(format_currency)
)

display_results["Estimated Incremental Revenue"] = (
    display_results["Estimated Incremental Revenue"]
    .map(format_currency)
)

display_results["Incremental ROAS"] = (
    display_results["Incremental ROAS"]
    .map(lambda x: f"{x:.2f}x")
)

st.dataframe(
    display_results,
    use_container_width=True,
    hide_index=True
)

fig = px.bar(
    results,
    x="Channel",
    y="Incremental ROAS",
    color="Channel",
    title="Incremental ROAS by Marketing Channel"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ============================================================
# 12. MARKETING EXPENDITURE VS REVENUE
# ============================================================

st.header("4. Marketing Expenditure vs Revenue Trends")

trend_df = df[
    ["date", "sales"]
].copy()

trend_df["Total Marketing Spend"] = (
    df[CHANNELS].sum(axis=1)
)

trend_df["Sales Index"] = (
    trend_df["sales"] /
    trend_df["sales"].iloc[0]
) * 100

trend_df["Marketing Spend Index"] = (
    trend_df["Total Marketing Spend"] /
    trend_df["Total Marketing Spend"].iloc[0]
) * 100

trend_long = trend_df.melt(
    id_vars="date",
    value_vars=[
        "Sales Index",
        "Marketing Spend Index"
    ],
    var_name="Metric",
    value_name="Index"
)

fig = px.line(
    trend_long,
    x="date",
    y="Index",
    color="Metric",
    markers=True,
    title="Indexed Marketing Spend vs Revenue"
)

fig.update_layout(
    hovermode="x unified",
    xaxis_title="Date",
    yaxis_title="Index (Base = 100)"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ============================================================
# 13. MARKETING EXPENDITURE DISTRIBUTION
# ============================================================

st.header("5. Marketing Expenditure Distribution")

spend_distribution = pd.DataFrame({

    "Channel": [
        CHANNEL_NAMES[c] for c in CHANNELS
    ],

    "Expenditure": [
        df[c].sum() for c in CHANNELS
    ]

})

fig = px.pie(
    spend_distribution,
    names="Channel",
    values="Expenditure",
    hole=0.55,
    title="Marketing Budget Distribution"
)

fig.update_traces(
    textposition="inside",
    textinfo="percent+label"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ============================================================
# 14. INCREMENTAL REVENUE CONTRIBUTION
# ============================================================

st.header("6. Incremental Revenue Contribution")

contribution = results.sort_values(
    "Estimated Incremental Revenue",
    ascending=True
)

fig = px.bar(
    contribution,
    x="Estimated Incremental Revenue",
    y="Channel",
    orientation="h",
    color="Incremental ROAS",
    color_continuous_scale="Viridis",
    title="Estimated Incremental Revenue by Channel",
    text_auto=",.0f"
)

fig.update_layout(
    xaxis_title="Estimated Incremental Revenue",
    yaxis_title="Channel"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ============================================================
# 15. ADVERTISING RESPONSE AND SATURATION
# ============================================================

st.header("7. Advertising Response and Saturation Analysis")

selected_channel = st.selectbox(
    "Select advertising channel",
    CHANNELS,
    format_func=lambda x: CHANNEL_NAMES[x]
)

spend_range = np.linspace(
    df[selected_channel].min() * 0.5,
    df[selected_channel].max() * 1.8,
    100
)

# Hold other model variables at their historical means.
base_features = pd.DataFrame(
    [X.mean().to_dict()],
    columns=X.columns
)

predicted_revenues = []

half_saturation = max(
    np.median(df[selected_channel]) *
    saturation_factor / (1 - decay + 1e-9),
    1.0
)

for spend in spend_range:

    test_features = base_features.copy()

    accumulated_spend = (
        spend / (1 - decay + 1e-9)
    )

    transformed_spend = hill_transform(
        np.array([accumulated_spend]),
        half_saturation
    )[0]

    test_features[selected_channel] = transformed_spend

    predicted_revenues.append(
        model.predict(test_features)[0]
    )

response_df = pd.DataFrame({

    "Weekly Advertising Spend": spend_range,

    "Predicted Revenue": predicted_revenues

})

fig = px.line(
    response_df,
    x="Weekly Advertising Spend",
    y="Predicted Revenue",
    title=f"Advertising Response Curve: {CHANNEL_NAMES[selected_channel]}"
)

fig.update_layout(
    xaxis_title="Weekly Advertising Expenditure",
    yaxis_title="Predicted Weekly Revenue"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.info(
    "This curve shows model-based predictions while other "
    "variables are held constant. It is not an experimentally "
    "validated causal response curve."
)

# ============================================================
# 16. MODEL DIAGNOSTICS
# ============================================================

st.header("8. Model Diagnostic Dashboard")

diagnostic_df = pd.DataFrame({

    "Predicted Revenue": predictions,

    "Residuals": residuals

})

tab1, tab2, tab3 = st.tabs([

    "Actual vs Predicted",

    "Residual Distribution",

    "Prediction Errors"

])

with tab1:

    fig = px.scatter(
        x=y,
        y=predictions,
        labels={
            "x": "Actual Revenue",
            "y": "Predicted Revenue"
        },
        title="Actual vs Predicted Revenue"
    )

    min_value = min(y.min(), predictions.min())

    max_value = max(y.max(), predictions.max())

    fig.add_trace(go.Scatter(
        x=[min_value, max_value],
        y=[min_value, max_value],
        mode="lines",
        name="Perfect Prediction"
    ))

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with tab2:

    fig = px.histogram(
        diagnostic_df,
        x="Residuals",
        nbins=30,
        marginal="box",
        title="Residual Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with tab3:

    fig = px.scatter(
        diagnostic_df,
        x="Predicted Revenue",
        y="Residuals",
        title="Residuals vs Predicted Revenue"
    )

    fig.add_hline(
        y=0,
        line_dash="dash"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

st.caption(
    "Model diagnostics above are based on training observations. "
    "Chronological holdout validation is recommended before "
    "using the model for operational forecasting."
)

# ============================================================
# 17. MARKETING CORRELATION HEATMAP
# ============================================================

st.header("9. Marketing Variable Correlation Analysis")

correlation_columns = CHANNELS + [

    "sales",

    "promo",

    "price_index"

]

correlation_matrix = df[
    correlation_columns
].corr()

correlation_matrix.index = [

    CHANNEL_NAMES.get(x, x)

    for x in correlation_matrix.index

]

correlation_matrix.columns = [

    CHANNEL_NAMES.get(x, x)

    for x in correlation_matrix.columns

]

fig = px.imshow(
    correlation_matrix,
    text_auto=".2f",
    aspect="auto",
    color_continuous_scale="RdBu_r",
    zmin=-1,
    zmax=1,
    title="Marketing Variables Correlation Heatmap"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ============================================================
# 18. DYNAMIC MARKETING BUDGET OPTIMIZATION
# ============================================================

st.header("10. Dynamic Marketing Budget Optimization")

st.markdown("""
The optimization engine reallocates the existing weekly
marketing budget across advertising channels to maximize
the revenue predicted by the fitted marketing model.
""")

weekly_budget = df[CHANNELS].sum().sum() / len(df)

st.metric(
    "Average Weekly Marketing Budget",
    format_currency(weekly_budget)
)

initial_budget = (
    df[CHANNELS].mean().values.astype(float)
)

average_controls = {

    "promo": df["promo"].mean(),

    "price_index": df["price_index"].mean()

}

def optimize_features(budgets):

    row = {}

    for i, channel in enumerate(CHANNELS):

        accumulated_spend = (
            budgets[i] / (1 - decay + 1e-9)
        )

        channel_half_saturation = max(
            np.median(df[channel]) *
            saturation_factor / (1 - decay + 1e-9),
            1.0
        )

        row[channel] = hill_transform(
            np.array([accumulated_spend]),
            channel_half_saturation
        )[0]

    row["promo"] = average_controls["promo"]

    row["price_index"] = average_controls["price_index"]

    # Use the same average trend assumption as the
    # historical feature representation.
    row["trend"] = X["trend"].mean()

    return pd.DataFrame(
        [row],
        columns=X.columns
    )


def objective(budgets):

    features = optimize_features(budgets)

    predicted_revenue = model.predict(
        features
    )[0]

    return -predicted_revenue


bounds = [

    (0, weekly_budget)

    for _ in CHANNELS

]

optimization = minimize(

    objective,

    initial_budget,

    method="SLSQP",

    bounds=bounds,

    constraints=[

        {

            "type": "eq",

            "fun": lambda b: np.sum(b) - weekly_budget

        }

    ],

    options={

        "maxiter": 500,

        "ftol": 1e-8

    }

)

if optimization.success:

    optimized_budget = optimization.x

    current_revenue = -objective(initial_budget)

    optimized_revenue = -objective(optimized_budget)

    budget_comparison = pd.DataFrame({

        "Channel": [

            CHANNEL_NAMES[c] for c in CHANNELS

        ],

        "Current Budget": initial_budget,

        "Optimized Budget": optimized_budget

    })

    budget_comparison["Change"] = (

        budget_comparison["Optimized Budget"] -

        budget_comparison["Current Budget"]

    )

    budget_comparison["Change (%)"] = np.where(

        budget_comparison["Current Budget"] > 0,

        budget_comparison["Change"] /

        budget_comparison["Current Budget"] * 100,

        np.nan

    )

    # --------------------------------------------------------
    # 18A. CURRENT VS OPTIMIZED BUDGET
    # --------------------------------------------------------

    st.subheader("Current vs Optimized Budget Allocation")

    st.dataframe(

        budget_comparison.style.format({

            "Current Budget": "₹{:,.0f}",

            "Optimized Budget": "₹{:,.0f}",

            "Change": "₹{:,.0f}",

            "Change (%)": "{:.2f}%"

        }),

        use_container_width=True,

        hide_index=True

    )

    chart_data = budget_comparison.melt(

        id_vars="Channel",

        value_vars=[

            "Current Budget",

            "Optimized Budget"

        ],

        var_name="Allocation",

        value_name="Budget"

    )

    fig = px.bar(

        chart_data,

        x="Channel",

        y="Budget",

        color="Allocation",

        barmode="group",

        title="Current vs Optimized Marketing Budget"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    # --------------------------------------------------------
    # 18B. BUDGET REALLOCATION IMPACT
    # --------------------------------------------------------

    st.subheader("Budget Reallocation Impact")

    fig = px.bar(

        budget_comparison,

        x="Channel",

        y="Change",

        color="Change",

        color_continuous_scale="RdYlGn",

        title="Increase or Decrease in Channel Budget"

    )

    fig.add_hline(

        y=0,

        line_dash="dash"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    # --------------------------------------------------------
    # 18C. OPTIMIZED BUDGET DISTRIBUTION
    # --------------------------------------------------------

    st.subheader("Optimized Budget Composition")

    fig = px.pie(

        budget_comparison,

        names="Channel",

        values="Optimized Budget",

        hole=0.5,

        title="Recommended Marketing Budget Distribution"

    )

    fig.update_traces(

        textinfo="percent+label"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    # --------------------------------------------------------
    # 18D. PROJECTED REVENUE IMPROVEMENT
    # --------------------------------------------------------

    revenue_improvement = (
        optimized_revenue - current_revenue
    )

    improvement_percentage = (

        revenue_improvement / current_revenue * 100

        if current_revenue != 0 else 0

    )

    c1, c2, c3 = st.columns(3)

    c1.metric(

        "Current Predicted Weekly Revenue",

        format_currency(current_revenue)

    )

    c2.metric(

        "Optimized Predicted Weekly Revenue",

        format_currency(optimized_revenue)

    )

    c3.metric(

        "Predicted Revenue Change",

        format_currency(revenue_improvement),

        delta=f"{improvement_percentage:.2f}%"

    )

    st.success(

        "The optimizer has identified a feasible budget "
        "allocation that maximizes the fitted model's "
        "predicted weekly revenue under the total-budget constraint."

    )

else:

    st.warning(

        "The optimization algorithm did not converge. "
        "Try different model parameters or inspect the dataset."

    )

# ============================================================
# 19. AUTOMATED MANAGEMENT RECOMMENDATIONS
# ============================================================

st.header("11. Automated Management Recommendations")

best_channel = results.loc[

    results["Incremental ROAS"].idxmax()

]

worst_channel = results.loc[

    results["Incremental ROAS"].idxmin()

]

col1, col2 = st.columns(2)

with col1:

    st.subheader("Highest Incremental ROAS")

    st.success(

        f"{best_channel['Channel']} demonstrates the "
        f"highest model-estimated incremental ROAS of "
        f"{best_channel['Incremental ROAS']:.2f}x."

    )

with col2:

    st.subheader("Lowest Incremental ROAS")

    st.warning(

        f"{worst_channel['Channel']} demonstrates the "
        f"lowest model-estimated incremental ROAS of "
        f"{worst_channel['Incremental ROAS']:.2f}x."

    )

if optimization.success:

    st.subheader("Budget Allocation Recommendations")

    for _, row in budget_comparison.iterrows():

        if row["Change"] > 0:

            st.write(

                f"📈 Increase {row['Channel']} allocation "
                f"by {format_currency(row['Change'])} per week."

            )

        elif row["Change"] < 0:

            st.write(

                f"📉 Reduce {row['Channel']} allocation "
                f"by {format_currency(abs(row['Change']))} per week."

            )

        else:

            st.write(

                f"➡️ Maintain {row['Channel']} allocation."

            )

st.info(

    "These recommendations are model-based decision-support "
    "outputs. Validate proposed changes through controlled "
    "experiments before implementing them at scale."

)

# ============================================================
# 20. DOWNLOAD ANALYTICAL RESULTS
# ============================================================

st.header("12. Export Analytical Results")

col1, col2 = st.columns(2)

with col1:

    channel_csv = results.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(

        "⬇️ Download Channel Analysis",

        data=channel_csv,

        file_name="marketing_channel_analysis.csv",

        mime="text/csv"

    )

with col2:

    if optimization.success:

        budget_csv = budget_comparison.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(

            "⬇️ Download Optimized Budget",

            data=budget_csv,

            file_name="optimized_marketing_budget.csv",

            mime="text/csv"

        )

# ============================================================
# 21. DATA PREVIEW
# ============================================================

with st.expander("View Marketing Dataset"):

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

# ============================================================
# 22. FOOTER
# ============================================================

st.divider()

st.markdown("""
### Marketing Intelligence & Budget Optimizer

**Powered by:** Python | Streamlit | Scikit-learn |
Plotly | SciPy

*An integrated analytical framework for marketing
effectiveness measurement, incremental revenue estimation,
and intelligent advertising budget allocation.*

**Note:** This prototype uses a simplified Marketing Mix
Model. Production deployment requires out-of-sample
validation, uncertainty quantification, appropriate causal
validation, and business-specific optimization constraints.
""")
