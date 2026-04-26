import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

st.set_page_config(
    page_title="PetNet Engagement Predictor",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# STYLE
# -----------------------------
st.markdown(
    """
<style>
header[data-testid="stHeader"] {
    background: #12324a;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #12324a 0%, #0d2233 100%);
}

section[data-testid="stSidebar"] * {
    color: #eaf7ff !important;
}

div[data-testid="metric-container"] {
    background: #f5fbff;
    border-left: 5px solid #27b4d8;
    border-radius: 10px;
    padding: 16px 18px;
}

div[data-testid="metric-container"] label {
    color: #12324a !important;
    font-weight: 700;
}

h1 {
    color: #12324a !important;
    border-bottom: 4px solid #27b4d8;
    padding-bottom: 8px;
}

h2, h3 {
    color: #12324a !important;
}

.petnet-card {
    background: #f5fbff;
    border-left: 5px solid #27b4d8;
    border-radius: 10px;
    padding: 18px 22px;
    margin-bottom: 16px;
}

.warning-card {
    background: #fff8e8;
    border-left: 5px solid #f0a500;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 16px;
}

.success-card {
    background: #ecfff6;
    border-left: 5px solid #28a745;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 16px;
}

.phase-label {
    font-size: 0.75rem;
    font-weight: 800;
    letter-spacing: 1.8px;
    color: #27b4d8;
    text-transform: uppercase;
    margin-bottom: 4px;
}
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# DATA + MODEL
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("dataset.csv")


df = load_data()

features = ["likes", "comments", "shares", "views", "followers"]
X = df[features]
y = df["engagement_score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

engagement_by_type = (
    df.groupby("post_type")["engagement_score"]
    .mean()
    .sort_values(ascending=False)
)

correlations = (
    df.corr(numeric_only=True)["engagement_score"]
    .sort_values(ascending=False)
)

best_type = engagement_by_type.index[0]
best_value = engagement_by_type.iloc[0]

coef_df = pd.DataFrame(
    {
        "Feature": features,
        "Coefficient": model.coef_,
    }
).sort_values("Coefficient", ascending=False)

# -----------------------------
# HELPERS
# -----------------------------
def hr():
    st.markdown("---")


def phase_label(text):
    st.markdown(f'<p class="phase-label">{text}</p>', unsafe_allow_html=True)


def card(text):
    st.markdown(f'<div class="petnet-card">{text}</div>', unsafe_allow_html=True)


def warning_card(text):
    st.markdown(f'<div class="warning-card">{text}</div>', unsafe_allow_html=True)


def success_card(text):
    st.markdown(f'<div class="success-card">{text}</div>', unsafe_allow_html=True)


# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    try:
        st.image("logo.png", width=140)
    except Exception:
        st.markdown("## 🐾 PetNet")

    st.markdown("## PetNet Analytics")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "Executive Summary",
            "Data Understanding",
            "Engagement Analysis",
            "Predictive Model",
            "Prediction Simulator",
            "Recommendations",
            "Technical Appendix",
        ],
    )

    st.markdown("---")
    st.markdown("**Project:** PetNet Engagement Predictor")
    st.markdown("**Author:** Bruno Barradas")
    st.markdown("[GitHub](https://github.com/Bruno-Barradas/petnet-engagement-predictor)")


# =============================================================================
# 1 — EXECUTIVE SUMMARY
# =============================================================================
if page == "Executive Summary":
    phase_label("Phase 1 — Product Analytics Overview")
    st.title("🐾 PetNet Engagement Predictor")
    st.markdown(
        "### Product analytics + machine learning for a social pet platform"
    )

    st.markdown(
        """
This project simulates post engagement behavior for **PetNet**, a social platform for pet owners.

The goal is to demonstrate how a product analytics workflow can be structured for a real app:
from dataset generation to exploratory analysis, predictive modeling, dashboarding, and business recommendations.
"""
    )

    hr()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Posts", f"{len(df):,}")
    col2.metric("Model R²", f"{r2:.2f}")
    col3.metric("MAE", f"{mae:.2f}")
    col4.metric("Top Driver", "Likes")

    hr()

    st.subheader("Executive Summary")

    summary = pd.DataFrame(
        {
            "Element": [
                "Business Context",
                "Dataset",
                "Target Variable",
                "Model",
                "Main Finding",
                "Main Limitation",
            ],
            "Description": [
                "PetNet is a real social pet app concept where users share pet-related content.",
                "Simulated dataset with likes, comments, shares, views, followers and post type.",
                "engagement_score",
                "Linear Regression",
                "Likes are the dominant driver of engagement in this simulation.",
                "The target is formula-based, so model performance is intentionally perfect.",
            ],
        }
    )

    st.dataframe(summary, use_container_width=True, hide_index=True)

    hr()

    st.subheader("Key Results")

    results = pd.DataFrame(
        {
            "Metric": [
                "Dataset size",
                "Best performing post type",
                "Average image engagement",
                "Average video engagement",
                "Model R²",
                "MAE",
                "Top driver",
            ],
            "Value": [
                f"{len(df):,} posts",
                best_type,
                f"{engagement_by_type.get('image', 0):.2f}",
                f"{engagement_by_type.get('video', 0):.2f}",
                f"{r2:.2f}",
                f"{mae:.2f}",
                "Likes",
            ],
        }
    )

    st.dataframe(results, use_container_width=True, hide_index=True)

    success_card(
        """
<b>Portfolio value:</b> This project shows product thinking, ML workflow, and dashboard delivery.
It does not pretend to be a real production model; it clearly documents the simulated nature of the dataset.
"""
    )


# =============================================================================
# 2 — DATA UNDERSTANDING
# =============================================================================
elif page == "Data Understanding":
    phase_label("Phase 2 — Data Understanding")
    st.title("Data Understanding")

    card(
        """
<b>Dataset design:</b> The dataset is simulated based on common social media engagement signals.
It was created to represent how a platform like PetNet could analyze post-level engagement behavior.
"""
    )

    st.subheader("Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    hr()

    st.subheader("Dataset Columns")

    dictionary = pd.DataFrame(
        {
            "Column": [
                "post_id",
                "user_id",
                "likes",
                "comments",
                "shares",
                "views",
                "followers",
                "post_type",
                "engagement_score",
            ],
            "Meaning": [
                "Unique post identifier",
                "User identifier",
                "Number of likes received by the post",
                "Number of comments received by the post",
                "Number of shares received by the post",
                "Number of views received by the post",
                "Follower count of the author",
                "Post format: image or video",
                "Weighted engagement target",
            ],
        }
    )

    st.dataframe(dictionary, use_container_width=True, hide_index=True)

    hr()

    st.subheader("Engagement Score Formula")

    st.code(
        "engagement_score = likes * 0.5 + comments * 0.3 + shares * 0.2",
        language="text",
    )

    warning_card(
        """
<b>Important:</b> This dataset is simulated. The engagement score is deterministic and formula-based.
This explains why the Linear Regression model reaches R² = 1.00 and MAE = 0.00.
"""
    )


# =============================================================================
# 3 — ENGAGEMENT ANALYSIS
# =============================================================================
elif page == "Engagement Analysis":
    phase_label("Phase 3 — Exploratory Data Analysis")
    st.title("Engagement Analysis")

    st.subheader("Average Engagement by Post Type")

    col1, col2 = st.columns([2, 1])

    with col1:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        engagement_by_type.plot(kind="bar", ax=ax)
        ax.set_title("Average Engagement by Post Type")
        ax.set_xlabel("Post Type")
        ax.set_ylabel("Average Engagement Score")
        ax.grid(True, alpha=0.3, axis="y")
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.metric("Best Type", best_type)
        st.metric("Avg Engagement", f"{best_value:.2f}")

    hr()

    st.subheader("Correlation with Engagement Score")
    st.dataframe(
        correlations.to_frame("Correlation"),
        use_container_width=True,
    )

    hr()

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Likes vs Engagement")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(df["likes"], df["engagement_score"], alpha=0.65)
        ax.set_xlabel("Likes")
        ax.set_ylabel("Engagement Score")
        ax.set_title("Likes vs Engagement Score")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    with col4:
        st.subheader("Followers vs Engagement")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(df["followers"], df["engagement_score"], alpha=0.65)
        ax.set_xlabel("Followers")
        ax.set_ylabel("Engagement Score")
        ax.set_title("Followers vs Engagement Score")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    success_card(
        """
<b>Main insight:</b> Likes are the dominant engagement driver in this simulated dataset.
Followers and views have weak direct relationship with engagement because they are not part of the target formula.
"""
    )


# =============================================================================
# 4 — PREDICTIVE MODEL
# =============================================================================
elif page == "Predictive Model":
    phase_label("Phase 4 — Machine Learning Model")
    st.title("Predictive Model")

    st.markdown(
        f"""
A Linear Regression model was trained to predict `engagement_score`.

**R² = {r2:.2f}**  
**MAE = {mae:.2f}**
"""
    )

    hr()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Feature Coefficients")
        st.dataframe(coef_df, use_container_width=True, hide_index=True)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(coef_df["Feature"], coef_df["Coefficient"])
        ax.set_title("Model Coefficients")
        ax.set_xlabel("Coefficient")
        ax.grid(True, alpha=0.3, axis="x")
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.subheader("Predicted vs Real")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(y_test, y_pred, alpha=0.65)
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], linestyle="--")
        ax.set_xlabel("Real Engagement")
        ax.set_ylabel("Predicted Engagement")
        ax.set_title("Predicted vs Real")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    warning_card(
        """
<b>Analytical honesty:</b> The model is perfect because the target variable is generated by a formula.
This is documented intentionally to show transparency and analytical maturity.
"""
    )


# =============================================================================
# 5 — PREDICTION SIMULATOR
# =============================================================================
elif page == "Prediction Simulator":
    phase_label("Phase 4 — Interactive Prediction")
    st.title("Prediction Simulator")

    st.markdown(
        "Adjust the inputs below to estimate the expected engagement score of a PetNet post."
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        likes = st.slider("Likes", 0, 500, 200)

    with c2:
        comments = st.slider("Comments", 0, 100, 30)

    with c3:
        shares = st.slider("Shares", 0, 50, 10)

    with c4:
        views = st.slider("Views", 100, 10000, 3000)

    with c5:
        followers = st.slider("Followers", 100, 10000, 2500)

    input_data = pd.DataFrame(
        [[likes, comments, shares, views, followers]],
        columns=features,
    )

    prediction = model.predict(input_data)[0]

    st.metric("Predicted Engagement Score", f"{prediction:.2f}")

    success_card(
        """
<b>Product use case:</b> In a real app, this type of simulator could help creators estimate expected engagement before posting.
"""
    )


# =============================================================================
# 6 — RECOMMENDATIONS
# =============================================================================
elif page == "Recommendations":
    phase_label("Phase 5 — Product Recommendations")
    st.title("Product Recommendations")

    st.markdown(
        "Recommendations based on the simulated data and product analytics workflow."
    )

    hr()

    recommendations = pd.DataFrame(
        {
            "Recommendation": [
                "Build a creator engagement preview tool",
                "Track engagement drivers by media type",
                "Add more realistic product features",
                "Test non-linear models after real data collection",
                "Move from simulated data to production analytics",
            ],
            "Why it matters": [
                "Helps creators estimate expected engagement before posting.",
                "Supports decisions about image vs video strategy.",
                "Improves model realism and business relevance.",
                "Real engagement is rarely perfectly linear.",
                "Real user data would validate whether simulated assumptions hold.",
            ],
            "Priority": [
                "High",
                "Medium",
                "High",
                "Medium",
                "High",
            ],
        }
    )

    st.dataframe(recommendations, use_container_width=True, hide_index=True)

    hr()

    card(
        """
<b>Strategic direction:</b> The next version of this project should include real behavioral data from the PetNet app,
including post time, hashtags, media type, comments, saves, profile visits, and retention outcomes.
"""
    )


# =============================================================================
# 7 — TECHNICAL APPENDIX
# =============================================================================
elif page == "Technical Appendix":
    phase_label("Technical Appendix")
    st.title("Technical Appendix")

    st.subheader("Tech Stack")

    tech = pd.DataFrame(
        {
            "Tool": [
                "Python",
                "Pandas",
                "Matplotlib",
                "Scikit-learn",
                "Streamlit",
            ],
            "Use": [
                "Core language",
                "Data manipulation",
                "Visual analysis",
                "Linear Regression model",
                "Interactive dashboard",
            ],
        }
    )

    st.dataframe(tech, use_container_width=True, hide_index=True)

    hr()

    st.subheader("Model Inputs")

    st.dataframe(
        pd.DataFrame({"Feature": features}),
        use_container_width=True,
        hide_index=True,
    )

    hr()

    st.subheader("Model Output")

    st.markdown("Target variable: `engagement_score`")

    st.code(
        "engagement_score = likes * 0.5 + comments * 0.3 + shares * 0.2",
        language="text",
    )

    hr()

    st.subheader("Repository")

    st.markdown("https://github.com/Bruno-Barradas/petnet-engagement-predictor")