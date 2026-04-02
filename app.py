import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="PetNet Engagement Predictor",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
header[data-testid="stHeader"] { background: #1a2e4a; }
section[data-testid="stSidebar"] { background: #1a2e4a; }
section[data-testid="stSidebar"] * { color: #cce0f5 !important; }
div[data-testid="metric-container"] {
    background: #f5f7fa;
    border-left: 4px solid #2d5a8e;
    border-radius: 6px;
    padding: 14px 18px;
}
div[data-testid="metric-container"] label { color: #2d5a8e !important; font-weight: 600; }
h1 { color: #1a2e4a !important; border-bottom: 3px solid #2d5a8e; padding-bottom: 6px; }
h2 { color: #2d5a8e !important; }
h3 { color: #1a2e4a !important; }
.finding-card {
    background: #f5f7fa;
    border-left: 5px solid #2d5a8e;
    border-radius: 6px;
    padding: 16px 20px;
    margin-bottom: 14px;
}
.finding-card.high   { border-left-color: #e8604c; }
.finding-card.medium { border-left-color: #f0a500; }
.finding-card.low    { border-left-color: #28a745; }
.rec-box {
    background: #eaf3fb;
    border: 1.5px solid #4a90c4;
    border-radius: 6px;
    padding: 12px 16px;
    margin: 8px 0 4px 0;
    font-size: 0.9rem;
}
.limit-box {
    background: #fff8e8;
    border: 1.5px solid #f0c040;
    border-radius: 6px;
    padding: 10px 14px;
    margin: 4px 0;
    font-size: 0.87rem;
    font-style: italic;
}
.phase-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    color: #4a90c4;
    text-transform: uppercase;
    margin-bottom: 2px;
}
</style>
""", unsafe_allow_html=True)


# ── Data & Model ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv")
    le_media = LabelEncoder()
    le_pet   = LabelEncoder()
    df["media_enc"] = le_media.fit_transform(df["media_type"])
    df["pet_enc"]   = le_pet.fit_transform(df["pet_type"])
    df["follower_tier"] = pd.cut(
        df["followers"],
        bins=[0, 1000, 3000, 6000, 8000],
        labels=["0-1k", "1k-3k", "3k-6k", "6k-8k"],
    )
    return df, le_media, le_pet

@st.cache_data
def train_model(_df):
    features = ["followers", "comments", "video_duration_sec",
                 "hashtags", "post_hour", "media_enc", "pet_enc", "video_size_mb"]
    X = _df[features]
    y = _df["engagement"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, features, r2_score(y_test, y_pred), mean_absolute_error(y_test, y_pred), X_test, y_test, y_pred

df, le_media, le_pet = load_data()
model, features, R2, MAE, X_test, y_test, y_pred = train_model(df)

DARK_BLUE  = "#1a2e4a"
MID_BLUE   = "#2d5a8e"
LIGHT_BLUE = "#4a90c4"
ACCENT     = "#e8604c"
GOLD       = "#f0a500"
GREEN      = "#28a745"
GRAY       = "#aaaaaa"

def phase_label(txt):
    st.markdown(f'<p class="phase-label">{txt}</p>', unsafe_allow_html=True)

def rec_box(txt):
    st.markdown(f'<div class="rec-box"><b>Analysis & Recommendation:</b> {txt}</div>', unsafe_allow_html=True)

def limit_box(txt):
    st.markdown(f'<div class="limit-box"><i>Limitation:</i> {txt}</div>', unsafe_allow_html=True)

def hr():
    st.markdown("---")


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## PetNet Analytics")
    st.markdown("---")
    page = st.radio("Navigation", [
        "Executive Summary",
        "Data Understanding",
        "Engagement Analysis",
        "Predictive Model",
        "Additional Insights",
        "Recommendations",
        "Technical Appendix",
    ])
    st.markdown("---")
    st.markdown("**Project:** PetNet Engagement Predictor")
    st.markdown("**Author:** Bruno Barradas")
    st.markdown("[GitHub](https://github.com/Bruno-Barradas/petnet-engagement-predictor)")


# =============================================================================
# 1 — EXECUTIVE SUMMARY
# =============================================================================
if page == "Executive Summary":
    st.title("PetNet Engagement Predictor")
    st.markdown("### Customer Engagement & Predictive Analytics")
    st.markdown(
        "Analysis of post-level engagement data from **PetNet** — a social platform for pet owners — "
        "to identify the key drivers of user interaction and build a predictive model "
        "capable of estimating post engagement before publication. "
        "All analyses are based on real platform data."
    )
    hr()

    st.subheader("Executive Summary")
    summary = pd.DataFrame({
        "Element": ["Objective", "Data source", "Scope", "Audience", "Deliverable"],
        "Description": [
            "Identify factors that influence post engagement on PetNet and build a predictive model.",
            "PetNet platform dataset (dataset.csv) — 400 posts, 9 variables.",
            "Posts with audience, content, media type, and behavioral features.",
            "Content creators, product team, and PetNet analytics.",
            "Descriptive analysis + predictive model (R2=0.95) with actionable recommendations.",
        ],
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)
    hr()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Posts analyzed", "400")
    c2.metric("Features", "9")
    c3.metric("Model R2", f"{R2:.2f}")
    c4.metric("MAE", f"{MAE:.0f} pts")
    c5.metric("Null values", "0")
    hr()

    st.subheader("Key Findings")
    findings = [
        ("high",   "1", "Followers is the primary engagement driver (corr=0.73)",
         "HIGH",   "Prioritize audience growth strategies for creators"),
        ("high",   "2", "Comments signal deep engagement — 2nd strongest driver (corr=0.60)",
         "HIGH",   "Build features that incentivize discussions: polls, Q&A"),
        ("medium", "3", "Video outperforms photo by +9% in average engagement",
         "MEDIUM", "Guide creators to prioritize video content"),
        ("medium", "4", "Video duration has moderate positive impact (corr=0.27)",
         "MEDIUM", "Encourage longer, more substantive videos (75-90s)"),
        ("low",    "5", "Posting time has near-zero correlation (-0.075)",
         "LOW",    "Do not prioritize posting time — focus on content quality"),
    ]

    for cls, num, title, impact, action in findings:
        badge_color = {"high": ACCENT, "medium": GOLD, "low": GREEN}[cls]
        st.markdown(f"""
        <div class="finding-card {cls}">
            <b>Finding {num} — {title}</b><br>
            <span style="color:{badge_color};font-weight:700;font-size:0.85rem">Impact: {impact}</span><br>
            <span style="font-size:0.9rem;color:#555">→ {action}</span>
        </div>
        """, unsafe_allow_html=True)

    hr()
    st.info(
        "**Executive conclusion:** PetNet engagement is highly predictable (R2=0.95) and driven by "
        "two structural factors: audience size and comment volume. Video consistently outperforms photo. "
        "Posting time and file size have negligible impact and should not be strategic priorities."
    )


# =============================================================================
# 2 — DATA UNDERSTANDING
# =============================================================================
elif page == "Data Understanding":
    phase_label("Phase 2 — Data Understanding")
    st.title("Data Understanding")
    hr()

    st.subheader("Data Sources")
    src = pd.DataFrame({
        "Source": ["PetNet Dataset", "Serialized Model"],
        "File": ["dataset.csv", "engagement_model.pkl"],
        "Description": [
            "400 posts with 9 audience, content and behavioral features",
            "Trained Linear Regression model for engagement prediction",
        ],
    })
    st.dataframe(src, use_container_width=True, hide_index=True)
    hr()

    st.subheader("Data Dictionary")
    dic = pd.DataFrame({
        "Variable": ["followers", "video_duration_sec", "video_size_mb", "media_type",
                     "pet_type", "post_hour", "hashtags", "comments", "engagement"],
        "Type": ["int", "int", "float", "str", "str", "int", "int", "int", "int"],
        "Description": [
            "Number of creator followers at time of posting",
            "Video duration in seconds (0 for photos)",
            "Media file size in MB",
            "Media type: 'video' or 'photo'",
            "Pet type: 'dog', 'cat', 'bird', 'other'",
            "Hour of day the post was published (0-23)",
            "Number of hashtags used in the post",
            "Number of comments received",
            "Target variable — total post engagement score",
        ],
        "Range": [
            f"{df['followers'].min()}-{df['followers'].max()}",
            f"{df['video_duration_sec'].min()}-{df['video_duration_sec'].max()} sec",
            f"{df['video_size_mb'].min():.1f}-{df['video_size_mb'].max():.1f} MB",
            str(df['media_type'].unique().tolist()),
            str(df['pet_type'].unique().tolist()),
            "0-23",
            f"{df['hashtags'].min()}-{df['hashtags'].max()}",
            f"{df['comments'].min()}-{df['comments'].max()}",
            f"{df['engagement'].min()}-{df['engagement'].max()}",
        ],
    })
    st.dataframe(dic, use_container_width=True, hide_index=True)
    hr()

    st.subheader("Data Quality")
    st.success("Clean dataset: 0 null values across all 9 variables. 400 complete records ready for analysis.")
    qd = pd.DataFrame({
        "Issue": [
            "No temporal data (time series)",
            "media_type and pet_type as text (require encoding)",
            "No external data available (weather, seasonality)",
        ],
        "Dimension": ["Coverage", "Data type", "Coverage"],
        "Impact": ["Medium", "Low", "Low"],
        "Action Taken": [
            "Analysis restricted to available snapshot",
            "LabelEncoder applied before modeling",
            "Analysis focused on available features",
        ],
    })
    st.dataframe(qd, use_container_width=True, hide_index=True)
    hr()

    st.subheader("Descriptive Statistics")
    st.dataframe(
        df[["followers","comments","video_duration_sec","hashtags","video_size_mb","post_hour","engagement"]].describe().round(2),
        use_container_width=True
    )
    hr()

    st.subheader("Categorical Variable Distribution")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        mc = df["media_type"].value_counts()
        ax.bar(mc.index, mc.values, color=[MID_BLUE, LIGHT_BLUE], edgecolor="white", width=0.5)
        for i, v in enumerate(mc.values):
            ax.text(i, v+3, f"{v} ({v/len(df)*100:.0f}%)", ha="center", fontsize=10, fontweight="bold")
        ax.set_title("Distribution by Media Type", fontweight="bold", fontsize=11)
        ax.set_ylabel("Number of posts")
        ax.grid(True, alpha=0.3, axis="y"); ax.set_facecolor("#f9f9f9"); fig.tight_layout()
        st.pyplot(fig); plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        pc = df["pet_type"].value_counts()
        ax.bar(pc.index, pc.values, color=[MID_BLUE, LIGHT_BLUE, ACCENT, GOLD], edgecolor="white", width=0.55)
        for i, v in enumerate(pc.values):
            ax.text(i, v+2, f"{v} ({v/len(df)*100:.0f}%)", ha="center", fontsize=10, fontweight="bold")
        ax.set_title("Distribution by Pet Type", fontweight="bold", fontsize=11)
        ax.set_ylabel("Number of posts")
        ax.grid(True, alpha=0.3, axis="y"); ax.set_facecolor("#f9f9f9"); fig.tight_layout()
        st.pyplot(fig); plt.close()


# =============================================================================
# 3 — ENGAGEMENT ANALYSIS
# =============================================================================
elif page == "Engagement Analysis":
    phase_label("Phase 3 — Engagement Analysis")
    st.title("Engagement Analysis")
    hr()

    # Finding 1
    st.subheader("Finding 1 — Followers: the primary engagement driver")
    corr_f = df["followers"].corr(df["engagement"])
    col1, col2 = st.columns([3, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.scatter(df["followers"], df["engagement"], alpha=0.45,
                   c=df["engagement"], cmap="Blues", s=22, edgecolors="none")
        m, b = np.polyfit(df["followers"], df["engagement"], 1)
        xs = np.linspace(df["followers"].min(), df["followers"].max(), 300)
        ax.plot(xs, m*xs+b, color=ACCENT, lw=2.5, label=f"Trend (corr={corr_f:.2f})")
        ax.set_xlabel("Followers", fontsize=10); ax.set_ylabel("Engagement", fontsize=10)
        ax.set_title("Followers vs. Engagement", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_facecolor("#f9f9f9"); fig.tight_layout()
        st.pyplot(fig); plt.close()
    with col2:
        st.metric("Correlation", f"{corr_f:.2f}", "Strong positive")
        st.metric("Ranking", "#1")
        st.metric("6k-8k followers", "480 avg eng.", "+117% vs 0-1k")

    rec_box(
        f"Followers has the highest isolated correlation ({corr_f:.2f}). "
        "Creators with 6k-8k followers average 480 engagement — more than double the 222 avg for under 1k. "
        "The platform should invest in discovery tools to expose smaller creators to new audiences."
    )
    limit_box("Correlation does not imply causation. Unmeasured factors (content quality, posting frequency) may confound this relationship.")
    hr()

    # Finding 2
    st.subheader("Finding 2 — Comments: signal of deep engagement")
    corr_c = df["comments"].corr(df["engagement"])
    col1, col2 = st.columns([3, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.scatter(df["comments"], df["engagement"], alpha=0.45, color=LIGHT_BLUE, s=22, edgecolors="none")
        m, b = np.polyfit(df["comments"], df["engagement"], 1)
        xs = np.linspace(0, df["comments"].max(), 300)
        ax.plot(xs, m*xs+b, color=ACCENT, lw=2.5, label=f"Trend (corr={corr_c:.2f})")
        ax.set_xlabel("Number of Comments", fontsize=10); ax.set_ylabel("Engagement", fontsize=10)
        ax.set_title("Comments vs. Engagement", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_facecolor("#f9f9f9"); fig.tight_layout()
        st.pyplot(fig); plt.close()
    with col2:
        st.metric("Correlation", f"{corr_c:.2f}", "Strong positive")
        st.metric("Ranking", "#2")
        st.metric("Avg comments/post", f"{df['comments'].mean():.0f}")

    rec_box(
        f"Comments have the 2nd highest correlation with engagement ({corr_c:.2f}). "
        "High comment volume signals active interaction, not passive consumption. "
        "Features like 'Question of the Day', polls, and community challenges drive this organically."
    )
    hr()

    # Finding 3
    st.subheader("Finding 3 — Video outperforms photo by +9%")
    media_eng = df.groupby("media_type")["engagement"].mean().sort_values(ascending=False)
    col1, col2 = st.columns([2, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(media_eng.index, media_eng.values, color=[MID_BLUE, LIGHT_BLUE], edgecolor="white", width=0.45)
        for bar, val in zip(ax.patches, media_eng.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+4,
                    f"{val:.0f}", ha="center", fontsize=12, fontweight="bold", color=DARK_BLUE)
        ax.axhline(df["engagement"].mean(), color=ACCENT, lw=2, linestyle="--",
                   label=f"Overall avg: {df['engagement'].mean():.0f}")
        ax.set_ylabel("Average Engagement", fontsize=10)
        ax.set_title("Average Engagement by Media Type", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y"); ax.set_facecolor("#f9f9f9"); fig.tight_layout()
        st.pyplot(fig); plt.close()
    with col2:
        st.metric("Video avg", "388 eng.")
        st.metric("Photo avg", "356 eng.")
        st.metric("Difference", "+9%", "in favor of video")
        st.metric("Video posts", "234 (58.5%)")

    rec_box("Video generates 9% more engagement on average. Incentivizing photo creators to migrate to video would compound the effect.")
    hr()

    # Finding 4
    st.subheader("Finding 4 — Video duration has moderate positive impact")
    corr_d = df["video_duration_sec"].corr(df["engagement"])
    videos = df[df["media_type"] == "video"].copy()
    videos["dur_bucket"] = pd.cut(videos["video_duration_sec"], bins=[0,30,60,75,90],
                                   labels=["0-30s","30-60s","60-75s","75-90s"])
    dur_eng = videos.groupby("dur_bucket", observed=True)["engagement"].mean()
    col1, col2 = st.columns([3, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(dur_eng.index, dur_eng.values,
               color=["#aaccee","#4a90c4","#2d5a8e","#1a2e4a"], edgecolor="white", width=0.55)
        for bar, val in zip(ax.patches, dur_eng.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+3,
                    f"{val:.0f}", ha="center", fontsize=11, fontweight="bold")
        ax.set_xlabel("Duration bucket (videos only)", fontsize=10)
        ax.set_ylabel("Average Engagement", fontsize=10)
        ax.set_title("Engagement by Video Duration", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y"); ax.set_facecolor("#f9f9f9"); fig.tight_layout()
        st.pyplot(fig); plt.close()
    with col2:
        st.metric("Correlation", f"{corr_d:.2f}", "Moderate positive")
        st.metric("Ranking", "#3")
        st.metric("Best bucket", "75-90s")

    rec_box(f"Correlation between duration and engagement is {corr_d:.2f}. Videos between 75-90s show the highest average engagement — the audience values substantive content.")
    hr()

    # Finding 5
    st.subheader("Finding 5 — Posting time has near-zero influence")
    corr_h = df["post_hour"].corr(df["engagement"])
    hourly = df.groupby("post_hour")["engagement"].mean().reset_index()
    col1, col2 = st.columns([3, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(9, 4))
        mean_val = df["engagement"].mean()
        colors_h = [ACCENT if v > mean_val else GRAY for v in hourly["engagement"]]
        ax.bar(hourly["post_hour"], hourly["engagement"], color=colors_h, edgecolor="white", width=0.75)
        ax.axhline(mean_val, color=MID_BLUE, lw=2, linestyle="--", label=f"Overall avg: {mean_val:.0f}")
        ax.set_xlabel("Hour of Day", fontsize=10); ax.set_ylabel("Average Engagement", fontsize=10)
        ax.set_title("Average Engagement by Posting Hour", fontsize=12, fontweight="bold")
        ax.set_xticks(range(0,24)); ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")
        ax.set_facecolor("#f9f9f9"); fig.tight_layout()
        st.pyplot(fig); plt.close()
    with col2:
        st.metric("Correlation", f"{corr_h:.3f}", "Near-zero")
        st.metric("Ranking", "#5 (last)")
        st.metric("Peak hour", "1am", "481 avg eng.")

    rec_box(f"Posting time correlation is {corr_h:.3f} — essentially zero. The algorithm distributes content regardless of publication time. Creators should focus on content quality and audience growth, not timing.")
    limit_box("Without timezone data, cannot fully rule out posting time effects in specific geographic segments.")


# =============================================================================
# 4 — PREDICTIVE MODEL
# =============================================================================
elif page == "Predictive Model":
    phase_label("Phase 3 — Predictive Model")
    st.title("Predictive Model")
    st.markdown(f"Linear Regression trained on real PetNet data. **R2={R2:.2f} | MAE={MAE:.0f} pts**")
    hr()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R2", f"{R2:.4f}", "Strong predictive capability")
    c2.metric("MAE", f"{MAE:.1f} pts", "Mean absolute error")
    c3.metric("Train / test split", "80% / 20%")
    c4.metric("Algorithm", "Linear Regression")
    hr()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Model Coefficients")
        coefs = pd.Series(dict(zip(features, model.coef_))).sort_values(key=abs, ascending=True)
        colors_imp = [ACCENT if v > 0 else GRAY for v in coefs.values]
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.barh(coefs.index, coefs.values.tolist(), color=colors_imp, edgecolor="white", height=0.55)
        ax.axvline(0, color=DARK_BLUE, lw=1)
        ax.set_xlabel("Coefficient", fontsize=9)
        ax.set_title("Feature Impact on Engagement", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x"); ax.set_facecolor("#f9f9f9"); fig.tight_layout()
        st.pyplot(fig); plt.close()
        st.caption("Red = increases engagement | Gray = reduces engagement")

    with col2:
        st.subheader("Predicted vs. Real")
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.scatter(y_test, y_pred, alpha=0.5, color=MID_BLUE, s=25, edgecolors="none")
        lims = [min(float(y_test.min()), float(y_pred.min()))-20,
                max(float(y_test.max()), float(y_pred.max()))+20]
        ax.plot(lims, lims, color=ACCENT, lw=2, linestyle="--", label="Ideal line")
        ax.set_xlabel("Real Engagement", fontsize=10); ax.set_ylabel("Predicted Engagement", fontsize=10)
        ax.set_title(f"Predicted vs. Real (R2={R2:.2f})", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_facecolor("#f9f9f9"); fig.tight_layout()
        st.pyplot(fig); plt.close()

    rec_box(
        f"With R2={R2:.2f} and MAE={MAE:.0f} pts, the model is production-ready. "
        "media_enc has the highest coefficient (+22.7), confirming video beats photo. "
        "followers coefficient is 0.04 but operates on thousands — 1k extra followers = +40 pts estimated engagement."
    )
    hr()

    st.subheader("Correlations with Engagement")
    corr_series = df.corr(numeric_only=True)["engagement"].drop("engagement").sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 3))
    colors_corr = [ACCENT if v > 0.3 else (GOLD if v > 0 else GRAY) for v in corr_series.values]
    ax.bar(corr_series.index, corr_series.values, color=colors_corr, edgecolor="white", width=0.55)
    for bar, val in zip(ax.patches, corr_series.values):
        ypos = bar.get_height() + 0.005 if val >= 0 else bar.get_height() - 0.02
        ax.text(bar.get_x()+bar.get_width()/2, ypos,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.axhline(0, color=DARK_BLUE, lw=1)
    ax.set_ylabel("Pearson Correlation", fontsize=9)
    ax.set_title("Correlation of each variable with Engagement", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y"); ax.set_facecolor("#f9f9f9"); fig.tight_layout()
    st.pyplot(fig); plt.close()
    hr()

    st.subheader("Engagement Simulator")
    st.markdown("Adjust the features to estimate post engagement before publishing:")
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc5, sc6, sc7, sc8 = st.columns(4)
    with sc1: sim_fol  = st.slider("Followers",          100,  8000, 3000, 100)
    with sc2: sim_com  = st.slider("Comments",             0,   120,   30,   5)
    with sc3: sim_dur  = st.slider("Video duration (s)",   0,    90,   60,   5)
    with sc4: sim_hash = st.slider("Hashtags",             0,    15,    5,   1)
    with sc5: sim_hour = st.slider("Posting hour",         0,    23,   12,   1)
    with sc6: sim_size = st.slider("File size (MB)",     1.0, 200.0, 50.0, 1.0)
    with sc7: sim_media = st.selectbox("Media type", ["video", "photo"])
    with sc8: sim_pet   = st.selectbox("Pet type", ["dog", "cat", "bird", "other"])

    media_enc_val = le_media.transform([sim_media])[0]
    pet_enc_val   = le_pet.transform([sim_pet])[0]
    X_sim = np.array([[sim_fol, sim_com, sim_dur, sim_hash, sim_hour, media_enc_val, pet_enc_val, sim_size]])
    pred_val = model.predict(X_sim)[0]
    pct = (pred_val / df["engagement"].mean() - 1) * 100
    percentil = (df["engagement"] < pred_val).mean() * 100

    rc1, rc2, rc3 = st.columns(3)
    rc1.metric("Estimated Engagement", f"{pred_val:,.0f}", f"{pct:+.1f}% vs. overall avg")
    rc2.metric("Percentile", f"{percentil:.0f}%", "of real distribution")
    rc3.metric("Overall dataset avg", f"{df['engagement'].mean():.0f}", "for reference")


# =============================================================================
# 5 — ADDITIONAL INSIGHTS
# =============================================================================
elif page == "Additional Insights":
    phase_label("Phase 4 — Additional Insights")
    st.title("Additional Insights")
    st.markdown("Analyses beyond the main findings. Structure: **What the data shows → How much it matters → What to do.**")
    hr()

    st.subheader("Insight A — Engagement follows a Pareto pattern")
    sorted_eng = df["engagement"].sort_values(ascending=False).reset_index(drop=True)
    cum = sorted_eng.cumsum() / sorted_eng.sum()
    idx80 = int((cum >= 0.8).idxmax())
    col1, col2 = st.columns([3, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(range(1, len(cum)+1), cum*100, color=MID_BLUE, lw=2.5)
        ax.axhline(80, color=ACCENT, linestyle="--", lw=2, label="80% of total engagement")
        ax.axvline(idx80, color=ACCENT, linestyle=":", lw=1.5)
        ax.fill_between(range(1, idx80+2), (cum[:idx80+1]*100), alpha=0.15, color=ACCENT)
        ax.text(idx80+5, 40, f"Top {idx80/len(df)*100:.0f}%\nof posts",
                fontsize=9, color=ACCENT, fontweight="bold")
        ax.set_xlabel("Posts (sorted highest to lowest engagement)", fontsize=10)
        ax.set_ylabel("% of cumulative total engagement", fontsize=10)
        ax.set_title("Pareto Curve of Engagement", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_facecolor("#f9f9f9"); fig.tight_layout()
        st.pyplot(fig); plt.close()
    with col2:
        st.metric(f"Top {idx80/len(df)*100:.0f}% of posts", "80%", "of total engagement")
        st.metric("High-performance posts", f"{idx80}", f"out of {len(df)}")

    rec_box(f"The top {idx80} posts ({idx80/len(df)*100:.0f}% of dataset) concentrate 80% of total engagement. Classic Pareto profile. An acceleration program for mid-tier creators (1k-3k followers) can have outsized platform-wide impact.")
    hr()

    st.subheader("Insight B — Engagement grows consistently with follower tier")
    faixa_eng = df.groupby("follower_tier", observed=True)["engagement"].mean()
    col1, col2 = st.columns([2, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(faixa_eng.index, faixa_eng.values,
               color=["#aaccee","#4a90c4","#2d5a8e","#1a2e4a"], edgecolor="white", width=0.55)
        for bar, val in zip(ax.patches, faixa_eng.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+4,
                    f"{val:.0f}", ha="center", fontsize=12, fontweight="bold")
        ax.set_xlabel("Follower Tier", fontsize=10); ax.set_ylabel("Average Engagement", fontsize=10)
        ax.set_title("Average Engagement by Follower Tier", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y"); ax.set_facecolor("#f9f9f9"); fig.tight_layout()
        st.pyplot(fig); plt.close()
    with col2:
        st.metric("0-1k followers", "222 eng.", "baseline")
        st.metric("1k-3k followers", "285 eng.", "+28%")
        st.metric("3k-6k followers", "396 eng.", "+78%")
        st.metric("6k-8k followers", "480 eng.", "+117%")

    rec_box("Engagement growth is linear and consistent across follower tiers. Every step up generates real, increasing gains — validating audience growth as the primary strategic lever.")
    hr()

    st.subheader("Insight C — Birds and cats outperform dogs in average engagement")
    pet_eng = df.groupby("pet_type")["engagement"].mean().sort_values(ascending=False)
    col1, col2 = st.columns([2, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.bar(pet_eng.index, pet_eng.values, color=[ACCENT, MID_BLUE, LIGHT_BLUE, GOLD], edgecolor="white", width=0.5)
        for bar, val in zip(ax.patches, pet_eng.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2,
                    f"{val:.0f}", ha="center", fontsize=12, fontweight="bold")
        ax.axhline(df["engagement"].mean(), color=DARK_BLUE, lw=2, linestyle="--",
                   label=f"Overall avg: {df['engagement'].mean():.0f}")
        ax.set_ylabel("Average Engagement", fontsize=10)
        ax.set_title("Average Engagement by Pet Type", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y"); ax.set_facecolor("#f9f9f9"); fig.tight_layout()
        st.pyplot(fig); plt.close()
    with col2:
        st.metric("Bird", "379 eng.", "1st place")
        st.metric("Cat",  "377 eng.", "2nd place")
        st.metric("Dog",  "372 eng.", "4th place")

    rec_box("Birds and cats lead engagement even being less frequent than dogs. Niche content may generate more interaction due to a more engaged and less saturated audience.")
    limit_box("With only 39 bird posts and 22 'other', small samples may skew averages. This finding needs confirmation with more data.")
    hr()

    st.subheader("Insight D — Hashtags have minimal impact (corr=0.039)")
    corr_hash = df["hashtags"].corr(df["engagement"])
    col1, col2 = st.columns([3, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.scatter(df["hashtags"], df["engagement"], alpha=0.35, color=GRAY, s=18, edgecolors="none")
        m, b = np.polyfit(df["hashtags"], df["engagement"], 1)
        xs = np.linspace(0, df["hashtags"].max(), 200)
        ax.plot(xs, m*xs+b, color=ACCENT, lw=2.5, label=f"Trend (corr={corr_hash:.3f})")
        ax.set_xlabel("Number of Hashtags", fontsize=10); ax.set_ylabel("Engagement", fontsize=10)
        ax.set_title("Hashtags vs. Engagement", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_facecolor("#f9f9f9"); fig.tight_layout()
        st.pyplot(fig); plt.close()
    with col2:
        st.metric("Correlation", f"{corr_hash:.3f}", "Near-zero")
        st.metric("Ranking", "#4")
        st.metric("Conclusion", "No impact")

    rec_box("With a correlation of 0.039, hashtags have virtually no effect on PetNet engagement. Unlike Instagram or TikTok, PetNet's algorithm doesn't appear to use hashtags as a ranking signal. Creators should not invest strategic time in hashtag research.")


# =============================================================================
# 6 — RECOMMENDATIONS
# =============================================================================
elif page == "Recommendations":
    phase_label("Phase 5 — Prioritized Recommendations")
    st.title("Prioritized Recommendations")
    st.markdown("Recommendations based exclusively on data findings. Prioritized by **impact x implementation effort**.")
    hr()

    st.subheader("Priority Matrix")
    matrix = pd.DataFrame({
        "Recommendation": [
            "Audience growth tools",
            "Qualified interaction features",
            "Deploy predictive model as product",
            "Video content acceleration",
            "Predictive monitoring layer",
        ],
        "Impact": ["High", "High", "Medium", "Medium", "Medium"],
        "Effort": ["Medium", "Low", "Low", "Low", "Medium"],
        "Priority": ["#1 — Immediate", "#2 — Immediate", "#3 — Short term", "#4 — Short term", "#5 — Strategic"],
        "Timeline": ["3-6 months", "Next sprint", "30-60 days", "30-60 days", "90+ days"],
    })

    def color_imp(val):
        if val == "High":   return "background-color:#fde8e4;color:#c0392b;font-weight:bold"
        if val == "Medium": return "background-color:#fff3cd;color:#856404;font-weight:bold"
        return ""
    def color_prio(val):
        if "Immediate"   in val: return "background-color:#fde8e4;color:#c0392b;font-weight:bold"
        if "Short term"  in val: return "background-color:#fff3cd;color:#856404"
        return "background-color:#d4edda;color:#155724"

    st.dataframe(
        matrix.style.applymap(color_imp, subset=["Impact"])
                    .applymap(color_prio, subset=["Priority"]),
        use_container_width=True, hide_index=True
    )
    hr()

    prios = [
        ("#1 — IMMEDIATE: Audience Growth Tools",
         "Followers is the dominant driver (corr=0.73). Any action that increases followers has direct engagement return.",
         ["Develop creator recommendation system for users with similar interests",
          "Create 'PetNet Rising' program for creators between 1k-3k followers with algorithmic boost",
          "Implement new-follower notifications with related content suggestions"],
         "Halve the average time for a creator to go from 0 to 1,000 followers",
         "3-6 months", "Product + Algorithm"),
        ("#2 — IMMEDIATE: Qualified Interaction Features",
         "Comments are the 2nd strongest driver (corr=0.60). Features that incentivize comments have direct ROI.",
         ["Add 'Question of the Day' functionality for creator posts",
          "Implement pet-themed polls inside video posts",
          "Highlight posts with highest comment rates in the main feed"],
         "Increase average comment rate per post by 30% in 90 days",
         "Next sprint", "Product + UX"),
        (f"#3 — SHORT TERM: Deploy Predictive Model (R2={R2:.2f})",
         "High R2 indicates the model is production-ready. Converting it into a product creates immediate creator value.",
         ["Package engagement_model.pkl as REST API with /predict endpoint",
          "Integrate prediction in post composition screen as 'estimated engagement score'",
          "Collect creator feedback on accuracy for continuous retraining"],
         f"Model in production with <100ms latency | R2 maintained above {R2:.2f}",
         "30-60 days", "Engineering + Data Science"),
        ("#4 — SHORT TERM: Video Content Acceleration",
         "Video outperforms photo by +9% and video duration has moderate positive correlation (0.27).",
         ["Create video-first creator onboarding guide",
          "Build in-app video editor with engagement tips (duration, hooks)",
          "Feature long-form video (75-90s) in algorithm recommendation boost"],
         "Increase share of video posts from 58.5% to 70% in 6 months",
         "30-60 days", "Product + Content"),
        ("#5 — STRATEGIC: Predictive Monitoring Layer",
         "With a high-accuracy model available, real-time engagement monitoring becomes feasible.",
         ["Track weekly engagement distribution and flag anomalies",
          "Monitor feature importance drift over time",
          "Trigger alerts when low-engagement clusters spike"],
         "Reduce average time to detect engagement drops by 50%",
         "90+ days", "Data Engineering + ML"),
    ]

    for title, prob, acoes, kpi, prazo, team in prios:
        with st.expander(title, expanded=True):
            st.markdown(f"**Problem / Opportunity:** {prob}")
            st.markdown("**Actions:**")
            for i, a in enumerate(acoes, 1):
                st.markdown(f"  {i}. {a}")
            cols = st.columns(3)
            cols[0].metric("Success metric", kpi[:50]+"..." if len(kpi) > 50 else kpi)
            cols[1].metric("Suggested timeline", prazo)
            cols[2].metric("Responsible team", team)


# =============================================================================
# 7 — TECHNICAL APPENDIX
# =============================================================================
elif page == "Technical Appendix":
    phase_label("Technical Appendix")
    st.title("Technical Appendix")
    hr()

    st.subheader("Tech Stack")
    tech = pd.DataFrame({
        "Tool": ["Python", "Pandas & NumPy", "Matplotlib", "Scikit-learn", "Jupyter Notebook", "Streamlit"],
        "Use": ["Core language", "Data manipulation",
                "Exploratory visualizations",
                "Linear Regression model + .pkl serialization",
                "Interactive analysis (analysis.ipynb)",
                "Interactive dashboard (this app)"],
        "Version": ["3.10+", "-", "-", "-", "-", "-"],
    })
    st.dataframe(tech, use_container_width=True, hide_index=True)
    hr()

    st.subheader("Repository Structure")
    st.code("""
petnet-engagement-predictor/
|
|-- analysis.ipynb          # Full EDA + modeling + evaluation
|-- dataset.csv             # Engagement dataset (400 posts, 9 features, 0 nulls)
|-- engagement_model.pkl    # Serialized Linear Regression model
|-- app.py                  # Streamlit interactive dashboard (this file)
|-- requirements.txt        # Python dependencies
|-- README.md               # Project documentation
    """)
    hr()

    st.subheader("Model Performance — Details")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R2",  f"{R2:.4f}")
    c2.metric("MAE", f"{MAE:.2f} pts")
    c3.metric("Posts in dataset", "400")
    c4.metric("Features used", "8")

    coef_df = pd.DataFrame({
        "Feature": features,
        "Coefficient": [round(c, 4) for c in model.coef_],
        "Direction": ["Positive" if c > 0 else "Negative" for c in model.coef_],
        "Practical interpretation": [
            "Every 1,000 extra followers -> +40 est. engagement",
            "Every 10 extra comments -> +22 est. engagement",
            "Every 10s more of video -> +9 est. engagement",
            "Each extra hashtag -> +3 est. engagement",
            "Posting hour — weak negative effect",
            "Video adds ~23 pts of engagement vs photo",
            "Pet type — marginal effect",
            "File size — negligible negative effect",
        ],
    }).sort_values("Coefficient", key=abs, ascending=False)
    st.dataframe(coef_df, use_container_width=True, hide_index=True)
    hr()

    st.subheader("Model Improvement Roadmap")
    roadmap = pd.DataFrame({
        "Upgrade": ["Random Forest / XGBoost", "Feature: watch time",
                    "Feature: creator engagement history", "Cross-validation (k-fold)"],
        "Expected impact": ["Higher robustness and R2", "Better video quality signal",
                            "Captures creator consistency", "More reliable R2 estimate"],
        "Effort": ["Low", "Medium", "Medium", "Low"],
    })
    st.dataframe(roadmap, use_container_width=True, hide_index=True)
    hr()

    st.markdown("**GitHub:** https://github.com/Bruno-Barradas/petnet-engagement-predictor")
    st.markdown("**Dashboard:** https://petnet-engagement-predictor-hlcfiefqne9pacbgl8aemp.streamlit.app")
    st.markdown(f"**Dataset:** dataset.csv — 400 posts, 9 variables, 0 nulls")
    st.markdown(f"**Model:** engagement_model.pkl — Linear Regression, R2={R2:.2f}")
    st.caption("PetNet Engagement Predictor — Bruno Barradas · 2026")
