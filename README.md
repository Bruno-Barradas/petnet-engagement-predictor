# 🐾 PetNet Engagement Predictor

> Machine Learning model that predicts post engagement on a social pet platform — EDA, feature analysis, and Linear Regression with **R²=0.97**.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML_Model-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)

---

## 📌 Overview

PetNet is a social platform for pet owners where users share photos, videos, and updates about their animals. Understanding what drives engagement is critical for content creators and platform growth teams.

This project builds a **predictive model for post engagement** using behavioral and content features, identifying which factors most influence how users interact with posts.

---

## 📊 Key Results

| Metric | Value |
|--------|-------|
| 🎯 Model R² | **0.97** — strong predictive capability |
| 📉 MAE | **~18 engagement points** |
| 🔍 Top Driver | Audience size (strongest correlation) |
| 💬 2nd Driver | Comment count (high interaction signal) |
| 🎥 3rd Driver | Video duration (moderate impact) |
| 🕐 Weakest Signal | Posting time (weak influence) |

---

## 🔍 Key Findings

### 1. Audience Size is the Primary Engagement Driver
Follower count has the strongest correlation with engagement. Larger audiences amplify reach — content quality matters less than distribution scale at this stage.

### 2. Comments Signal Deep Engagement
Posts with higher comment counts show significantly higher total engagement. Comments indicate active interaction, not passive consumption — a strong quality signal.

### 3. Video Duration Has Moderate Impact
Longer videos correlate moderately with engagement, suggesting that content depth matters to the PetNet audience — users watch and interact more with substantive content.

### 4. Posting Time Shows Weak Influence
Time of posting has minimal predictive power in this dataset, suggesting the algorithm or notification system distributes content regardless of posting time.

### 5. Stable Engagement Distribution
No extreme outliers detected — engagement distribution is consistent across the dataset, making it well-suited for linear modeling.

---

## 🧠 Methodology

```
Raw Data → EDA → Feature Analysis → Correlation Study → Model Training → Evaluation
```

| Step | Technique | Output |
|------|-----------|--------|
| Exploratory Analysis | Pandas, Matplotlib | Distribution plots, summary stats |
| Correlation Analysis | Pearson correlation | Feature importance ranking |
| Feature Engineering | Scikit-learn | Scaled features |
| Model Training | Linear Regression | Trained model (.pkl) |
| Evaluation | MAE, R² | Performance metrics |

---

## 🤖 Model Details

**Algorithm:** Linear Regression (scikit-learn)  
**Target:** Post engagement score (continuous)  
**Split:** 80% train / 20% test  

| Feature | Impact Direction | Strength |
|---------|-----------------|----------|
| `audience_size` | Positive | Strong |
| `comment_count` | Positive | Strong |
| `video_duration` | Positive | Moderate |
| `posting_time` | Variable | Weak |

---

## ⚡ Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/Bruno-Barradas/petnet-engagement-predictor
cd petnet-engagement-predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Open the notebook
jupyter notebook analysis.ipynb
```

---

## 📁 Project Structure

```
petnet-engagement-predictor/
│
├── analysis.ipynb          # Full analysis: EDA + modeling + evaluation
├── dataset.csv             # Platform engagement dataset
├── engagement_model.pkl    # Trained Linear Regression model
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## 🛠️ Tech Stack

- **Python 3.10+** — core language
- **Pandas & NumPy** — data manipulation
- **Matplotlib** — visualizations
- **Scikit-learn** — machine learning model

---

## 📈 Improvement Roadmap

| Upgrade | Expected Impact | Effort |
|---------|----------------|--------|
| Random Forest / XGBoost | Higher robustness | Low |
| Feature: hashtag count | Better content signals | Low |
| Feature: image vs video | Media type impact | Low |
| Cross-validation (k-fold) | More reliable R² estimate | Low |
| Time-series analysis | Trend detection | Medium |

---

## 💼 Business Applications

- **Content Strategy:** Guide creators on which factors to optimize for reach
- **Algorithm Design:** Weight features that correlate with healthy engagement
- **Creator Tools:** Build a score predictor before posting
- **Growth Analytics:** Track engagement drivers over time as platform scales

---

## 📄 License

MIT License — free to use, modify, and distribute with attribution.

---

**Built with** Python · Scikit-learn · Pandas · Matplotlib  
*PetNet Engagement Predictor — Social Platform Analytics*
