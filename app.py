import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PetNet Engagement Predictor",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
:root {
    --dark-blue:  #1a2e4a;
    --mid-blue:   #2d5a8e;
    --light-blue: #4a90c4;
    --accent:     #e8604c;
    --light-gray: #f5f7fa;
}
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
.finding-card.alto  { border-left-color: #e8604c; }
.finding-card.medio { border-left-color: #f0a500; }
.finding-card.baixo { border-left-color: #28a745; }
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


# ── Load & prepare data ──────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv")
    le_media = LabelEncoder()
    le_pet   = LabelEncoder()
    df["media_enc"] = le_media.fit_transform(df["media_type"])
    df["pet_enc"]   = le_pet.fit_transform(df["pet_type"])
    df["faixa_followers"] = pd.cut(
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
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2  = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return model, features, r2, mae, X_test, y_test, y_pred

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
    st.markdown(f'<div class="rec-box"><b>Analise e Recomendacao:</b> {txt}</div>',
                unsafe_allow_html=True)

def limit_box(txt):
    st.markdown(f'<div class="limit-box"><i>Limitacao:</i> {txt}</div>',
                unsafe_allow_html=True)

def hr():
    st.markdown("---")


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Petnet Analytics")
    st.markdown("---")
    page = st.radio("Navegacao", [
        "Apresentacao",
        "Entendimento dos Dados",
        "Analise de Engajamento",
        "Modelo Preditivo",
        "Insights Adicionais",
        "Recomendacoes",
        "Apendice Tecnico",
    ])
    st.markdown("---")
    st.markdown("**Projeto:** PetNet Engagement Predictor")
    st.markdown("**Autor:** Bruno Barradas 2025")
    st.markdown("[GitHub](https://github.com/Bruno-Barradas/petnet-engagement-predictor)")


# =============================================================================
# 1 - APRESENTACAO
# =============================================================================
if page == "Apresentacao":
    st.title("PetNet Engagement Predictor")
    st.markdown("### Analise de Engajamento em Plataforma Social para Pets")
    st.markdown(
        "Este projeto analisa os dados de posts da plataforma **PetNet** para identificar "
        "os fatores que mais influenciam o engajamento dos usuarios e construir um "
        "modelo preditivo de Regressao Linear. "
        "Todas as analises foram realizadas com dados reais da plataforma."
    )
    hr()

    st.subheader("Executive Summary")
    summary = pd.DataFrame({
        "Elemento": ["Objetivo", "Fonte dos dados", "Escopo", "Audiencia", "Entregavel"],
        "Descricao": [
            "Identificar fatores que influenciam o engajamento de posts na PetNet e construir modelo preditivo.",
            "Dataset CSV da plataforma PetNet (dataset.csv) com 400 posts e 9 variaveis.",
            "Posts com features de audiencia, conteudo, tipo de midia e comportamento.",
            "Criadores de conteudo, equipe de produto e analytics da PetNet.",
            "Analise descritiva + modelo preditivo (R2=0,95) com recomendacoes acionaveis.",
        ],
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)
    hr()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Posts analisados", "400")
    c2.metric("Variaveis", "9")
    c3.metric("R2 do Modelo", f"{R2:.2f}")
    c4.metric("MAE", f"{MAE:.0f} pts")
    c5.metric("Dados nulos", "0")
    hr()

    st.subheader("Os 5 Findings Principais")
    findings = [
        ("alto",  "1", "Followers e o principal driver de engajamento (corr=0,73)",
         "ALTO",  "Priorizar crescimento de base de seguidores nas estrategias de creators"),
        ("alto",  "2", "Comentarios sao o 2o driver - sinal de engajamento profundo (corr=0,60)",
         "ALTO",  "Criar features que incentivem discussoes: polls, perguntas do dia"),
        ("medio", "3", "Videos geram +9% mais engajamento que fotos",
         "MEDIO", "Orientar creators a priorizar conteudo em video"),
        ("medio", "4", "Duracao do video tem impacto moderado (corr=0,27)",
         "MEDIO", "Incentivar videos mais longos com conteudo substancial"),
        ("baixo", "5", "Horario de postagem tem correlacao negativa fraca (-0,075)",
         "BAIXO", "Nao priorizar horario - algoritmo distribui independente do horario"),
    ]

    for cls, num, title, impact, action in findings:
        badge_color = {"alto": ACCENT, "medio": GOLD, "baixo": GREEN}[cls]
        st.markdown(f"""
        <div class="finding-card {cls}">
            <b>Finding {num} - {title}</b><br>
            <span style="color:{badge_color};font-weight:700;font-size:0.85rem">Impacto: {impact}</span><br>
            <span style="font-size:0.9rem;color:#555">-> {action}</span>
        </div>
        """, unsafe_allow_html=True)

    hr()
    st.info(
        "**Conclusao executiva:** O engajamento na PetNet e altamente previsivel (R2=0,95) e "
        "concentrado em dois fatores estruturais: tamanho da audiencia e volume de comentarios. "
        "Video supera foto consistentemente. Horario de postagem e tamanho do arquivo tem "
        "influencia minima e nao devem ser priorizados nas estrategias de conteudo."
    )


# =============================================================================
# 2 - ENTENDIMENTO DOS DADOS
# =============================================================================
elif page == "Entendimento dos Dados":
    phase_label("Fase 2 - Entendimento dos Dados")
    st.title("Entendimento dos Dados")
    hr()

    st.subheader("Fontes de Dados")
    src = pd.DataFrame({
        "Fonte": ["Dataset PetNet", "Modelo serializado"],
        "Arquivo": ["dataset.csv", "engagement_model.pkl"],
        "Uso": ["400 posts com 9 features de audiencia, conteudo e comportamento",
                "Regressao Linear treinada para predicao de engajamento"],
    })
    st.dataframe(src, use_container_width=True, hide_index=True)
    hr()

    st.subheader("Dicionario de Variaveis")
    dic = pd.DataFrame({
        "Variavel": ["followers", "video_duration_sec", "video_size_mb", "media_type",
                     "pet_type", "post_hour", "hashtags", "comments", "engagement"],
        "Tipo": ["int", "int", "float", "str", "str", "int", "int", "int", "int"],
        "Descricao": [
            "Numero de seguidores do creator no momento do post",
            "Duracao do video em segundos (0 para fotos)",
            "Tamanho do arquivo de midia em MB",
            "Tipo de midia: video ou photo",
            "Tipo de pet: dog, cat, bird, other",
            "Hora do dia em que o post foi publicado (0-23)",
            "Quantidade de hashtags utilizadas no post",
            "Numero de comentarios recebidos",
            "Variavel alvo: engajamento total do post",
        ],
        "Range": [
            f"{df['followers'].min()}-{df['followers'].max()}",
            f"{df['video_duration_sec'].min()}-{df['video_duration_sec'].max()} seg",
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

    st.subheader("Qualidade dos Dados")
    st.success("Dataset limpo: 0 valores nulos em todas as 9 variaveis. 400 registros completos.")

    qd = pd.DataFrame({
        "Problema": [
            "Sem dados temporais (series historicas)",
            "media_type e pet_type como texto (precisam de encoding)",
            "Dados externos nao disponiveis (clima, sazonalidade)",
        ],
        "Dimensao": ["Cobertura", "Tipo de dado", "Cobertura"],
        "Impacto": ["Medio", "Baixo", "Baixo"],
        "Acao Tomada": [
            "Analise restrita ao snapshot disponivel",
            "LabelEncoder aplicado antes da modelagem",
            "Analise focada nas features disponiveis",
        ],
    })
    st.dataframe(qd, use_container_width=True, hide_index=True)
    hr()

    st.subheader("Estatisticas Descritivas")
    st.dataframe(
        df[["followers","comments","video_duration_sec","hashtags","video_size_mb","post_hour","engagement"]].describe().round(2),
        use_container_width=True
    )
    hr()

    st.subheader("Distribuicao das Variaveis Categoricas")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        media_counts = df["media_type"].value_counts()
        ax.bar(media_counts.index, media_counts.values,
               color=[MID_BLUE, LIGHT_BLUE], edgecolor="white", width=0.5)
        for i, v in enumerate(media_counts.values):
            ax.text(i, v + 3, f"{v} ({v/len(df)*100:.0f}%)", ha="center", fontsize=10, fontweight="bold")
        ax.set_title("Distribuicao por Tipo de Midia", fontweight="bold", fontsize=11)
        ax.set_ylabel("Quantidade de posts")
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_facecolor("#f9f9f9"); fig.tight_layout()
        st.pyplot(fig); plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        pet_counts = df["pet_type"].value_counts()
        ax.bar(pet_counts.index, pet_counts.values,
               color=[MID_BLUE, LIGHT_BLUE, ACCENT, GOLD], edgecolor="white", width=0.55)
        for i, v in enumerate(pet_counts.values):
            ax.text(i, v + 2, f"{v} ({v/len(df)*100:.0f}%)", ha="center", fontsize=10, fontweight="bold")
        ax.set_title("Distribuicao por Tipo de Pet", fontweight="bold", fontsize=11)
        ax.set_ylabel("Quantidade de posts")
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_facecolor("#f9f9f9"); fig.tight_layout()
        st.pyplot(fig); plt.close()


# =============================================================================
# 3 - ANALISE DE ENGAJAMENTO
# =============================================================================
elif page == "Analise de Engajamento":
    phase_label("Fase 3 - Analise: Engajamento na PetNet")
    st.title("Analise de Engajamento")
    hr()

    # Finding 1
    st.subheader("Finding 1 - Followers: o principal motor de engajamento")
    corr_f = df["followers"].corr(df["engagement"])
    col1, col2 = st.columns([3, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.scatter(df["followers"], df["engagement"],
                   alpha=0.45, c=df["engagement"], cmap="Blues", s=22, edgecolors="none")
        m, b = np.polyfit(df["followers"], df["engagement"], 1)
        xs = np.linspace(df["followers"].min(), df["followers"].max(), 300)
        ax.plot(xs, m*xs+b, color=ACCENT, lw=2.5, label=f"Tendencia (corr={corr_f:.2f})")
        ax.set_xlabel("Followers", fontsize=10)
        ax.set_ylabel("Engajamento", fontsize=10)
        ax.set_title("Followers vs. Engajamento", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        ax.set_facecolor("#f9f9f9"); fig.tight_layout()
        st.pyplot(fig); plt.close()
    with col2:
        st.metric("Correlacao", f"{corr_f:.2f}", "Positiva forte")
        st.metric("Ranking", "#1")
        st.metric("6k-8k seg.", "480 eng.", "+117% vs 0-1k")

    rec_box(
        f"Followers e o fator com maior correlacao isolada ({corr_f:.2f}). "
        "Posts de creators com 6k-8k seguidores atingem engajamento medio de 480, "
        "mais que o dobro dos creators com ate 1k (222). "
        "A plataforma deve investir em ferramentas de discovery para expor creators menores a novos publicos."
    )
    limit_box(
        "Correlacao nao implica causalidade. Outros fatores nao capturados no dataset "
        "(qualidade do conteudo, frequencia de postagem) podem confundir essa relacao."
    )
    hr()

    # Finding 2
    st.subheader("Finding 2 - Comentarios: sinal de engajamento profundo")
    corr_c = df["comments"].corr(df["engagement"])
    col1, col2 = st.columns([3, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.scatter(df["comments"], df["engagement"],
                   alpha=0.45, color=LIGHT_BLUE, s=22, edgecolors="none")
        m, b = np.polyfit(df["comments"], df["engagement"], 1)
        xs = np.linspace(0, df["comments"].max(), 300)
        ax.plot(xs, m*xs+b, color=ACCENT, lw=2.5, label=f"Tendencia (corr={corr_c:.2f})")
        ax.set_xlabel("Quantidade de Comentarios", fontsize=10)
        ax.set_ylabel("Engajamento", fontsize=10)
        ax.set_title("Comentarios vs. Engajamento", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        ax.set_facecolor("#f9f9f9"); fig.tight_layout()
        st.pyplot(fig); plt.close()
    with col2:
        st.metric("Correlacao", f"{corr_c:.2f}", "Positiva forte")
        st.metric("Ranking", "#2")
        st.metric("Media de comentarios", f"{df['comments'].mean():.0f}", "por post")

    rec_box(
        f"Comentarios tem a 2a maior correlacao com engajamento ({corr_c:.2f}). "
        "Posts com alto volume de comentarios indicam interacao ativa, nao consumo passivo. "
        "Funcionalidades como 'Pergunta do dia', polls e desafios de comunidade "
        "tendem a aumentar organicamente o volume de comentarios."
    )
    hr()

    # Finding 3
    st.subheader("Finding 3 - Video gera +9% mais engajamento que foto")
    media_eng = df.groupby("media_type")["engagement"].mean().sort_values(ascending=False)
    col1, col2 = st.columns([2, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(media_eng.index, media_eng.values,
                      color=[MID_BLUE, LIGHT_BLUE], edgecolor="white", width=0.45)
        for bar, val in zip(bars, media_eng.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+4,
                    f"{val:.0f}", ha="center", fontsize=12, fontweight="bold", color=DARK_BLUE)
        ax.axhline(df["engagement"].mean(), color=ACCENT, lw=2,
                   linestyle="--", label=f"Media geral: {df['engagement'].mean():.0f}")
        ax.set_ylabel("Engajamento Medio", fontsize=10)
        ax.set_title("Engajamento Medio por Tipo de Midia", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")
        ax.set_facecolor("#f9f9f9"); fig.tight_layout()
        st.pyplot(fig); plt.close()
    with col2:
        st.metric("Video", "388 eng.")
        st.metric("Foto", "356 eng.")
        st.metric("Diferenca", "+9%", "a favor do video")
        st.metric("Posts em video", "234 (58,5%)")

    rec_box(
        "Videos geram em media 9% mais engajamento que fotos. "
        "Com 58,5% dos posts ja em video, a plataforma favorece o formato, "
        "mas o gap pode ser aumentado incentivando creators de foto a migrarem. "
        "Tutoriais, rotinas diarias dos pets e videos longos tem maior potencial."
    )
    hr()

    # Finding 4
    st.subheader("Finding 4 - Duracao do video tem impacto moderado")
    corr_d = df["video_duration_sec"].corr(df["engagement"])
    videos = df[df["media_type"] == "video"].copy()
    videos["faixa_dur"] = pd.cut(videos["video_duration_sec"],
                                  bins=[0, 30, 60, 75, 90],
                                  labels=["0-30s", "30-60s", "60-75s", "75-90s"])
    dur_eng = videos.groupby("faixa_dur", observed=True)["engagement"].mean()

    col1, col2 = st.columns([3, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(dur_eng.index, dur_eng.values,
                      color=["#aaccee", "#4a90c4", "#2d5a8e", "#1a2e4a"],
                      edgecolor="white", width=0.55)
        for bar, val in zip(bars, dur_eng.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+3,
                    f"{val:.0f}", ha="center", fontsize=11, fontweight="bold")
        ax.set_xlabel("Faixa de duracao (apenas videos)", fontsize=10)
        ax.set_ylabel("Engajamento Medio", fontsize=10)
        ax.set_title("Engajamento por Faixa de Duracao do Video", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_facecolor("#f9f9f9"); fig.tight_layout()
        st.pyplot(fig); plt.close()
    with col2:
        st.metric("Correlacao", f"{corr_d:.2f}", "Moderada positiva")
        st.metric("Ranking", "#3")
        st.metric("Melhor faixa", "75-90s")

    rec_box(
        f"A correlacao entre duracao e engajamento e de {corr_d:.2f}, moderada mas consistente. "
        "Videos entre 75 e 90 segundos tem o maior engajamento medio. "
        "A audiencia da PetNet valoriza conteudo com substancia, "
        "nao apenas clips curtos."
    )
    hr()

    # Finding 5
    st.subheader("Finding 5 - Horario de postagem tem influencia fraca")
    corr_h = df["post_hour"].corr(df["engagement"])
    hourly = df.groupby("post_hour")["engagement"].mean().reset_index()

    col1, col2 = st.columns([3, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(9, 4))
        mean_val = df["engagement"].mean()
        colors_h = [ACCENT if v > mean_val else GRAY for v in hourly["engagement"]]
        ax.bar(hourly["post_hour"], hourly["engagement"],
               color=colors_h, edgecolor="white", width=0.75)
        ax.axhline(mean_val, color=MID_BLUE, lw=2, linestyle="--",
                   label=f"Media geral: {mean_val:.0f}")
        ax.set_xlabel("Hora de Postagem", fontsize=10)
        ax.set_ylabel("Engajamento Medio", fontsize=10)
        ax.set_title("Engajamento Medio por Hora de Postagem", fontsize=12, fontweight="bold")
        ax.set_xticks(range(0, 24))
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")
        ax.set_facecolor("#f9f9f9"); fig.tight_layout()
        st.pyplot(fig); plt.close()
    with col2:
        st.metric("Correlacao", f"{corr_h:.3f}", "Negativa fraca")
        st.metric("Ranking", "#5 (ultimo)")
        st.metric("Hora pico", "01h", "481 eng. medio")

    rec_box(
        f"O horario de postagem tem correlacao de apenas {corr_h:.3f} com o engajamento, "
        "praticamente zero. O algoritmo distribui o conteudo independente do horario de publicacao. "
        "Creators nao devem perder energia otimizando horario: o foco deve estar em qualidade "
        "de conteudo e crescimento da audiencia."
    )
    limit_box(
        "Sem dados de fuso horario ou segmentacao por pais, nao e possivel descartar "
        "completamente o efeito do horario em nichos geograficos especificos da plataforma."
    )


# =============================================================================
# 4 - MODELO PREDITIVO
# =============================================================================
elif page == "Modelo Preditivo":
    phase_label("Fase 3 - Modelo Preditivo")
    st.title("Modelo Preditivo de Engajamento")
    st.markdown(f"Regressao Linear treinada com dados reais da PetNet. **R2={R2:.2f} | MAE={MAE:.0f} pts**")
    hr()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R2", f"{R2:.4f}", "Alta capacidade preditiva")
    c2.metric("MAE", f"{MAE:.1f} pts", "Erro medio absoluto")
    c3.metric("Split treino/teste", "80% / 20%")
    c4.metric("Algoritmo", "Regressao Linear")
    hr()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Coeficientes do Modelo")
        coefs = pd.Series(dict(zip(features, model.coef_))).sort_values(key=abs, ascending=True)
        colors_imp = [ACCENT if v > 0 else GRAY for v in coefs.values]
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.barh(coefs.index, coefs.values.tolist(),
                color=colors_imp, edgecolor="white", height=0.55)
        ax.axvline(0, color=DARK_BLUE, lw=1)
        ax.set_xlabel("Coeficiente", fontsize=9)
        ax.set_title("Impacto de cada Feature no Engajamento", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x")
        ax.set_facecolor("#f9f9f9"); fig.tight_layout()
        st.pyplot(fig); plt.close()
        st.caption("Vermelho = aumenta engajamento | Cinza = reduz engajamento")

    with col2:
        st.subheader("Previsto vs. Real")
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.scatter(y_test, y_pred, alpha=0.5, color=MID_BLUE, s=25, edgecolors="none")
        lims = [min(float(y_test.min()), float(y_pred.min())) - 20,
                max(float(y_test.max()), float(y_pred.max())) + 20]
        ax.plot(lims, lims, color=ACCENT, lw=2, linestyle="--", label="Linha ideal")
        ax.set_xlabel("Engajamento Real", fontsize=10)
        ax.set_ylabel("Engajamento Previsto", fontsize=10)
        ax.set_title(f"Previsto vs. Real (R2={R2:.2f})", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        ax.set_facecolor("#f9f9f9"); fig.tight_layout()
        st.pyplot(fig); plt.close()

    rec_box(
        f"Com R2={R2:.2f} e MAE={MAE:.0f} pontos, o modelo esta pronto para uso em producao. "
        "O coeficiente de media_enc (tipo de midia) e o mais alto (22,7), confirmando que video supera foto. "
        "Followers tem coeficiente de 0,04 mas opera sobre valores de milhares: "
        "1.000 seguidores a mais equivalem a +40 pontos de engajamento estimado."
    )
    hr()

    st.subheader("Correlacoes com Engajamento")
    corr_series = df.corr(numeric_only=True)["engagement"].drop("engagement").sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 3))
    colors_corr = [ACCENT if v > 0.3 else (GOLD if v > 0 else GRAY) for v in corr_series.values]
    bars = ax.bar(corr_series.index, corr_series.values, color=colors_corr, edgecolor="white", width=0.55)
    for bar, val in zip(bars, corr_series.values):
        ypos = bar.get_height() + 0.005 if val >= 0 else bar.get_height() - 0.02
        ax.text(bar.get_x()+bar.get_width()/2, ypos,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.axhline(0, color=DARK_BLUE, lw=1)
    ax.set_ylabel("Correlacao de Pearson", fontsize=9)
    ax.set_title("Correlacao de cada variavel com Engajamento", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_facecolor("#f9f9f9"); fig.tight_layout()
    st.pyplot(fig); plt.close()
    hr()

    st.subheader("Simulador de Engajamento")
    st.markdown("Ajuste as features para estimar o engajamento de um post antes de publicar:")

    sc1, sc2, sc3, sc4 = st.columns(4)
    sc5, sc6, sc7, sc8 = st.columns(4)
    with sc1: sim_fol  = st.slider("Followers",           100,  8000, 3000, 100)
    with sc2: sim_com  = st.slider("Comentarios",           0,   120,   30,   5)
    with sc3: sim_dur  = st.slider("Duracao video (s)",     0,    90,   60,   5)
    with sc4: sim_hash = st.slider("Hashtags",              0,    15,    5,   1)
    with sc5: sim_hour = st.slider("Hora de postagem",      0,    23,   12,   1)
    with sc6: sim_size = st.slider("Tamanho arquivo (MB)", 1.0, 200.0, 50.0, 1.0)
    with sc7: sim_media = st.selectbox("Tipo de midia", ["video", "photo"])
    with sc8: sim_pet   = st.selectbox("Tipo de pet", ["dog", "cat", "bird", "other"])

    media_enc_val = le_media.transform([sim_media])[0]
    pet_enc_val   = le_pet.transform([sim_pet])[0]
    X_sim = np.array([[sim_fol, sim_com, sim_dur, sim_hash,
                        sim_hour, media_enc_val, pet_enc_val, sim_size]])
    pred_val = model.predict(X_sim)[0]
    pct = (pred_val / df["engagement"].mean() - 1) * 100
    percentil = (df["engagement"] < pred_val).mean() * 100

    rc1, rc2, rc3 = st.columns(3)
    rc1.metric("Engajamento Estimado", f"{pred_val:,.0f}", f"{pct:+.1f}% vs. media geral")
    rc2.metric("Percentil", f"{percentil:.0f}%", "da distribuicao real")
    rc3.metric("Media geral do dataset", f"{df['engagement'].mean():.0f}", "para referencia")


# =============================================================================
# 5 - INSIGHTS ADICIONAIS
# =============================================================================
elif page == "Insights Adicionais":
    phase_label("Fase 4 - Insights Adicionais")
    st.title("Insights Adicionais")
    st.markdown("Analises alem das perguntas principais. Estrutura: **O que os dados mostram -> Quanto vale -> O que fazer.**")
    hr()

    # Insight A
    st.subheader("Insight A - Engajamento tem perfil de Pareto")
    sorted_eng = df["engagement"].sort_values(ascending=False).reset_index(drop=True)
    cum = sorted_eng.cumsum() / sorted_eng.sum()
    idx80 = int((cum >= 0.8).idxmax())

    col1, col2 = st.columns([3, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(range(1, len(cum)+1), cum * 100, color=MID_BLUE, lw=2.5)
        ax.axhline(80, color=ACCENT, linestyle="--", lw=2, label="80% do engajamento total")
        ax.axvline(idx80, color=ACCENT, linestyle=":", lw=1.5)
        ax.fill_between(range(1, idx80+2), (cum[:idx80+1]*100), alpha=0.15, color=ACCENT)
        ax.text(idx80+5, 40, f"Top {idx80/len(df)*100:.0f}%\ndos posts",
                fontsize=9, color=ACCENT, fontweight="bold")
        ax.set_xlabel("Posts ordenados por engajamento (do maior ao menor)", fontsize=10)
        ax.set_ylabel("% acumulado do engajamento total", fontsize=10)
        ax.set_title("Curva de Pareto do Engajamento", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        ax.set_facecolor("#f9f9f9"); fig.tight_layout()
        st.pyplot(fig); plt.close()
    with col2:
        st.metric(f"Top {idx80/len(df)*100:.0f}% dos posts", "80%", "do engajamento total")
        st.metric("Posts de alta performance", f"{idx80}", f"de {len(df)}")

    rec_box(
        f"Os top {idx80} posts ({idx80/len(df)*100:.0f}% do dataset) concentram 80% do engajamento total. "
        "Perfil classico de Pareto. Criar um programa de aceleracao para creators em fase de "
        "crescimento (1k-3k seguidores) pode ter impacto desproporcional no engajamento medio da plataforma."
    )
    hr()

    # Insight B
    st.subheader("Insight B - Engajamento cresce consistentemente com faixa de followers")
    faixa_eng = df.groupby("faixa_followers", observed=True)["engagement"].mean()

    col1, col2 = st.columns([2, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(faixa_eng.index, faixa_eng.values,
                      color=["#aaccee", "#4a90c4", "#2d5a8e", "#1a2e4a"],
                      edgecolor="white", width=0.55)
        for bar, val in zip(bars, faixa_eng.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+4,
                    f"{val:.0f}", ha="center", fontsize=12, fontweight="bold")
        ax.set_xlabel("Faixa de Followers", fontsize=10)
        ax.set_ylabel("Engajamento Medio", fontsize=10)
        ax.set_title("Engajamento Medio por Faixa de Audiencia", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_facecolor("#f9f9f9"); fig.tight_layout()
        st.pyplot(fig); plt.close()
    with col2:
        st.metric("0-1k followers", "222 eng.", "linha de base")
        st.metric("1k-3k followers", "285 eng.", "+28%")
        st.metric("3k-6k followers", "396 eng.", "+78%")
        st.metric("6k-8k followers", "480 eng.", "+117%")

    rec_box(
        "O crescimento de engajamento com a faixa de followers e linear e consistente: "
        "cada salto de faixa representa ganho real e crescente. "
        "Isso valida a estrategia de priorizar crescimento de audiencia como alavanca principal."
    )
    hr()

    # Insight C
    st.subheader("Insight C - Passaros e gatos superam cachorros em engajamento medio")
    pet_eng = df.groupby("pet_type")["engagement"].mean().sort_values(ascending=False)

    col1, col2 = st.columns([2, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        bars = ax.bar(pet_eng.index, pet_eng.values,
                      color=[ACCENT, MID_BLUE, LIGHT_BLUE, GOLD], edgecolor="white", width=0.5)
        for bar, val in zip(bars, pet_eng.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2,
                    f"{val:.0f}", ha="center", fontsize=12, fontweight="bold")
        ax.axhline(df["engagement"].mean(), color=DARK_BLUE, lw=2,
                   linestyle="--", label=f"Media geral: {df['engagement'].mean():.0f}")
        ax.set_ylabel("Engajamento Medio", fontsize=10)
        ax.set_title("Engajamento Medio por Tipo de Pet", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis="y")
        ax.set_facecolor("#f9f9f9"); fig.tight_layout()
        st.pyplot(fig); plt.close()
    with col2:
        st.metric("Bird", "379 eng.", "1o lugar")
        st.metric("Cat",  "377 eng.", "2o lugar")
        st.metric("Dog",  "372 eng.", "4o lugar")

    rec_box(
        "Passaros e gatos lideram o engajamento mesmo sendo menos frequentes que cachorros. "
        "Isso pode indicar que conteudo de nicho gera mais interacao por ter audiencia "
        "mais engajada e menos saturada de conteudo similar."
    )
    limit_box(
        "Com apenas 39 posts de passaros e 22 de 'others', amostras menores podem "
        "distorcer as medias. Este finding precisa ser confirmado com mais dados."
    )
    hr()

    # Insight D
    st.subheader("Insight D - Hashtags tem impacto minimo (corr=0,039)")
    corr_hash = df["hashtags"].corr(df["engagement"])

    col1, col2 = st.columns([3, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.scatter(df["hashtags"], df["engagement"],
                   alpha=0.35, color=GRAY, s=18, edgecolors="none")
        m, b = np.polyfit(df["hashtags"], df["engagement"], 1)
        xs = np.linspace(0, df["hashtags"].max(), 200)
        ax.plot(xs, m*xs+b, color=ACCENT, lw=2.5, label=f"Tendencia (corr={corr_hash:.3f})")
        ax.set_xlabel("Quantidade de Hashtags", fontsize=10)
        ax.set_ylabel("Engajamento", fontsize=10)
        ax.set_title("Hashtags vs. Engajamento", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        ax.set_facecolor("#f9f9f9"); fig.tight_layout()
        st.pyplot(fig); plt.close()
    with col2:
        st.metric("Correlacao", f"{corr_hash:.3f}", "Quase zero")
        st.metric("Ranking", "#4")
        st.metric("Conclusao", "Nao impacta")

    rec_box(
        "Com correlacao de 0,039, hashtags praticamente nao afetam o engajamento na PetNet. "
        "O algoritmo da plataforma aparentemente nao usa hashtags como fator de ranqueamento. "
        "Creators nao devem desperdicar tempo estrategico com hashtag research para a PetNet."
    )


# =============================================================================
# 6 - RECOMENDACOES
# =============================================================================
elif page == "Recomendacoes":
    phase_label("Fase 5 - Recomendacoes Priorizadas")
    st.title("Recomendacoes Priorizadas")
    st.markdown("Recomendacoes baseadas exclusivamente nos findings dos dados. Priorizadas por **impacto x facilidade de implementacao**.")
    hr()

    st.subheader("Matriz de Prioridade")
    matrix = pd.DataFrame({
        "Recomendacao": [
            "Ferramentas de crescimento de audiencia",
            "Funcionalidades de interacao qualificada",
            "Deploy do modelo preditivo como produto",
            "Expandir dataset com novas variaveis",
        ],
        "Impacto": ["Alto", "Alto", "Medio", "Medio"],
        "Esforco": ["Medio", "Baixo", "Baixo", "Medio"],
        "Prioridade": ["#1 - Imediata", "#2 - Imediata", "#3 - Curto prazo", "#4 - Estrategica"],
        "Prazo": ["3-6 meses", "Proximo sprint", "30-60 dias", "60-90 dias"],
    })

    def color_imp(val):
        if val == "Alto":  return "background-color:#fde8e4;color:#c0392b;font-weight:bold"
        if val == "Medio": return "background-color:#fff3cd;color:#856404;font-weight:bold"
        return ""
    def color_prio(val):
        if "Imediata"    in val: return "background-color:#fde8e4;color:#c0392b;font-weight:bold"
        if "Curto prazo" in val: return "background-color:#fff3cd;color:#856404"
        return "background-color:#d4edda;color:#155724"

    st.dataframe(
        matrix.style.applymap(color_imp, subset=["Impacto"])
                    .applymap(color_prio, subset=["Prioridade"]),
        use_container_width=True, hide_index=True
    )
    hr()

    prios = [
        ("#1 - IMEDIATA: Ferramentas de crescimento de audiencia",
         "Followers e o fator dominante (corr=0,73). Qualquer acao que aumente seguidores tera retorno direto.",
         ["Desenvolver sistema de recomendacao de creators para usuarios com interesses similares",
          "Criar programa PetNet Rising para creators entre 1k-3k seguidores com boost algoritmico",
          "Implementar notificacoes de novo seguidor com sugestoes de conteudo relacionado"],
         "Reduzir o tempo medio para um creator sair de 0 para 1.000 seguidores pela metade",
         "3-6 meses", "Produto + Algoritmo"),
        ("#2 - IMEDIATA: Funcionalidades de interacao qualificada",
         "Comentarios sao o 2o driver mais forte (corr=0,60). Features que incentivem comentarios tem ROI direto.",
         ["Adicionar funcionalidade Pergunta do dia para posts de creators",
          "Implementar Polls sobre pets nos posts de video",
          "Criar destaque para posts com maior taxa de comentarios no feed"],
         "Aumentar taxa media de comentarios por post em 30% em 90 dias",
         "Proximo sprint", "Produto + UX"),
        (f"#3 - CURTO PRAZO: Deploy do modelo preditivo (R2={R2:.2f})",
         "R2 alto indica que o modelo esta pronto. Transforma-lo em produto gera valor imediato para creators.",
         ["Empacotar engagement_model.pkl como API REST com endpoint /predict",
          "Integrar predicao na tela de composicao do post como engagement score estimado",
          "Coletar feedback dos creators sobre a acuracia para retreinamento continuo"],
         f"Modelo em producao com latencia < 100ms | R2 mantido acima de {R2:.2f}",
         "30-60 dias", "Engenharia + Data Science"),
        ("#4 - ESTRATEGICA: Expandir dataset com novas variaveis",
         "O modelo atual tem lacunas que poderiam melhorar a performance preditiva.",
         ["Adicionar historico de engajamento por creator (media dos ultimos 30 dias)",
          "Coletar dados de tempo de visualizacao para videos",
          "Implementar auto-tagging de categoria de pet via visao computacional"],
         f"Aumentar R2 de {R2:.2f} para acima de 0,98 com as novas features",
         "60-90 dias", "Data Engineering + ML"),
    ]

    for title, prob, acoes, kpi, prazo, team in prios:
        with st.expander(title, expanded=True):
            st.markdown(f"**Problema / Oportunidade:** {prob}")
            st.markdown("**Acoes:**")
            for i, a in enumerate(acoes, 1):
                st.markdown(f"  {i}. {a}")
            cols = st.columns(3)
            cols[0].metric("Metrica de sucesso", kpi[:50]+"..." if len(kpi) > 50 else kpi)
            cols[1].metric("Prazo sugerido", prazo)
            cols[2].metric("Time responsavel", team)


# =============================================================================
# 7 - APENDICE TECNICO
# =============================================================================
elif page == "Apendice Tecnico":
    phase_label("Apendice Tecnico")
    st.title("Apendice Tecnico")
    hr()

    st.subheader("Tecnologias e Ferramentas")
    tech = pd.DataFrame({
        "Ferramenta": ["Python", "Pandas & NumPy", "Matplotlib",
                       "Scikit-learn", "Jupyter Notebook", "Streamlit"],
        "Uso": ["Linguagem principal", "Manipulacao de dados",
                "Visualizacoes exploratórias",
                "Modelo de Regressao Linear + serializacao .pkl",
                "Analise interativa (analysis.ipynb)",
                "Dashboard interativo (este app)"],
        "Versao": ["3.10+", "-", "-", "-", "-", "-"],
    })
    st.dataframe(tech, use_container_width=True, hide_index=True)
    hr()

    st.subheader("Estrutura do Repositorio")
    st.code("""
petnet-engagement-predictor/
|
|-- analysis.ipynb          # EDA + modelagem + avaliacao
|-- dataset.csv             # Dataset de engajamento (400 posts, 9 features)
|-- engagement_model.pkl    # Modelo de Regressao Linear serializado
|-- app.py                  # Dashboard Streamlit (este arquivo)
|-- requirements.txt        # Dependencias Python
|-- README.md               # Documentacao do projeto
    """)
    hr()

    st.subheader("Performance do Modelo - Detalhes")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R2",  f"{R2:.4f}")
    c2.metric("MAE", f"{MAE:.2f} pts")
    c3.metric("Posts no dataset", "400")
    c4.metric("Features usadas", "8")

    coef_df = pd.DataFrame({
        "Feature": features,
        "Coeficiente": [round(c, 4) for c in model.coef_],
        "Direcao": ["Positivo" if c > 0 else "Negativo" for c in model.coef_],
        "Interpretacao": [
            "A cada 1.000 followers a mais -> +40 eng. estimado",
            "A cada 10 comentarios a mais -> +22 eng. estimado",
            "A cada 10s a mais de video -> +9 eng. estimado",
            "A cada hashtag a mais -> +3 eng. estimado",
            "Hora de postagem - efeito negativo fraco",
            "Video supera foto em ~23 pts de engajamento",
            "Tipo de pet - efeito marginal",
            "Tamanho do arquivo - efeito negativo desprezivel",
        ],
    }).sort_values("Coeficiente", key=abs, ascending=False)
    st.dataframe(coef_df, use_container_width=True, hide_index=True)
    hr()

    st.subheader("Roadmap de Melhorias do Modelo")
    roadmap = pd.DataFrame({
        "Upgrade": ["Random Forest / XGBoost", "Feature: tempo de visualizacao",
                    "Feature: historico de eng. do creator", "Cross-validation (k-fold)"],
        "Impacto esperado": ["Maior robustez e R2 mais alto", "Melhor sinal de qualidade do video",
                             "Captura da consistencia do creator", "Estimativa de R2 mais confiavel"],
        "Esforco": ["Baixo", "Medio", "Medio", "Baixo"],
    })
    st.dataframe(roadmap, use_container_width=True, hide_index=True)
    hr()

    st.markdown("**GitHub:** https://github.com/Bruno-Barradas/petnet-engagement-predictor")
    st.markdown("**Dataset:** dataset.csv - 400 posts, 9 variaveis, 0 nulos")
    st.markdown(f"**Modelo:** engagement_model.pkl - Regressao Linear, R2={R2:.2f}")
    st.caption("PetNet Engagement Predictor - Bruno Barradas 2025")
