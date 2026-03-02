"""
app.py — RAG Confidence Dashboard
Run with:   streamlit run app.py
"""

import streamlit as st
import plotly.graph_objects as go
import time
import sys
import os

# ─────────────────────────────────────────────
# PAGE CONFIG (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Intelligence Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# INJECT CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Google Font ── */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] {
      font-family: 'Inter', sans-serif;
  }

  /* ── Dark gradient background ── */
  .stApp {
      background: linear-gradient(135deg, #0d1117 0%, #161b22 60%, #0d1117 100%);
      color: #e6edf3;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
      background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
      border-right: 1px solid #30363d;
  }
  [data-testid="stSidebar"] * { color: #c9d1d9 !important; }
  [data-testid="stSidebarNav"] { display: none; }

  /* ── Card component ── */
  .rag-card {
      background: linear-gradient(145deg, #1c2128, #21262d);
      border: 1px solid #30363d;
      border-radius: 14px;
      padding: 24px 28px;
      margin-bottom: 16px;
      box-shadow: 0 4px 24px rgba(0,0,0,0.4);
      transition: box-shadow 0.25s ease;
  }
  .rag-card:hover { box-shadow: 0 6px 32px rgba(0,0,0,0.6); }

  .card-title {
      font-size: 0.72rem;
      font-weight: 600;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: #8b949e;
      margin-bottom: 10px;
  }

  /* ── Answer text ── */
  .answer-text {
      font-size: 1.05rem;
      line-height: 1.75;
      color: #c9d1d9;
  }

  /* ── Confidence badges ── */
  .badge {
      display: inline-block;
      padding: 4px 14px;
      border-radius: 20px;
      font-size: 0.8rem;
      font-weight: 700;
      letter-spacing: 0.06em;
      text-transform: uppercase;
  }
  .badge-high   { background: #1a4731; color: #3fb950; border: 1px solid #238636; }
  .badge-medium { background: #3d2a00; color: #d29922; border: 1px solid #9e6a03; }
  .badge-low    { background: #3d0c0c; color: #f85149; border: 1px solid #da3633; }

  /* ── Fallback banner ── */
  .fallback-banner {
      background: linear-gradient(90deg, #3d0c0c, #21262d);
      border: 1px solid #da3633;
      border-radius: 10px;
      padding: 12px 20px;
      color: #f85149;
      font-weight: 600;
      font-size: 0.9rem;
      display: flex;
      align-items: center;
      gap: 10px;
      margin-bottom: 16px;
  }

  /* ── Query text area ── */
  textarea {
      background: #161b22 !important;
      color: #e6edf3 !important;
      border: 1px solid #30363d !important;
      border-radius: 10px !important;
      font-family: 'Inter', sans-serif !important;
  }
  textarea:focus { border-color: #58a6ff !important; box-shadow: 0 0 0 3px rgba(88,166,255,0.15) !important; }

  /* ── Submit button ── */
  div[data-testid="stButton"] > button {
      background: linear-gradient(135deg, #238636, #2ea043);
      color: #ffffff;
      border: none;
      border-radius: 8px;
      font-size: 0.95rem;
      font-weight: 600;
      padding: 0.55rem 2rem;
      width: 100%;
      transition: filter 0.2s ease, transform 0.1s ease;
  }
  div[data-testid="stButton"] > button:hover  { filter: brightness(1.15); transform: translateY(-1px); }
  div[data-testid="stButton"] > button:active { transform: translateY(0); }

  /* ── Expander ── */
  details {
      background: #161b22;
      border: 1px solid #30363d !important;
      border-radius: 10px !important;
      padding: 4px 12px;
  }
  summary { color: #58a6ff !important; font-weight: 600; }

  /* ── Metric labels ── */
  [data-testid="metric-container"] {
      background: #1c2128;
      border: 1px solid #30363d;
      border-radius: 10px;
      padding: 14px 18px;
  }
  [data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 0.75rem !important; }
  [data-testid="stMetricValue"] { color: #e6edf3 !important; font-size: 1.5rem !important; }
  [data-testid="stMetricDelta"] { font-size: 0.82rem !important; }

  /* ── Progress bar ── */
  .stProgress > div > div {
      border-radius: 8px;
      height: 10px !important;
  }

  /* ── Divider ── */
  hr { border-color: #30363d !important; margin: 24px 0; }

  /* ── Source item ── */
  .source-item {
      background: #161b22;
      border: 1px solid #30363d;
      border-radius: 8px;
      padding: 10px 16px;
      margin-bottom: 8px;
      font-size: 0.88rem;
      color: #8b949e;
      display: flex;
      align-items: center;
      gap: 10px;
  }
  .source-icon { font-size: 1.1rem; }

  /* ── Sidebar section header ── */
  .sidebar-heading {
      font-size: 0.68rem;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      color: #58a6ff;
      font-weight: 700;
      margin: 20px 0 6px;
  }
  .sidebar-body {
      font-size: 0.83rem;
      color: #8b949e;
      line-height: 1.6;
  }

  /* ── Hide Streamlit default decoration ── */
  #MainMenu, header, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def badge_html(label: str) -> str:
    cls = {"High": "badge-high", "Medium": "badge-medium", "Low": "badge-low"}.get(label, "badge-low")
    icon = {"High": "✅", "Medium": "⚠️", "Low": "❌"}.get(label, "❌")
    return f'<span class="badge {cls}">{icon} {label}</span>'


def confidence_color(label: str) -> str:
    return {"High": "#3fb950", "Medium": "#d29922", "Low": "#f85149"}.get(label, "#f85149")


def build_similarity_chart(sources: list, scores: list, conf_label: str) -> go.Figure:
    bar_color = confidence_color(conf_label)
    # Deduplicate per unique (source, score) pair for display
    labels = [f"Doc {i+1}: {s}" for i, s in enumerate(sources)]

    fig = go.Figure(go.Bar(
        x=scores,
        y=labels,
        orientation="h",
        marker=dict(
            color=scores,
            colorscale=[
                [0.0,  "#3d0c0c"],
                [0.45, "#3d2a00"],
                [0.75, "#1a4731"],
                [1.0,  "#3fb950"],
            ],
            cmin=0.0,
            cmax=1.0,
            line=dict(color="#30363d", width=1),
        ),
        text=[f"{s:.3f}" for s in scores],
        textposition="outside",
        textfont=dict(color="#c9d1d9", size=12),
        hovertemplate="<b>%{y}</b><br>Similarity: %{x:.4f}<extra></extra>",
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=60, t=14, b=10),
        xaxis=dict(
            range=[0, 1.12],
            tickfont=dict(color="#8b949e", size=11),
            gridcolor="#21262d",
            zerolinecolor="#30363d",
            title=dict(text="Cosine Similarity", font=dict(color="#8b949e", size=12)),
        ),
        yaxis=dict(
            tickfont=dict(color="#c9d1d9", size=11),
            automargin=True,
        ),
        height=max(160, 70 * len(scores)),
        font=dict(family="Inter, sans-serif"),
    )
    return fig


# ─────────────────────────────────────────────
# RAG SYSTEM — CACHED INIT
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_rag_system(data_folder: str):
    """Load and cache the RAG system across reruns."""
    from rr import RAGWithConfidence
    return RAGWithConfidence(data_folder=data_folder)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🧠 RAG Intelligence")
    st.markdown("---")

    # Docs folder config
    st.markdown('<div class="sidebar-heading">⚙️ Configuration</div>', unsafe_allow_html=True)
    docs_folder = st.text_input("Documents folder", value="./docs", key="docs_folder")

    st.markdown("---")
    st.markdown('<div class="sidebar-heading">📐 How Confidence Works</div>', unsafe_allow_html=True)

    with st.expander("📊 Retrieval Score", expanded=False):
        st.markdown("""
<div class="sidebar-body">
The <b>base similarity score</b> is the mean cosine similarity
between your query embedding and the top-<i>k</i> retrieved
document chunks. Higher = stronger semantic match.
</div>
""", unsafe_allow_html=True)

    with st.expander("📉 Variance Detection", expanded=False):
        st.markdown("""
<div class="sidebar-body">
The system measures the <b>spread</b> across retrieved chunk
scores (max − min). A large spread suggests inconsistent
retrieval quality and incurs a <b>variance penalty</b> of
<code>variance × 0.2</code>.
</div>
""", unsafe_allow_html=True)

    with st.expander("🔀 Cross-Check Logic", expanded=False):
        st.markdown("""
<div class="sidebar-body">
After generating an answer, the system computes the
<b>cosine similarity between the answer embedding and the
aggregated context embedding</b>. If this value is below
0.30, a <b>mismatch penalty of 0.20</b> is applied — guarding
against hallucination.
</div>
""", unsafe_allow_html=True)

    with st.expander("⛔ No-Answer / Fallback Mode", expanded=False):
        st.markdown("""
<div class="sidebar-body">
If the <b>base retrieval score falls below 0.25</b>,
the system triggers <b>fallback mode</b>: it replaces
the generated answer with a safe refusal message and
forces the confidence label to <b>Low</b>.
This prevents fabricated low-quality answers.
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="sidebar-heading">🎨 Confidence Legend</div>', unsafe_allow_html=True)
    st.markdown("""
<div class="sidebar-body">
<span style="color:#3fb950;">●</span> <b>High</b>  — score &gt; 0.75<br>
<span style="color:#d29922;">●</span> <b>Medium</b> — score 0.50–0.75<br>
<span style="color:#f85149;">●</span> <b>Low</b>    — score &lt; 0.50
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="sidebar-body" style="font-size:0.75rem; color:#484f58;">RAG Confidence Dashboard · v1.0</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MAIN DASHBOARD
# ─────────────────────────────────────────────

# ── Header ──────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 32px 0 20px;">
  <h1 style="font-size:2.2rem; font-weight:700; color:#e6edf3; margin:0;">
    🧠 RAG Intelligence Dashboard
  </h1>
  <p style="color:#8b949e; font-size:1rem; margin-top:8px;">
    Retrieval-Augmented Generation · Confidence Calibration · Explainable AI
  </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Query Input Row ──────────────────────────
q_col, btn_col = st.columns([5, 1], gap="medium")

with q_col:
    user_query = st.text_area(
        label="Your question",
        placeholder="Ask anything about your documents…",
        height=90,
        label_visibility="collapsed",
        key="user_query",
    )

with btn_col:
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    run_query = st.button("🔍 Analyse", key="run_btn", use_container_width=True)

# ── Session state ────────────────────────────
if "result" not in st.session_state:
    st.session_state.result = None
if "fallback" not in st.session_state:
    st.session_state.fallback = False
if "query_text" not in st.session_state:
    st.session_state.query_text = ""

# ── Execute Query ────────────────────────────
if run_query:
    if not user_query.strip():
        st.warning("⚠️  Please enter a question before running the analysis.")
    else:
        with st.spinner("🔄 Retrieving documents and computing confidence…"):
            try:
                rag = load_rag_system(docs_folder)
                raw = rag.query(user_query.strip())

                # Detect fallback: score < 0.25 base OR label forced Low by system
                base_scores = raw.get("similarity_scores", [])
                base_mean = sum(base_scores) / len(base_scores) if base_scores else 0.0
                fallback_triggered = (
                    base_mean < 0.25
                    or "cannot find reliable" in raw.get("answer", "").lower()
                    or not base_scores
                )

                st.session_state.result = raw
                st.session_state.fallback = fallback_triggered
                st.session_state.query_text = user_query.strip()
            except Exception as e:
                st.error(f"❌ Error running RAG system: {e}")
                st.stop()

# ── Results Dashboard ────────────────────────
result = st.session_state.result

if result:
    answer          = result.get("answer", "—")
    conf_score      = result.get("confidence_score", 0.0)
    conf_label      = result.get("confidence_label", "Low")
    sources         = result.get("sources", [])
    sim_scores      = result.get("similarity_scores", [])
    fallback        = st.session_state.fallback

    # ─ Fallback Banner ───────────────────────
    if fallback:
        st.markdown("""
<div class="fallback-banner">
  ⛔ <span>Fallback triggered — retrieval confidence too low to generate a reliable answer.</span>
</div>
""", unsafe_allow_html=True)

    # ─ Row 1: Answer + Confidence Metrics ────
    ans_col, conf_col = st.columns([3, 2], gap="large")

    with ans_col:
        st.markdown(f"""
<div class="rag-card">
  <div class="card-title">💬 Answer</div>
  <div class="answer-text">{answer}</div>
</div>
""", unsafe_allow_html=True)

    with conf_col:
        # Metric tiles
        mc1, mc2 = st.columns(2, gap="small")
        with mc1:
            st.metric(
                label="Confidence Score",
                value=f"{conf_score:.4f}",
                delta=f"{(conf_score - 0.5):+.4f} vs mid",
                delta_color="normal",
            )
        with mc2:
            st.metric(
                label="Documents Retrieved",
                value=len(sim_scores),
                delta=f"{len(sources)} unique sources",
                delta_color="off",
            )

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # Confidence label card
        st.markdown(f"""
<div class="rag-card" style="padding:16px 20px;">
  <div class="card-title">Confidence Level</div>
  <div style="display:flex; align-items:center; gap:14px; margin-top:4px;">
    {badge_html(conf_label)}
    <span style="color:#8b949e; font-size:0.82rem;">
      {'Reliable answer' if conf_label == 'High' else 'Partial reliability' if conf_label == 'Medium' else 'Low reliability'}
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

        # Animated progress bar
        st.markdown(f"""
<div class="rag-card" style="padding:16px 20px;">
  <div class="card-title">Confidence Progress</div>
""", unsafe_allow_html=True)
        # Use native st.progress for the animated value
        prog_bar = st.progress(0)
        for pct in range(0, int(conf_score * 100) + 1, 3):
            prog_bar.progress(pct / 100)
            time.sleep(0.008)
        prog_bar.progress(conf_score)
        st.markdown(f"""
  <div style="text-align:right; font-size:0.8rem; color:{confidence_color(conf_label)}; margin-top:4px;">
    {conf_score * 100:.1f}% confidence
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")

    # ─ Row 2: Similarity Chart + Sources ─────
    chart_col, src_col = st.columns([3, 2], gap="large")

    with chart_col:
        st.markdown("""
<div class="rag-card">
  <div class="card-title">📊 Document Similarity Scores</div>
""", unsafe_allow_html=True)

        if sim_scores:
            # Build labels from sources list (parallel to scores)
            chart_sources = result.get("sources", [])
            # If unique sources < scores, repeat last source as label
            padded_labels = []
            raw_chunks_sources = result.get("sources", [])
            # sources in result may be deduplicated; use sim_scores length
            for i in range(len(sim_scores)):
                if i < len(raw_chunks_sources):
                    padded_labels.append(raw_chunks_sources[i])
                else:
                    padded_labels.append(f"Chunk {i+1}")

            fig = build_similarity_chart(padded_labels, sim_scores, conf_label)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("No similarity scores available.")

        st.markdown("</div>", unsafe_allow_html=True)

    with src_col:
        st.markdown("""
<div class="rag-card">
  <div class="card-title">📂 Retrieved Sources</div>
""", unsafe_allow_html=True)

        if sources:
            with st.expander(f"📁 View {len(sources)} source document(s)", expanded=True):
                for i, src in enumerate(sources):
                    ext = src.split(".")[-1].upper() if "." in src else "FILE"
                    icon = "📄" if ext == "TXT" else "📕" if ext == "PDF" else "📎"
                    st.markdown(f"""
<div class="source-item">
  <span class="source-icon">{icon}</span>
  <div>
    <div style="color:#c9d1d9; font-weight:600;">{src}</div>
    <div style="font-size:0.76rem; color:#484f58;">{ext} document · rank {i+1}</div>
  </div>
</div>
""", unsafe_allow_html=True)
        else:
            st.info("No sources were retrieved.")

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    # ─ Row 3: Score Breakdown Details ────────
    st.markdown("""
<div class="rag-card">
  <div class="card-title">🔬 Score Breakdown Details</div>
""", unsafe_allow_html=True)

    d1, d2, d3, d4 = st.columns(4, gap="medium")

    base_mean = sum(sim_scores) / len(sim_scores) if sim_scores else 0.0
    variance  = (max(sim_scores) - min(sim_scores)) if len(sim_scores) > 1 else 0.0
    var_pen   = round(variance * 0.2, 4)
    n_chunks  = len(sim_scores)

    with d1:
        st.metric("Base Similarity", f"{base_mean:.4f}", help="Mean cosine similarity across retrieved chunks")
    with d2:
        st.metric("Score Variance", f"{variance:.4f}", help="Spread between max and min similarity scores")
    with d3:
        st.metric("Variance Penalty", f"-{var_pen:.4f}", delta_color="inverse", help="Applied confidence deduction from high variance")
    with d4:
        st.metric("Chunks Analysed", n_chunks, help="Total document chunks retrieved from FAISS index")

    st.markdown("</div>", unsafe_allow_html=True)

else:
    # ── Empty state ──
    st.markdown("""
<div style="text-align:center; padding: 60px 0; color: #484f58;">
  <div style="font-size: 3.5rem; margin-bottom: 16px;">🔍</div>
  <div style="font-size: 1.1rem; color: #8b949e;">
    Enter your query above and click <b style="color:#58a6ff;">Analyse</b> to see results.
  </div>
  <div style="font-size: 0.85rem; margin-top: 10px; color: #484f58;">
    Make sure your documents are in the configured <code>./docs</code> folder.
  </div>
</div>
""", unsafe_allow_html=True)
