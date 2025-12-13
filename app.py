import streamlit as st
import requests
from bs4 import BeautifulSoup
from bias_detector_hf import BiasDetectorHF
from gemini_handler import GeminiHandler
import plotly.graph_objects as go

st.set_page_config(
    page_title="News Bias Analysis Platform",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "Advanced AI-powered news bias detection system"}
)

# Professional CSS styling
st.markdown("""
    <style>
    :root {
        --primary: #1e3a8a;
        --secondary: #0f172a;
        --accent: #3b82f6;
        --success: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
        --neutral: #6b7280;
        --light-bg: #f9fafb;
    }
    
    * {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', sans-serif;
    }
    
    .header-container {
        background: linear-gradient(135deg, #0f172a 0%, #1a202c 50%, #0f172a 100%);
        padding: 3.5rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 3rem;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(59, 130, 246, 0.1);
    }
    
    .header-container h1 {
        margin: 0;
        color: #ffffff;
        font-size: 2.4rem;
        font-weight: 700;
        letter-spacing: -0.8px;
    }
    
    .header-container p {
        margin: 0.5rem 0 0 0;
        color: #cbd5e1;
        font-size: 0.98rem;
        font-weight: 400;
        letter-spacing: 0.3px;
    }
    
    .input-container {
        background: #ffffff;
        padding: 2.5rem;
        border-radius: 14px;
        border: 1px solid #d1d5db;
        margin-bottom: 2.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .metric-card:hover {
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
        border-color: #cbd5e1;
        transform: translateY(-2px);
    }
    
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.75rem;
        color: #6b7280;
    }
    
    .metric-value {
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0;
        line-height: 1;
    }
    
    .metric-left { color: #dc2626; }
    .metric-neutral { color: #2563eb; }
    .metric-right { color: #059669; }
    
    .chart-container {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 2.5rem;
        border-radius: 14px;
        border: 1px solid #e2e8f0;
        margin: 2.5rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
    }
    
    .analysis-result {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 2.5rem;
        border-radius: 14px;
        border-left: 5px solid #3b82f6;
        margin: 2.5rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
    }
    
    .result-badge {
        display: inline-block;
        padding: 0.6rem 1.2rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.95rem;
        margin-bottom: 1rem;
    }
    
    .badge-neutral { background: #dbeafe; color: #1e40af; }
    .badge-slightly-left { background: #fee2e2; color: #7f1d1d; }
    .badge-slightly-right { background: #dcfce7; color: #15803d; }
    .badge-left { background: #fecaca; color: #7f1d1d; }
    .badge-right { background: #bbf7d0; color: #15803d; }
    
    .summary-text {
        color: #374151;
        margin: 1rem 0 0 0;
        font-size: 0.975rem;
        line-height: 1.6;
        letter-spacing: 0.3px;
    }
    
    .divider {
        height: 1px;
        background: #e5e7eb;
        margin: 2rem 0;
    }
    
    .info-box {
        background: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 1.25rem;
        border-radius: 8px;
        margin-top: 2rem;
    }
    
    .info-box p {
        margin: 0;
        color: #1e40af;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    .error-box {
        background: #fef2f2;
        border-left: 4px solid #dc2626;
        padding: 1.25rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .error-box p {
        margin: 0;
        color: #7f1d1d;
        font-size: 0.95rem;
    }
    
    .section-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1e3a8a;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e5e7eb;
    }
    
    .meta-info {
        color: #6b7280;
        margin: 1rem 0 0 0;
        font-size: 0.9rem;
    }
    
    .loading-text {
        color: #3b82f6;
        font-weight: 500;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="header-container">
        <h1>News Bias Analysis Platform</h1>
    </div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_detector():
    return BiasDetectorHF()

@st.cache_resource
def load_gemini_handler():
    return GeminiHandler()

detector = load_detector()
gemini_handler = load_gemini_handler()

@st.cache_data
def extract_text_from_url(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        html = requests.get(url, timeout=10, headers=headers).text
        soup = BeautifulSoup(html, "html.parser")
        
        for script in soup(["script", "style"]):
            script.decompose()
        
        paragraphs = soup.find_all("p")
        text = " ".join(p.get_text(strip=True) for p in paragraphs)
        return text if text.strip() and len(text) > 100 else None
    except Exception as e:
        return None

# Input Section
st.markdown('<div class="input-container">', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    input_type = st.radio(
        "Select Input Method",
        ["Paste Text", "News URL"],
        horizontal=True,
        label_visibility="collapsed"
    )

if input_type == "Paste Text":
    text = st.text_area(
        "Article Text",
        height=200,
        placeholder="Paste your news article here...",
        label_visibility="collapsed"
    )
else:
    url = st.text_input(
        "News URL",
        placeholder="https://example.com/article",
        label_visibility="collapsed"
    )
    if url:
        text = extract_text_from_url(url)
        if text is None:
            st.markdown("""
                <div class="error-box">
                    <p><strong>Unable to extract content:</strong> Please verify the URL is valid or paste the text directly.</p>
                </div>
            """, unsafe_allow_html=True)
            text = ""
    else:
        text = ""

st.markdown('</div>', unsafe_allow_html=True)

# Analyze Button
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    analyze_btn = st.button(
        "Analyze Content",
        use_container_width=True,
        type="primary"
    )

if analyze_btn:
    if not text.strip():
        st.markdown("""
            <div class="error-box">
                <p><strong>Input Required:</strong> Please provide article text to analyze.</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        # Create placeholders for progressive loading
        results_placeholder = st.empty()
        metrics_placeholder = st.empty()
        chart_placeholder = st.empty()
        assessment_placeholder = st.empty()
        
        with results_placeholder.container():
            st.markdown('<h2 class="section-title">Analysis in Progress</h2>', unsafe_allow_html=True)
            st.info("Processing your content with AI models...")
        
        # Detect bias
        result = detector.detect_bias(text)
        
        # Get summary from Gemini
        summary = gemini_handler.summarize_news(text)
        
        # Clear loading message
        results_placeholder.empty()
        
        # Display results
        st.markdown('<h2 class="section-title">Analysis Results</h2>', unsafe_allow_html=True)

        # Metrics Row
        metric_col1, metric_col2, metric_col3 = st.columns(3, gap="medium")
        
        with metric_col1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label metric-left">Left Bias</div>
                    <p class="metric-value metric-left">{result['left']:.3f}%</p>
                </div>
            """, unsafe_allow_html=True)
        
        with metric_col2:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label metric-neutral">Neutral</div>
                    <p class="metric-value metric-neutral">{result['neutral']:.3f}%</p>
                </div>
            """, unsafe_allow_html=True)
        
        with metric_col3:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label metric-right">Right Bias</div>
                    <p class="metric-value metric-right">{result['right']:.3f}%</p>
                </div>
            """, unsafe_allow_html=True)

        # Pie Chart
        st.markdown('<h3 class="section-title">Distribution Breakdown</h3>', unsafe_allow_html=True)
        
        fig = go.Figure(data=[go.Pie(
            labels=["Left Bias", "Neutral", "Right Bias"],
            values=[result['left'], result['neutral'], result['right']],
            marker=dict(colors=["#dc2626", "#2563eb", "#059669"]),
            hovertemplate="<b>%{label}</b><br>%{value:.3f}%<extra></extra>",
            textposition="inside",
            textinfo="label+percent",
            textfont=dict(size=13, color="white", family="Arial, sans-serif"),
        )])
        
        fig.update_layout(
            height=450,
            margin=dict(l=20, r=20, t=20, b=20),
            font=dict(size=12, family="Arial, sans-serif"),
            showlegend=True,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.02,
                bgcolor="rgba(0,0,0,0)",
                bordercolor="rgba(0,0,0,0)"
            )
        )
        
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

        # Overall Assessment
        st.markdown('<h3 class="section-title">Assessment</h3>', unsafe_allow_html=True)
        
        badge_class_map = {
            "Neutral": "badge-neutral",
            "Slightly Left": "badge-slightly-left",
            "Slightly Right": "badge-slightly-right",
            "Left": "badge-left",
            "Right": "badge-right",
        }
        
        badge_class = badge_class_map.get(result["overall_bias"], "badge-neutral")
        
        st.markdown(f"""
            <div class="analysis-result">
                <span class="result-badge {badge_class}">{result["overall_bias"]}</span>
                <p class="summary-text">
                    {summary}
                </p>
            </div>
        """, unsafe_allow_html=True)

        # Divider
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        # Info box
        st.markdown("""
            <div class="info-box">
                <p><strong>Disclaimer:</strong> This analysis uses machine learning models trained on text classification tasks. Results should be considered as one input among multiple sources for comprehensive media literacy assessment. No automated system is 100% accurate.</p>
            </div>
        """, unsafe_allow_html=True)
