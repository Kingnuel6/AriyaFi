import streamlit as st
import os
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools import tool
from langchain import hub
from langchain.schema import HumanMessage, SystemMessage
import PyPDF2
import io

st.set_page_config(
    page_title="Financial Report Analyzer",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Syne', sans-serif;
    }
    .stApp {
        background: #0a0a0f;
        color: #e8e6e0;
    }
    .main-header {
        font-family: 'Syne', sans-serif;
        font-size: 3.2rem;
        font-weight: 800;
        letter-spacing: -2px;
        background: linear-gradient(135deg, #f0e6d0 0%, #c9a96e 50%, #f0e6d0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.1;
        margin-bottom: 0.3rem;
    }
    .sub-header {
        font-family: 'DM Mono', monospace;
        font-size: 0.8rem;
        color: #6b6860;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #14141a;
        border: 1px solid #2a2a35;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin: 0.5rem 0;
    }
    .metric-label {
        font-family: 'DM Mono', monospace;
        font-size: 0.7rem;
        color: #6b6860;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }
    .metric-value {
        font-family: 'Syne', sans-serif;
        font-size: 1.6rem;
        font-weight: 700;
        color: #c9a96e;
    }
    .analysis-box {
        background: #14141a;
        border: 1px solid #2a2a35;
        border-left: 3px solid #c9a96e;
        border-radius: 0 12px 12px 0;
        padding: 1.5rem;
        margin: 1rem 0;
        font-family: 'DM Mono', monospace;
        font-size: 0.85rem;
        line-height: 1.8;
        color: #c8c4bc;
    }
    .section-title {
        font-family: 'Syne', sans-serif;
        font-size: 1rem;
        font-weight: 600;
        color: #c9a96e;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin: 1.5rem 0 0.8rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .stButton > button {
        background: #c9a96e !important;
        color: #0a0a0f !important;
        border: none !important;
        font-family: 'Syne', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: 1px !important;
        padding: 0.6rem 2rem !important;
        border-radius: 6px !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        background: #f0c97e !important;
        transform: translateY(-1px) !important;
    }
    .upload-area {
        border: 1px dashed #2a2a35;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        background: #14141a;
    }
    .tag {
        display: inline-block;
        background: #1e1e28;
        border: 1px solid #2a2a35;
        border-radius: 20px;
        padding: 3px 12px;
        font-family: 'DM Mono', monospace;
        font-size: 0.7rem;
        color: #6b6860;
        margin: 2px;
        letter-spacing: 1px;
    }
    .positive { color: #5fb97e !important; }
    .negative { color: #e05252 !important; }
    .neutral { color: #c9a96e !important; }
    div[data-testid="stTextInput"] input,
    div[data-testid="stTextArea"] textarea {
        background: #14141a !important;
        border: 1px solid #2a2a35 !important;
        color: #e8e6e0 !important;
        font-family: 'DM Mono', monospace !important;
        border-radius: 8px !important;
    }
    .stSelectbox > div > div {
        background: #14141a !important;
        border: 1px solid #2a2a35 !important;
        color: #e8e6e0 !important;
    }
    .stFileUploader > div {
        background: #14141a !important;
        border: 1px dashed #2a2a35 !important;
    }
    hr { border-color: #2a2a35 !important; }
    .stSpinner > div { border-top-color: #c9a96e !important; }
</style>
""", unsafe_allow_html=True)


def get_llm():
    api_key = st.session_state.get("openrouter_key") or os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        return None
    return ChatOpenAI(
        model="anthropic/claude-3.5-sonnet",
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.2,
        max_tokens=4000,
    )


def extract_pdf_text(uploaded_file):
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text[:12000]


def analyze_report(text, company_name, analysis_type, llm):
    system_prompt = """You are a senior financial analyst at a top-tier investment bank.
You analyze financial reports with exceptional depth and precision.
Always structure your response clearly with sections.
Use specific numbers from the report whenever possible.
Format numbers with proper notation (e.g., $2.4B, 12.3%, -$450M).
Be direct, specific, and insightful — not generic."""

    prompts = {
        "Full Analysis": f"""Analyze this financial report for {company_name}. Provide:

## Executive Summary
2-3 sentence overview of financial health.

## Key Financial Metrics
Extract and highlight: Revenue, Net Income, EPS, EBITDA, Free Cash Flow, Debt-to-Equity, and any other critical numbers.

## Revenue & Growth Analysis
Detailed breakdown of revenue trends, growth rates, and segment performance.

## Profitability Assessment
Margins analysis (gross, operating, net), profitability trends, cost structure.

## Risk Factors
Top 3-5 material risks identified in the report.

## Investment Outlook
Bull case, bear case, and neutral assessment. Rate overall financial health: Strong / Moderate / Weak.

REPORT TEXT:
{text}""",

        "Risk Assessment": f"""Perform a comprehensive risk assessment for {company_name} based on this financial report.

## Market Risks
Competitive pressures, market share threats, macroeconomic exposure.

## Financial Risks
Liquidity, debt levels, covenant risks, credit exposure.

## Operational Risks
Supply chain, regulatory, execution risks.

## Risk Rating
Overall risk score: Low / Medium / High / Critical — with justification.

REPORT TEXT:
{text}""",

        "Growth Metrics": f"""Extract and analyze all growth indicators for {company_name}.

## Revenue Growth
YoY and QoQ growth rates, trend analysis.

## Market Expansion
New markets, customer acquisition, TAM discussion.

## R&D & Innovation
Investment in future growth, pipeline strength.

## Growth Forecast
Management guidance analysis and your independent assessment.

REPORT TEXT:
{text}""",

        "Competitive Position": f"""Analyze {company_name}'s competitive positioning based on this financial report.

## Market Share & Position
Evidence of competitive standing.

## Moat Analysis
Pricing power, switching costs, network effects visible in financials.

## Strategic Initiatives
Key strategic moves and their financial implications.

## Competitive Strengths & Weaknesses
Based purely on financial evidence.

REPORT TEXT:
{text}"""
    }

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompts[analysis_type])
    ]

    response = llm.invoke(messages)
    return response.content


def quick_metrics(text, company_name, llm):
    prompt = f"""Extract ONLY the key financial metrics from this report for {company_name}.
Return EXACTLY this format (use 'N/A' if not found):

REVENUE: [value]
NET_INCOME: [value]
EPS: [value]
GROSS_MARGIN: [value]
OPERATING_MARGIN: [value]
FCF: [value]
DEBT_TO_EQUITY: [value]
YOY_GROWTH: [value]
SENTIMENT: [Positive/Neutral/Negative]
HEALTH_SCORE: [1-10]

Report:
{text[:4000]}"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


def parse_metrics(metrics_text):
    metrics = {}
    for line in metrics_text.strip().split('\n'):
        if ':' in line:
            key, val = line.split(':', 1)
            metrics[key.strip()] = val.strip()
    return metrics


# ─── SIDEBAR ────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-title">⚙ Configuration</div>', unsafe_allow_html=True)

    api_key_input = st.text_input(
        "OpenRouter API Key",
        type="password",
        placeholder="sk-or-...",
        help="Get your free key at openrouter.ai"
    )
    if api_key_input:
        st.session_state["openrouter_key"] = api_key_input
        st.success("✓ Key saved")

    st.markdown("---")
    st.markdown('<div class="section-title">📋 Analysis Type</div>', unsafe_allow_html=True)
    analysis_type = st.selectbox(
        "",
        ["Full Analysis", "Risk Assessment", "Growth Metrics", "Competitive Position"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown('<div class="section-title">🔗 About</div>', unsafe_allow_html=True)
    st.markdown("""
<div style="font-family: 'DM Mono', monospace; font-size: 0.72rem; color: #6b6860; line-height: 1.8;">
Built with LangChain + OpenRouter<br>
Powered by Claude 3.5 Sonnet<br><br>
Upload any earnings report,<br>annual report, or 10-K/10-Q.<br><br>
<span style="color: #c9a96e;">Try with Apple, Tesla,<br>Google Q4 2024 reports.</span>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
<div style="font-family: 'DM Mono', monospace; font-size: 0.65rem; color: #3a3a45; text-align: center;">
LANGCHAIN · OPENROUTER · STREAMLIT
</div>
""", unsafe_allow_html=True)


# ─── MAIN CONTENT ────────────────────────────────────────────
col_title, col_badge = st.columns([4, 1])
with col_title:
    st.markdown('<div class="main-header">Financial<br>Report Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered · Institutional Grade · Real-Time</div>', unsafe_allow_html=True)

with col_badge:
    st.markdown("""
<div style="text-align:right; padding-top: 1rem;">
<span class="tag">LangChain</span><br>
<span class="tag">GPT-4 / Claude</span><br>
<span class="tag">OpenRouter</span>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Input section
col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown('<div class="section-title">📄 Upload Report</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drop PDF here",
        type=["pdf"],
        label_visibility="collapsed"
    )

    st.markdown('<div class="section-title">✏ Or Paste Text</div>', unsafe_allow_html=True)
    pasted_text = st.text_area(
        "Paste financial report content",
        height=160,
        placeholder="Paste earnings report, 10-K excerpts, or any financial disclosure text here...",
        label_visibility="collapsed"
    )

with col2:
    st.markdown('<div class="section-title">🏢 Company Details</div>', unsafe_allow_html=True)
    company_name = st.text_input(
        "Company Name",
        placeholder="e.g. Apple Inc., Tesla, Google",
        label_visibility="collapsed"
    )

    st.markdown('<div class="section-title">💡 Example Reports to Try</div>', unsafe_allow_html=True)
    st.markdown("""
<div style="font-family: 'DM Mono', monospace; font-size: 0.75rem; color: #6b6860; line-height: 2; background: #14141a; padding: 1rem; border-radius: 8px; border: 1px solid #2a2a35;">
→ Apple Q4 2024 Earnings<br>
→ Tesla Annual Report 2024<br>
→ NVIDIA Q3 2024 10-Q<br>
→ JPMorgan 2024 Annual Report<br>
→ Any public company 10-K/10-Q
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    analyze_btn = st.button("⚡ Analyze Report", use_container_width=True)


# ─── ANALYSIS EXECUTION ──────────────────────────────────────
if analyze_btn:
    llm = get_llm()

    if not llm:
        st.error("⚠️ Please enter your OpenRouter API key in the sidebar first.")
        st.stop()

    report_text = ""
    if uploaded_file:
        with st.spinner("Extracting PDF content..."):
            report_text = extract_pdf_text(uploaded_file)
    elif pasted_text.strip():
        report_text = pasted_text.strip()

    if not report_text:
        st.warning("Please upload a PDF or paste report text.")
        st.stop()

    if not company_name.strip():
        company_name = "the company"

    st.markdown("---")

    col_m1, col_m2 = st.columns([1, 2])

    with col_m1:
        st.markdown('<div class="section-title">📊 Key Metrics</div>', unsafe_allow_html=True)
        with st.spinner("Extracting metrics..."):
            try:
                metrics_raw = quick_metrics(report_text, company_name, llm)
                metrics = parse_metrics(metrics_raw)

                def render_metric(label, key, color_class="neutral"):
                    val = metrics.get(key, "N/A")
                    st.markdown(f"""
<div class="metric-card">
    <div class="metric-label">{label}</div>
    <div class="metric-value {color_class}">{val}</div>
</div>""", unsafe_allow_html=True)

                render_metric("Revenue", "REVENUE")
                render_metric("Net Income", "NET_INCOME")
                render_metric("EPS", "EPS")
                render_metric("YoY Growth", "YOY_GROWTH")
                render_metric("Gross Margin", "GROSS_MARGIN")
                render_metric("Free Cash Flow", "FCF")

                health = metrics.get("HEALTH_SCORE", "N/A")
                sentiment = metrics.get("SENTIMENT", "N/A")
                sentiment_class = "positive" if sentiment == "Positive" else ("negative" if sentiment == "Negative" else "neutral")

                st.markdown(f"""
<div class="metric-card" style="border-left: 3px solid #c9a96e;">
    <div class="metric-label">Overall Sentiment</div>
    <div class="metric-value {sentiment_class}">{sentiment}</div>
    <div style="font-family: 'DM Mono', monospace; font-size: 0.72rem; color: #6b6860; margin-top: 4px;">
        Health Score: {health}/10
    </div>
</div>""", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Metrics extraction error: {str(e)}")

    with col_m2:
        st.markdown(f'<div class="section-title">🔍 {analysis_type}</div>', unsafe_allow_html=True)
        with st.spinner(f"Running {analysis_type}..."):
            try:
                analysis = analyze_report(report_text, company_name, analysis_type, llm)
                st.markdown(f'<div class="analysis-box">{analysis}</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Analysis error: {str(e)}")

    st.markdown("---")
    st.markdown(f"""
<div style="font-family: 'DM Mono', monospace; font-size: 0.7rem; color: #3a3a45; text-align: center; padding: 1rem 0;">
Analysis powered by LangChain + OpenRouter · {company_name} · {analysis_type} · Report length: {len(report_text):,} chars
</div>
""", unsafe_allow_html=True)

else:
    st.markdown("""
<div style="text-align: center; padding: 3rem 0; color: #3a3a45; font-family: 'DM Mono', monospace; font-size: 0.8rem; letter-spacing: 2px;">
UPLOAD A REPORT · ENTER API KEY · CLICK ANALYZE
</div>
""", unsafe_allow_html=True)
