# 📊 Financial Report Analyzer

> AI-powered financial report analysis built with LangChain + OpenRouter + Streamlit

**Live Demo:** [your-app.streamlit.app](https://your-app.streamlit.app)

---

## What It Does

Upload any earnings report, 10-K, 10-Q, or annual report PDF and get:

- **Key Metrics Extraction** — Revenue, Net Income, EPS, Margins, FCF automatically pulled
- **Full Analysis** — Executive summary, growth trends, profitability deep-dive
- **Risk Assessment** — Market, financial, and operational risk identification
- **Growth Metrics** — YoY trends, expansion signals, management guidance analysis
- **Competitive Position** — Moat analysis, strategic positioning from financial evidence

## Tech Stack

| Layer | Technology |
|---|---|
| AI Framework | LangChain |
| LLM Provider | OpenRouter (Claude 3.5 Sonnet) |
| Frontend | Streamlit |
| PDF Processing | PyPDF2 |
| Monitoring | LangSmith |

## Quick Start

```bash
git clone https://github.com/yourusername/financial-report-analyzer
cd financial-report-analyzer
pip install -r requirements.txt
```

Create a `.env` file:
```
OPENROUTER_API_KEY=sk-or-your-key-here
```

Run locally:
```bash
streamlit run app.py
```

## Deploy to Streamlit Cloud (Free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Add `OPENROUTER_API_KEY` in Secrets
5. Deploy → get your live link

## Get Your Free API Key

1. Go to [openrouter.ai](https://openrouter.ai)
2. Sign up (free)
3. Go to API Keys → Create key
4. Free credits included to get started

---

Built by [Your Name] · LangChain Portfolio Project
