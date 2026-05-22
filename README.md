# Talk to Your Data

### Agentic data analyst built with LangChain that turns natural-language questions into CSV insights, visualizations, and summaries — enabling teams to make decisions without SQL or spreadsheets.

---

# Overview

Talk to Your Data is an AI-powered data analysis application that allows users to upload CSV datasets and interact with them using natural language.

Built using Streamlit, LangChain, and LLMs (OpenAI / Claude), the system acts as an agentic data analyst capable of:
- understanding analytical intent,
- reasoning over tabular datasets,
- generating Python-based visualizations,
- executing analytical workflows,
- and returning insights in real time.

Instead of manually writing SQL queries, spreadsheet formulas, or Pandas code, users can simply ask questions such as:

- "What are the pricing trends for Vistara flights?"
- "Plot the distribution of ticket prices."
- "Show the top 10 most expensive destinations."
- "Visualize the relationship between flight duration and price."

The application dynamically generates and executes analytical logic using LLM-powered reasoning and Python code execution.

---

# Business Use Case

Modern business teams frequently rely on CSV exports from:
- CRMs
- ERP systems
- airline booking systems
- finance dashboards
- customer support tools
- marketing platforms
- operations databases

However, extracting insights from these datasets often requires:
- SQL knowledge,
- spreadsheet expertise,
- BI tools,
- or dedicated analysts.

Talk to Your Data reduces this friction by enabling non-technical users to:
- explore datasets conversationally,
- generate instant visualizations,
- identify trends and anomalies,
- and make faster data-driven decisions.

---

# Key Features

## Natural Language Data Analysis
Ask questions in plain English and receive analytical insights instantly.

### Example
"What is the average ticket price for Delhi to Mumbai flights?"

---

## AI-Powered Visualization Generation
The system dynamically generates Python-based visualizations using:
- Matplotlib
- Pandas

### Supported Visualizations
- Histograms
- Scatter plots
- Bar charts
- Distribution analysis
- Trend analysis

---

## Agentic Workflow Execution
The application uses LangChain agents capable of:
- understanding user intent,
- selecting analytical operations,
- generating executable Python code,
- and reasoning over structured datasets.

---

## CSV Upload & Exploration
Users can:
- upload CSV datasets,
- preview datasets,
- inspect rows and columns,
- and explore data interactively.

---

## Query History
Maintains session-level query history for:
- conversational continuity,
- result tracking,
- and iterative analysis.

---

# Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| LLM Orchestration | LangChain |
| Models | OpenAI GPT-4o / Claude 3.5 Sonnet |
| Data Processing | Pandas |
| Visualization | Matplotlib |
| Environment Management | Python Dotenv |
| Agent Framework | LangChain CSV Agent |

---

# System Architecture

```text
User Query
    ↓
Streamlit Interface
    ↓
LangChain CSV Agent
    ↓
LLM Reasoning (OpenAI / Claude)
    ↓
Python Code Generation
    ↓
Pandas + Matplotlib Execution
    ↓
Visualizations + Insights
    ↓
Rendered in Streamlit
