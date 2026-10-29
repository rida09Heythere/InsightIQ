# InsightIQ — AI-Powered Business Intelligence & Research Copilot

InsightIQ is an AI-powered application that combines **retrieval-augmented generation (RAG), document intelligence, business analytics, and multi-agent AI workflows** to help users extract information from documents and transform datasets into actionable business insights.

Built with Python, Streamlit, Google Gemini, ChromaDB, and LangGraph, the project brings document question answering and data-driven business analysis into one application.

## Key Features

### 1. AI Document Intelligence
- Upload PDF documents and ask questions in natural language.
- Extract and chunk document content for semantic retrieval.
- Generate context-aware answers using Google Gemini.
- Retrieve relevant passages using Gemini embeddings and ChromaDB.
- Provide source filename and page references to support answers.

### 2. Business Analytics
- Upload CSV and Excel datasets.
- Explore dataset structure, column types, missing values, and duplicate records.
- Generate numerical summaries and key performance indicators (KPIs).
- Analyze monthly trends and identify potential anomalies using IQR-based detection.
- Create interactive Plotly visualizations.
- Clean datasets and download the processed data as CSV.

### 3. Natural-Language Data Analysis
- Ask questions about uploaded datasets in plain English.
- Convert questions into structured analysis plans.
- Execute supported analytical operations using pandas.
- Analyze aggregations, correlations, missing values, rankings, and trends.

### 4. AI-Powered Business Insights
- Generate business performance summaries.
- Identify key findings and potential risks.
- Receive data-backed recommendations.
- Highlight limitations in the available dataset.

### 5. LangGraph Multi-Agent Workflow
- Coordinate a structured data-analysis workflow using LangGraph.
- Calculate analytical evidence using pandas-based functions.
- Generate an AI-written business report using Google Gemini.
- Include report review and error-handling stages.
- Use model fallback to improve resilience when a Gemini model is unavailable or over quota.

**Current development status:** The LangGraph workflow has been tested independently. Integration into the main Streamlit interface remains in progress.

## Technology Stack

| Component | Technologies |
|---|---|
| Programming language | Python |
| Application interface | Streamlit |
| AI model | Google Gemini API |
| Retrieval-augmented generation | RAG |
| Vector database | ChromaDB |
| Embeddings | Google Gemini Embeddings |
| Data analysis | Pandas, NumPy |
| Visualization | Plotly |
| Document processing | PyPDF |
| Multi-agent orchestration | LangGraph |
| Environment configuration | python-dotenv |
| Version control | Git and GitHub |
| Planned automation | n8n |

## System Architecture

```text
                    ┌─────────────────────┐
                    │   Streamlit App     │
                    │      app.py         │
                    └──────────┬──────────┘
                               │
                ┌──────────────┴──────────────┐
                │                             │
       ┌────────▼────────┐           ┌────────▼────────┐
       │ Document Q&A   │           │ Business        │
       │    rag.py      │           │ Analytics       │
       └────────┬────────┘           │ analytics.py    │
                │                    └────────┬────────┘
       ┌────────▼────────┐                    │
       │ Gemini          │           ┌────────▼────────┐
       │ Embeddings      │           │ LangGraph       │
       │ + ChromaDB      │           │ agents.py       │
       └─────────────────┘           └────────┬────────┘
                                              │
                                     ┌────────▼────────┐
                                     │ Gemini AI       │
                                     │ Business Report │
                                     └─────────────────┘

                         Future:
                    n8n Automation
```

## Project Structure

```text
InsightIQ/
├── app.py                 # Main Streamlit application
├── rag.py                 # Document ingestion, retrieval and Q&A
├── rag_backup.py          # Backup of RAG implementation
├── analytics.py           # Data analytics and AI recommendations
├── agents.py              # LangGraph analysis workflow
├── requirements.txt       # Python dependencies
├── .env                   # Local API credentials (not committed)
├── .gitignore             # Excluded files and directories
├── data/
│   ├── sample_sales.csv   # Demonstration sales dataset
│   ├── documents/         # Local document storage
│   └── uploads/           # Local uploaded files
└── chroma_db/             # Persistent vector database
```

## Getting Started

### Prerequisites

- Python 3.11 or compatible version
- pip
- A Google Gemini API key
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/rida09Heythere/InsightIQ.git
cd InsightIQ
```

### 2. Create a Virtual Environment

Linux / ChromeOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

Replace the placeholder with your own API key. Never commit `.env` or publish API credentials.

### 5. Run the Application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal, usually `http://localhost:8501`.

## Sample Dataset

The repository includes `data/sample_sales.csv` for testing the business analytics features. It contains sample sales records with dates, products, categories, regions, sales, profit, and quantities.

Use your own CSV or Excel dataset to explore the analytics capabilities.

## AI Model Usage and Limitations

- InsightIQ uses the Google Gemini API for language generation and embeddings.
- Available models and API quotas depend on your Google AI account.
- Model fallback can help when a configured model is unavailable, but it cannot guarantee uninterrupted access when all eligible models are rate-limited.
- AI-generated recommendations should be reviewed before being used for business decisions.
- Analysis quality depends on the accuracy, completeness, and structure of the input data.

## Roadmap

- [x] Streamlit application foundation
- [x] PDF ingestion and semantic document retrieval
- [x] Document question answering with source references
- [x] CSV and Excel analytics
- [x] KPI calculations and interactive visualizations
- [x] Natural-language data analysis
- [x] AI-generated business recommendations
- [x] LangGraph workflow created and independently tested
- [ ] Integrate LangGraph into the Streamlit interface
- [ ] Expand end-to-end testing and evaluation
- [ ] Integrate n8n for workflow automation
- [ ] Improve error handling, observability, and reliability
- [ ] Redesign the user interface and user experience
- [ ] Deploy the application
- [ ] Add screenshots and a project demonstration

## Security

- Keep API keys and secrets in environment variables.
- Do not commit `.env`, private documents, or personal files.
- Review `.gitignore` before pushing changes.
- Avoid uploading confidential business datasets to public repositories.

## Project Status

**Status: Active development**

InsightIQ is being developed as a practical project exploring AI-assisted document understanding, business analytics, retrieval-augmented generation, and multi-agent orchestration.

## Author

**Rida Rafique**

GitHub: https://github.com/rida09Heythere/InsightIQ
