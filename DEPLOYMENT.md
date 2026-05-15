# Deployment Instructions

## Prerequisites

- Python 3.11+
- An Anthropic API key (get one at console.anthropic.com)
- Optional: Tavily API key for higher-quality web search

---

## 1. Install the package

```bash
# From the repo root
pip install -e ".[dev,web]"
```

| Extra | What it adds |
|-------|-------------|
| *(none)* | Core agent + DuckDuckGo search |
| `web` | FastAPI + Uvicorn for the web frontend |
| `dev` | LangChain Anthropic/OpenAI adapters + pytest |
| `tavily` | Tavily search provider |

---

## 2. Jupyter Notebook (`research.ipynb`)

### One-time setup

```bash
pip install jupyter
```

### Run

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
# Optional:
# export TAVILY_API_KEY="tvly-..."

jupyter notebook research.ipynb
```

The notebook opens in your browser. Work through the cells top-to-bottom:

1. **Setup** – asserts your API key is present and imports the agent.
2. **Configure the agent** – tweak `Config` flags (`enable_clarification`, `enable_gap_review`, `enable_draft_review`) and model names.
3. **Run a query** – change `QUERY` and run the cell. If any interactive flags are enabled, the cell will call `input()` and wait for your response.
4. **Report** – renders the final Markdown report inline.
5. **Sources** – lists every source with relevance score and URL.
6. **Inspect state** – shows loop count, findings, and source totals.

### Interactive flags

| Flag | Effect when `True` |
|------|--------------------|
| `enable_clarification` | Pauses before research starts; prompts you to answer clarifying questions |
| `enable_gap_review` | Pauses after research; lets you approve or redirect gap-filling queries |
| `enable_draft_review` | Pauses before finalising; lets you approve the draft or give feedback |

---

## 3. Web Frontend (`web/app.py`)

### Run locally

```bash
uvicorn web.app:app --reload --port 8000
```

Open `http://localhost:8000` in your browser.

### Environment variables (optional)

The web UI accepts the API key in the request form, so no server-side env var is required. If you want to pre-fill defaults via environment, edit `web/static/index.html`.

### Configuration options exposed in the UI

| Field | Default | Description |
|-------|---------|-------------|
| API key | *(required)* | Anthropic (or OpenAI) key |
| Provider | `anthropic` | `anthropic` or `openai` |
| Fast model | `claude-haiku-4-5-20251001` | Used for cheaper/faster steps |
| Powerful model | `claude-sonnet-4-6` | Used for planning and writing |
| Max loops | `2` | Research iterations |
| Breadth | `3` | Search queries per loop |
| Enable clarification | off | Ask user questions before researching |
| Enable gap review | off | Let user approve gap-filling queries |
| Enable draft review | off | Let user approve draft before finalising |
| Search provider | `duckduckgo` | `duckduckgo` or `tavily` |
| Tavily API key | *(optional)* | Required only when using Tavily |

### Production deployment

```bash
# Install production deps only (no dev extras)
pip install -e ".[web]"

# Run with multiple workers
uvicorn web.app:app --host 0.0.0.0 --port 8000 --workers 4
```

> **Note:** The in-memory job store (`_jobs` dict in `web/app.py`) is not shared across workers. For multi-worker production use, either keep `--workers 1` or replace the job store with a shared backend (Redis, database, etc.).

### Docker (optional)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -e ".[web]"
EXPOSE 8000
CMD ["uvicorn", "web.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t deep-research .
docker run -p 8000:8000 deep-research
```

---

## 4. API key security

- **Notebook**: set `ANTHROPIC_API_KEY` as an environment variable before launching Jupyter; never hard-code keys in notebook cells.
- **Web UI**: keys are sent in the POST body over HTTPS. For public deployments, put the app behind TLS (e.g. nginx + Let's Encrypt, or a managed platform like Railway/Fly.io).
- Never commit `.env` files or notebooks with keys saved in cell output.
