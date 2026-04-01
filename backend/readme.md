# CSV Analyzer with PandasAI - Backend

## Quick Start

### 1. Create Virtual Environment
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set OpenAI API Key (for PandasAI chat feature)
```bash
export OPENAI_API_KEY='your-openai-api-key'
# On Windows: set OPENAI_API_KEY=your-openai-api-key
```

### 4. Run the Server
```bash
python app.py
```

Server will start at `http://localhost:5000`

## API Endpoints

### Health Check
```
GET /api/health
```

### Upload CSV
```
POST /api/upload
Content-Type: multipart/form-data
Body: file (CSV file)
```

### Create Visualization
```
POST /api/visualize
Content-Type: application/json
Body: {
    "chart_type": "histogram|bar|scatter|line|box|heatmap|pairplot|pie",
    "x_column": "column_name",
    "y_column": "column_name" (optional, for scatter/line)
}
```

### Chat with Data (PandasAI)
```
POST /api/chat
Content-Type: application/json
Body: {
    "query": "What is the average value of column X?"
}
```

### Execute Custom Code
```
POST /api/custom-code
Content-Type: application/json
Body: {
    "code": "result = df.describe()"
}
```

### Clear Session
```
POST /api/clear
```

## Example Chat Queries

- "What is the average of [column]?"
- "Show me the top 5 rows"
- "Create a bar chart of [column]"
- "What is the correlation between [col1] and [col2]?"
- "How many missing values are there?"
- "Group by [column] and show the sum"
- "Plot a histogram of [column]"

## Supported Chart Types

| Type | Description | Required Columns |
|------|-------------|-----------------|
| histogram | Frequency distribution | x_column |
| bar | Categorical count | x_column |
| scatter | X vs Y scatter plot | x_column, y_column |
| line | Line chart | x_column, y_column |
| box | Box plot | x_column |
| heatmap | Correlation matrix | None (uses all numeric) |
| pairplot | Pairwise relationships | None (uses all numeric) |
| pie | Pie chart | x_column |

## Using Different LLMs

PandasAI supports multiple LLM providers. Modify `app.py`:

### OpenAI (Default)
```python
from pandasai.llm import OpenAI
llm = OpenAI(api_token="your-key")
```

### Anthropic Claude
```python
from pandasai.llm import Anthropic
llm = Anthropic(api_token="your-key")
```

### Google PaLM
```python
from pandasai.llm import GooglePalm
llm = GooglePalm(api_token="your-key")
```

### Local LLM (Ollama)
```python
from pandasai.llm import Ollama
llm = Ollama(model="llama2")
```

## Troubleshooting

### PandasAI not working
- Ensure OPENAI_API_KEY is set
- Check API key has credits
- Try simpler queries first

### CORS errors
- Backend uses flask-cors with permissive settings
- Check frontend is pointing to correct URL

### Memory issues with large files
- Limit file size in upload
- Sample data before process
backend/readme.md