"""
CSV Analyzer with PandasAI - Flask Backend
Run: python app.py
"""

import os
import io
import base64
import pandas as pd
import matplotlib

matplotlib.use('Agg')  # Non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, request, jsonify, session
from flask_cors import CORS
from werkzeug.utils import secure_filename
import uuid
import json

# PandasAI imports
try:
    from pandasai import SmartDataframe
    from pandasai.llm import OpenAI

    PANDASAI_AVAILABLE = True
except ImportError:
    PANDASAI_AVAILABLE = False
    print("Warning: PandasAI not installed. Chat feature will be disabled.")

app = Flask(__name__)
app.secret_key = os.urandom(24)
CORS(app, supports_credentials=True)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Store dataframes in memory (for production, use Redis or database)
dataframes = {}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_session_id():
    """Get or create session ID"""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return session['session_id']


def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{img_base64}"


# ============ ROUTES ============

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'pandasai_available': PANDASAI_AVAILABLE
    })


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Upload CSV file and perform initial EDA"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Only CSV files are allowed'}), 400

    try:
        # Read CSV
        df = pd.read_csv(file)

        # Store in memory with session ID
        session_id = get_session_id()
        dataframes[session_id] = df

        # Save file to disk (optional)
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, f"{session_id}_{filename}")
        df.to_csv(filepath, index=False)

        # Perform EDA
        eda_results = perform_eda(df)

        return jsonify({
            'success': True,
            'filename': filename,
            'session_id': session_id,
            'eda': eda_results
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


def perform_eda(df):
    """Perform Exploratory Data Analysis"""

    # Basic info
    basic_info = {
        'rows': len(df),
        'columns': len(df.columns),
        'duplicates': int(df.duplicated().sum()),
        'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB",
        'total_missing': int(df.isnull().sum().sum())
    }

    # Column info
    columns_info = []
    for col in df.columns:
        col_info = {
            'name': col,
            'dtype': str(df[col].dtype),
            'non_null': int(df[col].notna().sum()),
            'null_count': int(df[col].isnull().sum()),
            'null_percentage': round(df[col].isnull().sum() / len(df) * 100, 2),
            'unique': int(df[col].nunique())
        }
        columns_info.append(col_info)

    # Data preview (first 10 rows)
    preview = df.head(10).to_dict(orient='records')

    # Numeric columns statistics
    numeric_stats = {}
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    if numeric_cols:
        stats_df = df[numeric_cols].describe()
        numeric_stats = stats_df.to_dict()

    # Categorical columns info
    categorical_info = {}
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    for col in categorical_cols[:10]:  # Limit to 10 categorical columns
        value_counts = df[col].value_counts().head(10).to_dict()
        categorical_info[col] = {
            'unique_values': int(df[col].nunique()),
            'top_values': {str(k): int(v) for k, v in value_counts.items()}
        }

    return {
        'basic_info': basic_info,
        'columns_info': columns_info,
        'preview': preview,
        'column_names': df.columns.tolist(),
        'numeric_columns': numeric_cols,
        'categorical_columns': categorical_cols,
        'numeric_stats': numeric_stats,
        'categorical_info': categorical_info
    }


@app.route('/api/visualize', methods=['POST'])
def create_visualization():
    """Generate visualization based on user request"""
    session_id = get_session_id()

    if session_id not in dataframes:
        return jsonify({'error': 'No data uploaded. Please upload a CSV first.'}), 400

    df = dataframes[session_id]
    data = request.json

    chart_type = data.get('chart_type')
    x_column = data.get('x_column')
    y_column = data.get('y_column')

    try:
        fig, ax = plt.subplots(figsize=(10, 6))

        if chart_type == 'histogram':
            if x_column not in df.columns:
                return jsonify({'error': f'Column {x_column} not found'}), 400
            df[x_column].hist(ax=ax, bins=30, edgecolor='black', color='#3b82f6')
            ax.set_xlabel(x_column)
            ax.set_ylabel('Frequency')
            ax.set_title(f'Histogram of {x_column}')

        elif chart_type == 'bar':
            if x_column not in df.columns:
                return jsonify({'error': f'Column {x_column} not found'}), 400
            value_counts = df[x_column].value_counts().head(20)
            value_counts.plot(kind='bar', ax=ax, color='#3b82f6', edgecolor='black')
            ax.set_xlabel(x_column)
            ax.set_ylabel('Count')
            ax.set_title(f'Bar Chart of {x_column}')
            plt.xticks(rotation=45, ha='right')

        elif chart_type == 'scatter':
            if x_column not in df.columns or y_column not in df.columns:
                return jsonify({'error': 'Invalid columns specified'}), 400
            ax.scatter(df[x_column], df[y_column], alpha=0.6, color='#3b82f6')
            ax.set_xlabel(x_column)
            ax.set_ylabel(y_column)
            ax.set_title(f'Scatter Plot: {x_column} vs {y_column}')

        elif chart_type == 'line':
            if x_column not in df.columns or y_column not in df.columns:
                return jsonify({'error': 'Invalid columns specified'}), 400
            df_sorted = df.sort_values(by=x_column)
            ax.plot(df_sorted[x_column], df_sorted[y_column], color='#3b82f6', linewidth=2)
            ax.set_xlabel(x_column)
            ax.set_ylabel(y_column)
            ax.set_title(f'Line Chart: {x_column} vs {y_column}')

        elif chart_type == 'box':
            if x_column not in df.columns:
                return jsonify({'error': f'Column {x_column} not found'}), 400
            df.boxplot(column=x_column, ax=ax)
            ax.set_title(f'Box Plot of {x_column}')

        elif chart_type == 'heatmap':
            plt.close(fig)
            numeric_df = df.select_dtypes(include=['int64', 'float64'])
            if numeric_df.empty:
                return jsonify({'error': 'No numeric columns for correlation'}), 400

            fig, ax = plt.subplots(figsize=(12, 10))
            corr = numeric_df.corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax, fmt='.2f')
            ax.set_title('Correlation Heatmap')

        elif chart_type == 'pairplot':
            plt.close(fig)
            numeric_df = df.select_dtypes(include=['int64', 'float64'])
            if numeric_df.empty:
                return jsonify({'error': 'No numeric columns for pair plot'}), 400

            # Limit to first 5 numeric columns for performance
            cols = numeric_df.columns[:5].tolist()
            fig = sns.pairplot(df[cols], diag_kind='hist', plot_kws={'alpha': 0.6})
            img_base64 = fig_to_base64(fig.fig)
            return jsonify({'success': True, 'image': img_base64})

        elif chart_type == 'pie':
            if x_column not in df.columns:
                return jsonify({'error': f'Column {x_column} not found'}), 400
            value_counts = df[x_column].value_counts().head(10)
            ax.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%')
            ax.set_title(f'Pie Chart of {x_column}')

        else:
            return jsonify({'error': f'Unknown chart type: {chart_type}'}), 400

        plt.tight_layout()
        img_base64 = fig_to_base64(fig)

        return jsonify({'success': True, 'image': img_base64})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/chat', methods=['POST'])
def chat_with_data():
    """Chat with data using PandasAI"""
    if not PANDASAI_AVAILABLE:
        return jsonify({
            'error': 'PandasAI is not installed. Install it with: pip install pandasai'
        }), 400

    session_id = get_session_id()

    if session_id not in dataframes:
        return jsonify({'error': 'No data uploaded. Please upload a CSV first.'}), 400

    df = dataframes[session_id]
    data = request.json
    query = data.get('query', '')

    if not query:
        return jsonify({'error': 'No query provided'}), 400

    try:
        # Initialize OpenAI LLM
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            return jsonify({
                'error': 'OPENAI_API_KEY environment variable not set'
            }), 400

        llm = OpenAI(api_token=api_key)

        # Create SmartDataframe
        smart_df = SmartDataframe(df, config={
            "llm": llm,
            "enable_cache": False,
            "save_charts": False,
            "verbose": False
        })

        # Execute query
        response = smart_df.chat(query)

        # Check if response is a figure
        if isinstance(response, plt.Figure):
            img_base64 = fig_to_base64(response)
            return jsonify({
                'success': True,
                'type': 'image',
                'response': img_base64
            })

        # Check if response is a DataFrame
        if isinstance(response, pd.DataFrame):
            return jsonify({
                'success': True,
                'type': 'table',
                'response': response.to_dict(orient='records'),
                'columns': response.columns.tolist()
            })

        # Otherwise return as text
        return jsonify({
            'success': True,
            'type': 'text',
            'response': str(response)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/custom-code', methods=['POST'])
def execute_custom_code():
    """Execute custom pandas code (for advanced users)"""
    session_id = get_session_id()

    if session_id not in dataframes:
        return jsonify({'error': 'No data uploaded'}), 400

    df = dataframes[session_id]
    data = request.json
    code = data.get('code', '')

    if not code:
        return jsonify({'error': 'No code provided'}), 400

    try:
        # Create a safe namespace for code execution
        namespace = {
            'df': df.copy(),
            'pd': pd,
            'np': __import__('numpy'),
            'plt': plt,
            'sns': sns
        }

        # Execute the code
        exec(code, namespace)

        # Check if there's a figure to return
        if plt.get_fignums():
            fig = plt.gcf()
            img_base64 = fig_to_base64(fig)
            return jsonify({
                'success': True,
                'type': 'image',
                'response': img_base64
            })

        # Check if 'result' variable was set
        if 'result' in namespace:
            result = namespace['result']
            if isinstance(result, pd.DataFrame):
                return jsonify({
                    'success': True,
                    'type': 'table',
                    'response': result.head(100).to_dict(orient='records')
                })
            return jsonify({
                'success': True,
                'type': 'text',
                'response': str(result)
            })

        return jsonify({
            'success': True,
            'type': 'text',
            'response': 'Code executed successfully'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/clear', methods=['POST'])
def clear_session():
    """Clear session data"""
    session_id = get_session_id()

    if session_id in dataframes:
        del dataframes[session_id]

    session.clear()

    return jsonify({'success': True, 'message': 'Session cleared'})


# ============ MAIN ============

if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("CSV Analyzer with PandasAI")
    print("=" * 50)
    print(f"PandasAI Available: {PANDASAI_AVAILABLE}")
    print("\nTo enable chat feature, set OPENAI_API_KEY:")
    print("  export OPENAI_API_KEY='your-api-key'")
    print("\nStarting server at http://localhost:5000")
    print("=" * 50 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000)

