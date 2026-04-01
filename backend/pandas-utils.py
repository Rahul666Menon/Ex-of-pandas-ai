"""
PandasAI Utilities and Advanced Features
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict, Any

# PandasAI imports with error handling
try:
    from pandasai import SmartDataframe, SmartDatalake
    from pandasai.llm import OpenAI, Anthropic
    from pandasai.callbacks import BaseCallback
    from pandasai.responses.response_parser import ResponseParser

    PANDASAI_AVAILABLE = True
except ImportError:
    PANDASAI_AVAILABLE = False


class QueryCallback(BaseCallback):
    """Custom callback to track query execution"""

    def __init__(self):
        self.queries = []
        self.code_executed = []

    def on_prompt(self, prompt: str):
        """Called when a prompt is generated"""
        self.queries.append(prompt)

    def on_code(self, code: str):
        """Called when code is generated"""
        self.code_executed.append(code)


class CustomResponseParser(ResponseParser):
    """Custom response parser for better formatting"""

    def parse(self, response: Any) -> Dict[str, Any]:
        """Parse and format the response"""
        if isinstance(response, pd.DataFrame):
            return {
                'type': 'dataframe',
                'data': response.to_dict(orient='records'),
                'columns': response.columns.tolist(),
                'shape': response.shape
            }
        elif isinstance(response, plt.Figure):
            return {
                'type': 'figure',
                'data': response
            }
        elif isinstance(response, (list, dict)):
            return {
                'type': 'json',
                'data': response
            }
        else:
            return {
                'type': 'text',
                'data': str(response)
            }


class CSVAnalyzer:
    """
    Main class for CSV analysis with PandasAI
    """

    def __init__(self, df: pd.DataFrame, api_key: Optional[str] = None):
        """
        Initialize the analyzer

        Args:
            df: Pandas DataFrame
            api_key: OpenAI API key (uses env var if not provided)
        """
        self.df = df
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        self.smart_df = None
        self.callback = QueryCallback()

        if PANDASAI_AVAILABLE and self.api_key:
            self._init_smart_df()

    def _init_smart_df(self):
        """Initialize SmartDataframe"""
        llm = OpenAI(api_token=self.api_key)

        self.smart_df = SmartDataframe(
            self.df,
            config={
                "llm": llm,
                "enable_cache": True,
                "verbose": False,
                "save_charts": False,
                "save_charts_path": "charts/",
                "custom_whitelisted_dependencies": ["seaborn"],
                "callback": self.callback
            }
        )

    def basic_info(self) -> Dict[str, Any]:
        """Get basic DataFrame information"""
        return {
            'shape': self.df.shape,
            'rows': len(self.df),
            'columns': len(self.df.columns),
            'column_names': self.df.columns.tolist(),
            'dtypes': self.df.dtypes.astype(str).to_dict(),
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / (1024 * 1024),
            'duplicates': self.df.duplicated().sum(),
            'missing_values': self.df.isnull().sum().to_dict()
        }

    def numeric_summary(self) -> pd.DataFrame:
        """Get summary statistics for numeric columns"""
        numeric_cols = self.df.select_dtypes(include=['int64', 'float64'])
        if numeric_cols.empty:
            return pd.DataFrame()
        return numeric_cols.describe()

    def categorical_summary(self) -> Dict[str, Dict[str, int]]:
        """Get summary for categorical columns"""
        cat_cols = self.df.select_dtypes(include=['object', 'category'])
        summary = {}
        for col in cat_cols.columns:
            summary[col] = {
                'unique': self.df[col].nunique(),
                'top_values': self.df[col].value_counts().head(5).to_dict()
            }
        return summary

    def correlation_matrix(self) -> pd.DataFrame:
        """Get correlation matrix for numeric columns"""
        numeric_cols = self.df.select_dtypes(include=['int64', 'float64'])
        if numeric_cols.empty:
            return pd.DataFrame()
        return numeric_cols.corr()

    def detect_outliers(self, column: str, method: str = 'iqr') -> pd.Series:
        """
        Detect outliers in a numeric column

        Args:
            column: Column name
            method: 'iqr' or 'zscore'

        Returns:
            Boolean series indicating outliers
        """
        if column not in self.df.columns:
            raise ValueError(f"Column {column} not found")

        col_data = self.df[column]

        if method == 'iqr':
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return (col_data < lower_bound) | (col_data > upper_bound)

        elif method == 'zscore':
            mean = col_data.mean()
            std = col_data.std()
            z_scores = (col_data - mean) / std
            return abs(z_scores) > 3

        else:
            raise ValueError(f"Unknown method: {method}")

    def chat(self, query: str) -> Any:
        """
        Chat with the data using natural language

        Args:
            query: Natural language query

        Returns:
            Response from PandasAI
        """
        if not PANDASAI_AVAILABLE:
            raise ImportError("PandasAI is not installed")

        if not self.smart_df:
            raise ValueError("SmartDataframe not initialized. Check API key.")

        return self.smart_df.chat(query)

    def get_executed_code(self) -> List[str]:
        """Get list of code that was executed"""
        return self.callback.code_executed

    def create_visualization(
            self,
            chart_type: str,
            x: Optional[str] = None,
            y: Optional[str] = None,
            hue: Optional[str] = None,
            title: Optional[str] = None
    ) -> plt.Figure:
        """
        Create a visualization

        Args:
            chart_type: Type of chart
            x: X-axis column
            y: Y-axis column
            hue: Color grouping column
            title: Chart title

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        if chart_type == 'histogram':
            self.df[x].hist(ax=ax, bins=30, edgecolor='black')
            ax.set_xlabel(x)
            ax.set_ylabel('Frequency')

        elif chart_type == 'scatter':
            sns.scatterplot(data=self.df, x=x, y=y, hue=hue, ax=ax)

        elif chart_type == 'line':
            self.df.plot(x=x, y=y, ax=ax, kind='line')

        elif chart_type == 'bar':
            if y:
                self.df.groupby(x)[y].mean().plot(kind='bar', ax=ax)
            else:
                self.df[x].value_counts().plot(kind='bar', ax=ax)

        elif chart_type == 'box':
            if hue:
                sns.boxplot(data=self.df, x=hue, y=x, ax=ax)
            else:
                self.df.boxplot(column=x, ax=ax)

        elif chart_type == 'heatmap':
            plt.close(fig)
            fig, ax = plt.subplots(figsize=(12, 10))
            corr = self.correlation_matrix()
            sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax)

        elif chart_type == 'violin':
            sns.violinplot(data=self.df, x=hue, y=x, ax=ax)

        elif chart_type == 'kde':
            sns.kdeplot(data=self.df, x=x, hue=hue, ax=ax)

        else:
            raise ValueError(f"Unknown chart type: {chart_type}")

        if title:
            ax.set_title(title)

        plt.tight_layout()
        return fig

    def auto_eda(self) -> Dict[str, Any]:
        """
        Perform automatic EDA and return comprehensive report
        """
        report = {
            'basic_info': self.basic_info(),
            'numeric_summary': self.numeric_summary().to_dict() if not self.numeric_summary().empty else {},
            'categorical_summary': self.categorical_summary(),
            'correlation_matrix': self.correlation_matrix().to_dict() if not self.correlation_matrix().empty else {},
            'missing_values_percentage': (self.df.isnull().sum() / len(self.df) * 100).to_dict(),
        }

        # Detect potential issues
        issues = []

        # High missing values
        for col, pct in report['missing_values_percentage'].items():
            if pct > 20:
                issues.append(f"High missing values in '{col}': {pct:.1f}%")

        # High cardinality categorical columns
        for col, info in report['categorical_summary'].items():
            unique_ratio = info['unique'] / len(self.df)
            if unique_ratio > 0.5 and info['unique'] > 100:
                issues.append(f"High cardinality in '{col}': {info['unique']} unique values")

        # Duplicates
        if report['basic_info']['duplicates'] > 0:
            dup_pct = report['basic_info']['duplicates'] / len(self.df) * 100
            issues.append(f"Found {report['basic_info']['duplicates']} duplicates ({dup_pct:.1f}%)")

        report['potential_issues'] = issues

        return report


# ============ Example Usage ============

if __name__ == '__main__':
    # Create sample data
    import numpy as np

    np.random.seed(42)
    n = 1000

    df = pd.DataFrame({
        'age': np.random.randint(18, 80, n),
        'salary': np.random.normal(60000, 20000, n),
        'department': np.random.choice(['Engineering', 'Marketing', 'Sales', 'HR'], n),
        'experience': np.random.randint(0, 30, n),
        'satisfaction': np.random.uniform(1, 10, n)
    })

    # Add some missing values
    df.loc[np.random.choice(n, 50), 'salary'] = np.nan

    # Initialize analyzer
    analyzer = CSVAnalyzer(df)

    # Get basic info
    print("Basic Info:")
    print(analyzer.basic_info())

    # Get numeric summary
    print("\nNumeric Summary:")
    print(analyzer.numeric_summary())

    # Get auto EDA report
    print("\nAuto EDA Report:")
    report = analyzer.auto_eda()
    print(f"Potential Issues: {report['potential_issues']}")

    # Create visualization
    fig = analyzer.create_visualization('histogram', x='age', title='Age Distribution')
    fig.savefig('age_histogram.png')
    print("\nSaved age_histogram.png")

    # If PandasAI is available and API key is set
    if PANDASAI_AVAILABLE and os.environ.get('OPENAI_API_KEY'):
        print("\nChatting with data...")
        result = analyzer.chat("What is the average salary by department?")
        print(f"Result: {result}")
    else:
        print("\nPandasAI chat not available. Set OPENAI_API_KEY to enable.")
