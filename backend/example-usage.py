"""
Example usage of the CSV Analyzer API
Run this after starting app.py
"""

import requests
import json

BASE_URL = "http://localhost:5000/api"

# Create a session to maintain cookies
session = requests.Session()


def test_health():
    """Test health endpoint"""
    response = session.get(f"{BASE_URL}/health")
    print("Health Check:", response.json())
    return response.json()


def upload_csv(filepath):
    """Upload a CSV file"""
    with open(filepath, 'rb') as f:
        files = {'file': f}
        response = session.post(f"{BASE_URL}/upload", files=files)

    result = response.json()
    print("\nUpload Result:")
    print(f"  Rows: {result['eda']['basic_info']['rows']}")
    print(f"  Columns: {result['eda']['basic_info']['columns']}")
    print(f"  Numeric columns: {result['eda']['numeric_columns']}")
    return result


def create_visualization(chart_type, x_column, y_column=None):
    """Create a visualization"""
    data = {
        'chart_type': chart_type,
        'x_column': x_column,
        'y_column': y_column
    }

    response = session.post(
        f"{BASE_URL}/visualize",
        json=data
    )

    result = response.json()
    if 'error' in result:
        print(f"Error: {result['error']}")
    else:
        print(f"\nVisualization created: {chart_type}")
        # Save the image
        if 'image' in result:
            img_data = result['image'].split(',')[1]
            import base64
            with open(f'{chart_type}_chart.png', 'wb') as f:
                f.write(base64.b64decode(img_data))
            print(f"  Saved to {chart_type}_chart.png")

    return result


def chat_with_data(query):
    """Chat with the data using PandasAI"""
    response = session.post(
        f"{BASE_URL}/chat",
        json={'query': query}
    )

    result = response.json()
    print(f"\nQuery: {query}")

    if 'error' in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Response type: {result['type']}")
        if result['type'] == 'text':
            print(f"Answer: {result['response']}")
        elif result['type'] == 'table':
            print(f"Table with {len(result['response'])} rows")

    return result


def execute_custom_code(code):
    """Execute custom pandas code"""
    response = session.post(
        f"{BASE_URL}/custom-code",
        json={'code': code}
    )

    result = response.json()
    print(f"\nCustom code executed")
    if 'error' in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Result: {result.get('response', 'Success')}")

    return result


# ============ MAIN ============

if __name__ == '__main__':
    # Test health
    test_health()

    # Create sample CSV for testing
    sample_csv = """name,age,salary,department,years_experience
Alice,28,75000,Engineering,5
Bob,35,85000,Marketing,8
Charlie,42,95000,Engineering,15
Diana,31,70000,Sales,6
Eve,27,65000,Marketing,3
Frank,45,120000,Engineering,20
Grace,33,80000,Sales,9
Henry,29,72000,Engineering,4
Ivy,38,90000,Marketing,12
Jack,26,60000,Sales,2"""

    with open('sample_data.csv', 'w') as f:
        f.write(sample_csv)

    print("\nCreated sample_data.csv")

    # Upload the CSV
    upload_result = upload_csv('sample_data.csv')

    # Create visualizations
    create_visualization('histogram', 'salary')
    create_visualization('bar', 'department')
    create_visualization('scatter', 'age', 'salary')
    create_visualization('heatmap', None)

    # Chat with data (requires OPENAI_API_KEY)
    chat_with_data("What is the average salary?")
    chat_with_data("Which department has the highest average salary?")
    chat_with_data("Create a bar chart showing average salary by department")

    # Execute custom code
    execute_custom_code("""
result = df.groupby('department')['salary'].agg(['mean', 'min', 'max'])
    """)

    print("\n" + "=" * 50)
    print("All tests completed!")
    print("=" * 50)
