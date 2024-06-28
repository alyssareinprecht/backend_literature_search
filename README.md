# Flask Backend README

## Introduction
This Flask backend provides a set of APIs to rank research papers based on keywords, generate word clouds and retrieve keyword and document information. This guide will walk you through setting up and running the application.

## Prerequisites
Before running the application, ensure you have the following software installed:
- Python 3.6+
- pip (Python package installer)

## Installation

1. **Use the provided folder OR clone the repository**
    ```bash
    git clone https://github.com/alyssareinprecht/backend_literature_search
    cd backend_literature_search
    ```

3. **Install required packages:**
    ```bash
    pip install -r requirements.txt
    ```
    Or install them seperatley:
    ```
    Flask==2.0.3
    Flask-CORS==3.0.10
    pandas==1.3.1
    wordcloud==1.9.3
    ```

4. **Prepare the data:**
    Ensure you have a CSV file named `research_papers_with_wordinfo.csv` inside a folder named `data`. This CSV file should contain the necessary document data.

## Running the Application

1. **Start the Flask server:**
    ```bash
    python app.py
    ```
    or 
    ```bash
    py app.py
    ```

2. **Access the application:**
    Open your web browser and navigate to `http://127.0.0.1:5000/` or click directly on the link provided in the terminal after running the aforementioned command.