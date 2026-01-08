# 📊 SQLGenPro - Complete Repository Documentation Report

## 📋 Executive Summary

**SQLGenPro** is a Text-to-SQL application that leverages OpenAI's LLM (Large Language Model) capabilities combined with LangChain framework to convert natural language questions into executable SQL queries. The application is built using Streamlit for the web interface and connects to Databricks SQL for data warehousing.

**Target Users:** Product Managers, Business Stakeholders, and Intermediate Coders who need to query SQL databases without writing complex SQL code.

---

## 🏗️ Project Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                           │
│                      (Streamlit Web App)                        │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     AUTHENTICATION LAYER                        │
│              (streamlit_authenticator + YAML)                   │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      LLM PROCESSING LAYER                       │
│            (LangChain + OpenAI GPT-4o-mini)                     │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ ERD Diagram │  │ SQL Query   │  │ Quick Analysis          │  │
│  │ Generation  │  │ Generation  │  │ Question Generation     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      DATA LAYER                                 │
│                   (Databricks SQL)                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 File-by-File Detailed Explanation

---

### 1️⃣ `SQLGenPro.py` - Main Application File (244 lines)

This is the **primary Streamlit application** file. Here's a line-by-line breakdown:

#### **Lines 1-12: Import Statements**
```python
import os, sys 
import pandas as pd 
import numpy as np
import streamlit as st
import sqlparse
from collections import OrderedDict, Counter
from github import Github
from databricks import sql 
import streamlit_authenticator as stauth
import yaml 
from yaml.loader import SafeLoader
from dotenv import load_dotenv
load_dotenv()
```
- **`os, sys`**: Operating system and system path manipulation
- **`pandas, numpy`**: Data manipulation and numerical operations
- **`streamlit`**: Web application framework
- **`sqlparse`**: SQL parsing utilities
- **`databricks.sql`**: Databricks SQL connector for database operations
- **`streamlit_authenticator`**: User authentication module
- **`yaml`**: YAML file parsing for configuration
- **`dotenv`**: Load environment variables from `.env` file

#### **Lines 15-20: Custom Module Imports**
```python
sys.path.append(os.path.abspath('src'))
from src.add_logo import add_logo
from src.utils import list_catalog_schema_tables, create_erd_diagram, mermaid, ...
```
- Adds the `src` folder to Python path
- Imports custom utility functions for:
  - Logo rendering
  - Database schema operations
  - ERD diagram generation
  - SQL generation and validation

#### **Lines 23-30: Streamlit Page Configuration**
```python
st.set_page_config(
    page_title="SQLGenPro",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="expanded",
)
```
- Sets up the Streamlit page with title, icon, and layout

#### **Lines 33-37: Application Header**
```python
st.markdown("<h1 style='text-align: center; color: orange;'> SQLGenPro &#128640; </h1>", unsafe_allow_html=True)
st.markdown("<h6 style='text-align: center; color: white;'> Productivity Improvement tool...</h6>", unsafe_allow_html=True)
add_logo("artifacts/project_pro_logo_white.png")
```
- Displays the application title with custom styling
- Adds the company logo to the sidebar

#### **Lines 40-56: User Authentication**
```python
with open('authenticator.yml') as f:
    config = yaml.load(f, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

name, authentication_status, user_name = authenticator.login()
```
- Loads authentication configuration from YAML file
- Creates authenticator object with credentials
- Handles user login flow

#### **Lines 58-85: Database Connection & Schema Selection**
```python
if authentication_status:
    result_tables = list_catalog_schema_tables()
    df_databricks = pd.DataFrame(result_tables).iloc[:,:4]
    df_databricks.columns=["catalog","schema","table","table_type"]
    
    catalog = st.sidebar.selectbox("Select the catalog", ...)
    schema = st.sidebar.selectbox("Select the schema", ...)
    table_list = st.sidebar.multiselect("Select the table", ...)
```
- After authentication, fetches all tables from Databricks
- Creates dynamic dropdown menus for:
  - **Catalog Selection**: Top-level database container
  - **Schema Selection**: Filtered based on selected catalog
  - **Table Selection**: Filtered based on selected schema

#### **Lines 86-100: ERD Diagram Generation**
```python
with st.expander(":red[View the ERD Diagram]"):
    response = create_erd_diagram(catalog,schema,table_list)
    mermaid_code = process_llm_response_for_mermaid(response)
    mermaid(mermaid_code)
    table_schema = get_enriched_database_schema(catalog,schema,table_list)
```
- Uses LLM to generate Entity-Relationship Diagram
- Converts LLM response to Mermaid.js code
- Renders interactive ERD diagram
- Retrieves enriched schema for SQL generation

#### **Lines 102-170: Quick Analysis Section**
```python
quick_analysis_questions = quick_analysis(user_name,mermaid_code)
selected_question = st.selectbox("Select the question", options=questions)
if st.checkbox("Analyze"):
    response_sql_qa = create_sql(selected_question,table_schema)
    response_sql_qa = process_llm_response_for_sql(response_sql_qa)
    
    # Self-correction loop
    flag, response_sql_qa = validate_and_correct_sql(response_sql_qa,table_schema)
    while flag != 'Correct':
        flag, response_sql_qa = validate_and_correct_sql(response_sql_qa,table_schema)
```
- **LLM generates 5 suggested business questions** based on schema
- User selects a question from dropdown
- **SQL query is auto-generated** from natural language
- **Self-correction loop**: If SQL fails, LLM corrects it automatically
- Results can be queried and saved to favorites

#### **Lines 172-244: Deep-Dive Analysis Section**
```python
dd_question = st.text_area("Enter your question here..")
if generate_sql_1:
    response_sql_1 = create_sql(dd_question,table_schema)
    # Self-correction loop
    flag, response_sql_1 = validate_and_correct_sql(dd_question,response_sql_1,table_schema)
    
    if build_1:
        response_sql_2 = create_advanced_sql(dd_question_2,response_sql_1,table_schema)
```
- **Custom question input**: Users type their own business questions
- **Advanced SQL generation**: Build on top of previous queries
- **Iterative query refinement**: Chain multiple questions together

---

### 2️⃣ `SQLGenPro_Live.py` - Production Version (265 lines)

This is the **live/production version** with additional features:

#### **Additional Feature: Favorites Section (Lines 180-200)**
```python
st.markdown("<h2 style='text-align: left; color: red;'> Your Favourites </h2>")
with st.expander(":red[View the Section]"):
    fav_df = load_user_query_history(user_name=name)
    fav_question = st.selectbox("Select the question",options=fav_df['question'].unique().tolist())
    fav_sql = fav_df[fav_df['question']==fav_question]['query'].values[0]
```
- Loads user's saved favorite queries from database
- Allows quick re-execution of previously saved queries
- Provides query history management

---

### 3️⃣ `src/utils.py` - Core Utility Functions (448 lines)

This file contains **all the business logic** and LLM interactions:

#### **Database Connection Functions**

```python
@st.cache_data
def load_user_query_history(user_name):
    conn = sql.connect(server_hostname = os.getenv("DATABRICKS_SERVER_HOSTNAME"),
                    http_path       = os.getenv("DATABRICKS_HTTP_PATH"),
                    access_token    = os.getenv("DATABRICKS_ACCESS_TOKEN"))
    query = f"SELECT * FROM dev_tools.sqlgenpro_user_query_history WHERE user_name = '{user_name}'"
    df = pd.read_sql(sql=query,con=conn)
    return df
```
- **`@st.cache_data`**: Caches results to avoid repeated database calls
- Connects to Databricks using environment variables
- Retrieves user's query history

#### **`list_catalog_schema_tables()`**
```python
@st.cache_data
def list_catalog_schema_tables():
    with sql.connect(...) as connection:
        with connection.cursor() as cursor:
            cursor.tables()
            result_tables = cursor.fetchall()
            return result_tables
```
- Lists all catalogs, schemas, and tables in Databricks
- Uses context manager for proper connection handling

#### **`get_enriched_database_schema()`**
```python
def get_enriched_database_schema(catalog,schema,tables_list):
    for table in tables_list:
        # 1. Get CREATE TABLE statement
        query = f"SHOW CREATE TABLE `{catalog}`.{schema}.{table}"
        
        # 2. Get column data types
        query = f"DESCRIBE TABLE `{catalog}`.{schema}.{table}"
        
        # 3. Get categorical column values (for columns with ≤20 distinct values)
        sql_distinct = f"SELECT ARRAY_AGG(DISTINCT {col}) AS values FROM ..."
        
        # 4. Get sample rows
        query = f"SELECT * FROM ... LIMIT 3"
```
- **Creates rich context for LLM** including:
  - Table structure (CREATE TABLE statement)
  - Column names and data types
  - Categorical values (for string columns with ≤20 unique values)
  - Sample data rows

#### **LLM Processing Functions**

**`process_llm_response_for_mermaid()`**
```python
def process_llm_response_for_mermaid(response: str) -> str:
    start_idx = response.find("```mermaid") + len("```mermaid")
    end_idx = response.find("```", start_idx)
    mermaid_code = response[start_idx:end_idx].strip()
    return mermaid_code
```
- Extracts Mermaid code from LLM markdown response

**`process_llm_response_for_sql()`**
```python
def process_llm_response_for_sql(response: str) -> str:
    start_idx = response.find("```sql") + len("```sql")
    end_idx = response.find("```", start_idx)
    sql_code = response[start_idx:end_idx].strip()
    return sql_code
```
- Extracts SQL code from LLM markdown response

#### **`mermaid()` - Diagram Renderer**
```python
def mermaid(code: str) -> None:
    components.html(
        f"""
        <div id="mermaid-container">
            <pre class="mermaid">{code_escaped}</pre>
        </div>
        <script type="module">
            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
            mermaid.initialize({{ startOnLoad: true }});
        </script>
        """,
        height=800
    )
```
- Renders Mermaid.js diagrams using HTML/JavaScript injection
- Creates interactive ERD visualizations

#### **`create_erd_diagram()` - ERD Generation via LLM**
```python
@st.cache_data 
def create_erd_diagram(catalog,schema,tables_list):
    template_string = """ 
    You are an expert in creating ERD diagrams (Entity Relationship Diagrams) for databases. 
    You have been given the task to create an ERD diagram for the selected tables...
    Generate the Mermaid code for the complete ERD diagram.
    
    {table_schema}
    """
    llm_chain = LLMChain(
        llm=ChatOpenAI(model="gpt-4o-mini",temperature=0),
        prompt=prompt_template
    )
    response = llm_chain.invoke({"table_schema":table_schema})
```
- **Uses GPT-4o-mini** to analyze table structures
- **Generates Mermaid code** for ERD visualization
- **Cached** to avoid repeated LLM calls

#### **`quick_analysis()` - Auto-Generated Business Questions**
```python
@st.cache_data
def quick_analysis(table_schema):
    template_string = """
    Using the provided SCHEMA, generate the top 5 "quick analysis" questions 
    based on the relationships between the tables which can be answered by 
    creating a Databricks SQL code.
    These questions should be practical and insightful, targeting the kind 
    of business inquiries a product manager or analyst would typically investigate.
    
    SCHEMA: {table_schema}
    """
    output_schema = ResponseSchema(name="quick_analysis_questions",...)
    output_parser = StructuredOutputParser.from_response_schemas([output_schema])
```
- LLM generates **5 business-relevant questions** from schema
- Uses **StructuredOutputParser** for JSON output format
- Questions are actionable SQL queries

#### **`create_sql()` - Natural Language to SQL**
```python
@st.cache_data
def create_sql(question,table_schema):
    template_string = """ 
    You are a expert data engineer working with a Databricks environment.
    Your task is to generate a working SQL query in Databricks SQL dialect.
    
    Rules:
    - During join if column name are same please use alias
    - If a column is string, the value should be enclosed in quotes
    - If you are writing CTEs then include all the required columns
    - For string columns, check if it is a categorical column and use appropriate values
    
    SCHEMA: {table_schema}
    QUESTION: {question}
    
    OUTPUT: Just the SQL code
    """
```
- **Core Text-to-SQL function**
- Converts natural language questions to SQL queries
- **Prompt engineering** includes rules for proper SQL generation

#### **`create_advanced_sql()` - Build on Existing Queries**
```python
def create_advanced_sql(question,sql_code,table_schema):
    template_string = """ 
    Enclose the complete SQL_CODE in a WITH clause and name it as MASTER.
    DON'T ALTER THE given SQL_CODE.
    Then based on the QUESTION and the master WITH clause, generate the final SQL query.
    
    SQL_CODE: {sql_code}
    SCHEMA: {table_schema}
    QUESTION: {question}
    """
```
- Allows **chaining queries** together
- Wraps previous query in CTE (Common Table Expression)
- Enables iterative data exploration

#### **Self-Correction System**

**`self_correction()` - Error Detection**
```python
def self_correction(query):
    try:
        df = load_data_from_query(query)
        error_msg = "Successful"
    except Exception as e:
        error_msg = str(e)
    return error_msg
```
- Attempts to execute generated SQL
- Returns error message if execution fails

**`correct_sql()` - LLM-based Error Correction**
```python
def correct_sql(question,sql_code,table_schema,error_msg):
    template_string = """ 
    Your task is to modify the SQL_CODE using Databricks SQL dialect 
    based on the QUESTION, SCHEMA and the ERROR_MESSAGE.
    If ERROR_MESSAGE is provided, make sure to correct the SQL query.
    
    SCHEMA: {table_schema}
    ERROR_MESSAGE: {error_msg}
    SQL_CODE: {sql_code}
    QUESTION: {question}
    """
```
- **LLM fixes broken SQL** based on error message
- Uses schema context to understand corrections needed

**`validate_and_correct_sql()` - Main Correction Loop**
```python
def validate_and_correct_sql(question,query,table_schema):
    error_msg = self_correction(query)
    
    if error_msg == "Successful":
        return "Correct", query
    else:
        modified_query = correct_sql(question,query,table_schema,error_msg=error_msg)
        return "Incorrect", modified_query
```
- **Orchestrates the self-correction loop**
- Returns corrected query or confirms success

#### **`add_to_user_history()` - Save Queries**
```python
def add_to_user_history(user_name,question,query,favourite_ind):
    user_history_table = "hive_metastore.dev_tools.sqlgenpro_user_query_history"
    query = f"""INSERT INTO {user_history_table} 
                VALUES ('{user_name}',current_timestamp(),'{question}',"{query}",{favourite_ind})"""
```
- Saves user queries to Databricks table
- Enables query history and favorites functionality

---

### 4️⃣ `src/add_logo.py` - Logo Utility (40 lines)

```python
def get_base64_of_bin_file(png_file):
    with open(png_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def add_logo(png_file):
    logo_markup = build_markup_for_logo(png_file)
    st.markdown(logo_markup, unsafe_allow_html=True)
```
- Converts PNG image to Base64 string
- Injects CSS to display logo in Streamlit sidebar
- Uses `background-image` CSS property

---

### 5️⃣ `authenticator.yml` - Authentication Configuration

```yaml
credentials:
  usernames:
    hariharan:
      email: hariharan@gmail.com
      name: Hariharan
      password: $2b$12$tdKt5p7nfhQgcDG30FldqeTZLUio5P.vAqKXSqiUHxURaei7Npkq2  # abc123
    jack:
      email: jack@gmail.com
      name: Jack Sparrow
      password: $2b$12$N9w9IGLBfLv1tne8lzalUOm2hH4ytnvbTgQ.vs0jV6EbKt9x2QPr6  # blackpearl
cookie:
  expiry_days: 7
  key: random_signature_key
  name: random_cookie_name
preauthorized:
  emails: []
```
- **User credentials storage** with bcrypt-hashed passwords
- **Cookie configuration** for session management
- **Pre-authorized emails** for auto-registration

---

### 6️⃣ `Langchain_Intro.ipynb` - LangChain Tutorial Notebook

This notebook demonstrates **LangChain fundamentals**:

#### **Section 1: Creating Prompts**
```python
template_string = """ 
You are an expert researcher analyst in customer success team.
Given the following customer feedback, provide:
1. Product: Name of the product in review
2. Summary: Summary with sentiment
3. Positives: List positive feedbacks
4. Negatives: List negative feedbacks

Customer Feedback: {feedback}
"""
prompt = PromptTemplate.from_template(template_string)
```
- Demonstrates **PromptTemplate** usage
- Variables in curly braces `{feedback}` are placeholders

#### **Section 2: Creating LLM Chain**
```python
llm_chain = LLMChain(
    llm=ChatOpenAI(model="gpt-4o-mini",temperature=0),
    prompt=prompt
)
output = llm_chain.invoke({"feedback":review})
```
- Combines **LLM + Prompt** into executable chain
- `temperature=0` for deterministic outputs

#### **Section 3: Output Parsing**
```python
output_schema = ResponseSchema(name="review_analysis",description="Review analysis output")
output_parser = StructuredOutputParser.from_response_schemas([output_schema])
format_instructions = output_parser.get_format_instructions()
```
- **Structured output** with JSON schema
- Auto-generates format instructions for LLM

---

### 7️⃣ `helper.ipynb` - Development Helper Notebook

This notebook contains **testing and development code**:

#### **Password Hashing**
```python
from streamlit_authenticator.utilities.hasher import Hasher
hashed_passwords = Hasher(['abc123']).generate()
```
- Generates bcrypt hashes for new user passwords

#### **Snowflake Connection Testing**
```python
con = snowflake.connector.connect(
    user='hariharan',
    password=os.getenv('SNOWSQL_PWD'),
    account=os.getenv('SNOWSQL_ACCOUNT'),
    warehouse=os.getenv('SNOWSQL_WAREHOUSE'),
)
```
- Alternative data warehouse connection (Snowflake)
- Shows database exploration commands

#### **Databricks Connection Testing**
```python
connection = sql.connect(
    server_hostname=os.getenv("DATABRICKS_SERVER_HOSTNAME"),
    http_path=os.getenv("DATABRICKS_HTTP_PATH"),
    access_token=os.getenv("DATABRICKS_ACCESS_TOKEN")
)
df = pd.read_sql("SELECT * FROM hive_metastore.online_food_business.menu_items", connection)
```
- Primary database connection
- Tests SQL execution

#### **OpenAI API Testing**
```python
from openai import OpenAI
client = OpenAI()
completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Compose a poem about recursion"}]
)
```
- Direct OpenAI API usage for testing

---

### 8️⃣ `utils.ipynb` - Utility Function Development

Contains **prototype code** for utility functions before moving to `src/utils.py`:

- ERD diagram generation functions
- Mermaid code processing
- Database schema enrichment
- LLM chain testing

---

### 9️⃣ `requirements.txt` - Python Dependencies (138 packages)

Key dependencies:
| Package | Purpose |
|---------|---------|
| `streamlit` | Web application framework |
| `langchain-openai` | LangChain + OpenAI integration |
| `databricks-sql-connector` | Databricks database connection |
| `streamlit-authenticator` | User authentication |
| `pandas`, `numpy` | Data manipulation |
| `PyGithub` | GitHub API integration |
| `python-dotenv` | Environment variable management |

---

### 🔟 `test.sql` - SQL Test Queries

Contains test SQL queries for **identifying categorical columns**:
```sql
SELECT 'order_id' AS column_name, COUNT(DISTINCT order_id) AS cnt, 
       ARRAY_AGG(DISTINCT order_id) AS values 
FROM `hive_metastore`.online_food_business.orders 
UNION ALL ...
```

---

## 📊 Data Files (CSV)

The `data/` folder contains sample food delivery business data:

| File | Records | Description |
|------|---------|-------------|
| `food_delivery_users.csv` | 200,000 | User profiles (name, email, phone, address) |
| `food_delivery_orders.csv` | 147,500 | Order transactions |
| `food_delivery_restaurants.csv` | 400 | Restaurant details |
| `food_delivery_menu_items.csv` | - | Menu items per restaurant |
| `food_delivery_reviews.csv` | - | Customer reviews |
| `food_delivery_payments.csv` | - | Payment transactions |
| `food_delivery_order_details.csv` | - | Line items per order |

---

## 🔄 Application Flow Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           USER LOGIN                                      │
│                    (authenticator.yml credentials)                        │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                     SELECT DATABASE CONTEXT                               │
│              Catalog → Schema → Tables (Multi-select)                     │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                      GENERATE ERD DIAGRAM                                 │
│     LLM analyzes schema → Generates Mermaid code → Renders diagram        │
└────────────────────────────────┬─────────────────────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
┌─────────────────────────────┐   ┌─────────────────────────────┐
│      QUICK ANALYSIS         │   │      DEEP-DIVE ANALYSIS     │
│                             │   │                             │
│ 1. LLM generates 5 questions│   │ 1. User types custom        │
│ 2. User selects question    │   │    business question        │
│ 3. LLM generates SQL        │   │ 2. LLM generates SQL        │
│ 4. Self-correction loop     │   │ 3. Self-correction loop     │
│ 5. Execute & display results│   │ 4. Execute & display results│
│ 6. Save to favorites        │   │ 5. Build on previous query  │
└─────────────────────────────┘   └─────────────────────────────┘
                    │                         │
                    └────────────┬────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                     SELF-CORRECTION LOOP                                  │
│                                                                           │
│  ┌─────────┐    ┌─────────────┐    ┌────────────┐    ┌─────────────────┐ │
│  │ Execute │───▶│ Error?      │───▶│ Send error │───▶│ LLM corrects    │ │
│  │   SQL   │    │             │    │ to LLM     │    │ SQL query       │ │
│  └─────────┘    └──────┬──────┘    └────────────┘    └────────┬────────┘ │
│                        │ No error                              │          │
│                        ▼                                       │          │
│              ┌──────────────────┐                              │          │
│              │  Return Results  │◀─────────────────────────────┘          │
│              └──────────────────┘                                         │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 🔐 Environment Variables Required

Create a `.env` file with these variables:

```env
# Databricks Configuration
DATABRICKS_SERVER_HOSTNAME=your-databricks-host.cloud.databricks.com
DATABRICKS_HTTP_PATH=/sql/1.0/warehouses/your-warehouse-id
DATABRICKS_ACCESS_TOKEN=dapi_your_access_token

# OpenAI Configuration
OPENAI_API_KEY=sk-your-openai-api-key

# Optional: Snowflake (if using)
SNOWSQL_PWD=your_password
SNOWSQL_ACCOUNT=your_account
SNOWSQL_WAREHOUSE=your_warehouse
```

---

## 🚀 How to Run the Application

### Step 1: Create Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Configure Environment
- Create `.env` file with required variables
- Update `authenticator.yml` with user credentials

### Step 4: Run Application
```bash
streamlit run SQLGenPro.py
```

### Step 5: Access Application
- Open browser to `http://localhost:8501`
- Login with configured credentials

---

## 🧠 Key Technologies Used

| Technology | Purpose |
|------------|---------|
| **Streamlit** | Web application framework |
| **LangChain** | LLM orchestration framework |
| **OpenAI GPT-4o-mini** | Language model for SQL generation |
| **Databricks SQL** | Cloud data warehouse |
| **Mermaid.js** | ERD diagram visualization |
| **bcrypt** | Password hashing |

---

## 📈 Features Summary

| Feature | Description |
|---------|-------------|
| **User Authentication** | Secure login with hashed passwords |
| **Dynamic Schema Selection** | Browse catalogs, schemas, tables |
| **Auto ERD Generation** | LLM-generated entity relationship diagrams |
| **Quick Analysis** | Pre-generated business questions |
| **Deep-Dive Analysis** | Custom natural language queries |
| **Self-Correction** | Automatic SQL error fixing |
| **Query History** | Save and reload favorite queries |
| **Iterative Queries** | Build complex queries step-by-step |

---

## 📝 Notes for Developers

1. **Caching**: Functions decorated with `@st.cache_data` cache results to improve performance
2. **Error Handling**: Self-correction loop handles SQL errors automatically
3. **Prompt Engineering**: Carefully crafted prompts ensure high-quality SQL generation
4. **Security**: Never commit `.env` file or expose API keys
5. **Extensibility**: Add new LLM functions in `src/utils.py`

---

**Document Generated:** January 8, 2026  
**Version:** 1.0  
**Author:** Auto-generated Documentation
