import streamlit as st
import os
from dotenv import load_dotenv
from vanna.weaviate import WeaviateDatabase
from vanna.base import VannaBase
import google.generativeai as genai

# Load environment variables
load_dotenv()

class GeminiChat(VannaBase):
    """Custom Gemini chat implementation for Vanna"""
    
    def __init__(self, config=None):
        VannaBase.__init__(self, config=config)
        if config and 'google_api_key' in config:
            genai.configure(api_key=config['google_api_key'])
            self.model = genai.GenerativeModel(config.get('model', 'gemini-2.0-flash-exp'))
    
    def system_message(self, message: str) -> dict:
        return {"role": "system", "content": message}
    
    def user_message(self, message: str) -> dict:
        return {"role": "user", "content": message}
    
    def assistant_message(self, message: str) -> dict:
        return {"role": "assistant", "content": message}
    
    # Remove custom generate_sql - let Vanna's base class handle it
    # Vanna will use submit_prompt() for LLM calls
    
    # Remove custom generate_explanation - let Vanna's base class handle it
    # Vanna will use submit_prompt() for LLM calls
    
    def submit_prompt(self, prompt, **kwargs) -> str:
        """Submit prompt to Gemini - required abstract method"""
        try:
            # Handle different prompt formats
            if isinstance(prompt, list):
                # Convert Vanna's message format to plain text
                text_prompt = ""
                for message in prompt:
                    if isinstance(message, dict):
                        if 'content' in message:
                            text_prompt += message['content'] + "\n"
                        elif 'text' in message:
                            text_prompt += message['text'] + "\n"
                    else:
                        text_prompt += str(message) + "\n"
                prompt = text_prompt.strip()
            elif isinstance(prompt, dict):
                # Single message dict
                if 'content' in prompt:
                    prompt = prompt['content']
                elif 'text' in prompt:
                    prompt = prompt['text']
                else:
                    prompt = str(prompt)
            
            # Ensure prompt is a string
            if not isinstance(prompt, str):
                prompt = str(prompt)
            
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error: {str(e)}"

@st.cache_resource(ttl=3600)
def setup_vanna():
    """Setup Vanna with Weaviate Cloud connection."""
    try:
        # Get Weaviate configuration from environment or Streamlit secrets
        weaviate_url = os.getenv("WEAVIATE_URL") or st.secrets.get("WEAVIATE_URL")
        weaviate_api_key = os.getenv("WEAVIATE_API_KEY") or st.secrets.get("WEAVIATE_API_KEY")
        google_api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
        model_name = os.getenv("WEAVIATE_MODEL", "excel_data_model") or st.secrets.get("WEAVIATE_MODEL", "excel_data_model")
        
        if not weaviate_url or not weaviate_api_key or not google_api_key:
            st.error("❌ Missing Weaviate or Google API configuration. Please set up your .env file or Streamlit secrets.")
            return None
        
        # Create Vanna instance with Weaviate and Google Gemini
        class VannaWeaviate(WeaviateDatabase, GeminiChat):
            def __init__(self, config=None):
                WeaviateDatabase.__init__(self, config=config)
                GeminiChat.__init__(self, config=config)
        
        # Configuration for Weaviate
        config = {
            'weaviate_url': weaviate_url,
            'weaviate_api_key': weaviate_api_key,
            'google_api_key': google_api_key,
            'model': 'gemini-2.0-flash-exp',
            'class_name': model_name
        }
        
        vn = VannaWeaviate(config=config)
        
        # Connect to database
        if 'db_path' in st.session_state and st.session_state['db_path']:
            vn.connect_to_sqlite(st.session_state['db_path'])
        else:
            # Check for excel_data.db in project directory
            project_dir = os.getcwd()
            excel_db_path = os.path.join(project_dir, "excel_data.db")
            if os.path.exists(excel_db_path):
                vn.connect_to_sqlite(excel_db_path)
                # Store in session state for consistency
                st.session_state['db_path'] = excel_db_path
                st.success(f"✅ Connected to Excel database: {excel_db_path}")
            else:
                # Fallback to Chinook database if no custom database
                vn.connect_to_sqlite("https://vanna.ai/Chinook.sqlite")
        
        return vn
        
    except Exception as e:
        st.error(f"❌ Error setting up Vanna with Weaviate: {str(e)}")
        return None

def setup_vanna_with_custom_db(db_path: str):
    """Setup Vanna with Weaviate Cloud and a custom database path."""
    try:
        # Get Weaviate configuration
        weaviate_url = os.getenv("WEAVIATE_URL") or st.secrets.get("WEAVIATE_URL")
        weaviate_api_key = os.getenv("WEAVIATE_API_KEY") or st.secrets.get("WEAVIATE_API_KEY")
        google_api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
        model_name = os.getenv("WEAVIATE_MODEL", "excel_data_model") or st.secrets.get("WEAVIATE_MODEL", "excel_data_model")
        
        if not weaviate_url or not weaviate_api_key or not google_api_key:
            raise ValueError("Missing Weaviate or Google API configuration")
        
        # Create Vanna instance
        class VannaWeaviate(WeaviateDatabase, GeminiChat):
            def __init__(self, config=None):
                WeaviateDatabase.__init__(self, config=config)
                GeminiChat.__init__(self, config=config)
        
        config = {
            'weaviate_url': weaviate_url,
            'weaviate_api_key': weaviate_api_key,
            'google_api_key': google_api_key,
            'model': 'gemini-2.0-flash-exp',
            'class_name': model_name
        }
        
        vn = VannaWeaviate(config=config)
        vn.connect_to_sqlite(db_path)
        return vn
        
    except Exception as e:
        st.error(f"❌ Error setting up Vanna with custom database: {str(e)}")
        return None

@st.cache_data(show_spinner="Generating sample questions ...")
def generate_questions_cached():
    vn = setup_vanna()
    return vn.generate_questions()


@st.cache_data(show_spinner="Generating SQL query ...")
def generate_sql_cached(question: str):
    vn = setup_vanna()
    return vn.generate_sql(question=question, allow_llm_to_see_data=True)

@st.cache_data(show_spinner="Checking for valid SQL ...")
def is_sql_valid_cached(sql: str):
    vn = setup_vanna()
    return vn.is_sql_valid(sql=sql)

def run_sql_cached(sql: str):
    vn = setup_vanna()
    
    # Clean the SQL query - remove markdown formatting
    cleaned_sql = sql.strip()
    if cleaned_sql.startswith('```'):
        # Remove markdown code block formatting
        lines = cleaned_sql.split('\n')
        cleaned_sql = '\n'.join(line for line in lines if not line.strip().startswith('```'))
        cleaned_sql = cleaned_sql.strip()
    
    # Remove any language identifiers
    if cleaned_sql.lower().startswith('sqlite'):
        cleaned_sql = cleaned_sql[6:].strip()
    
    # Debug info
    st.info(f"🔍 Executing SQL: {cleaned_sql}")
    
    try:
        result = vn.run_sql(sql=cleaned_sql)
        if result is not None:
            st.success(f"✅ Query executed successfully! Found {len(result)} rows.")
        return result
    except Exception as e:
        st.error(f"❌ SQL Error: {str(e)}")
        return None

@st.cache_data(show_spinner="Checking if we should generate a chart ...")
def should_generate_chart_cached(question, sql, df):
    vn = setup_vanna()
    return vn.should_generate_chart(df=df)

@st.cache_data(show_spinner="Generating Plotly code ...")
def generate_plotly_code_cached(question, sql, df):
    vn = setup_vanna()
    code = vn.generate_plotly_code(question=question, sql=sql, df=df)
    return code


@st.cache_data(show_spinner="Running Plotly code ...")
def generate_plot_cached(code, df):
    vn = setup_vanna()
    return vn.get_plotly_figure(plotly_code=code, df=df)


@st.cache_data(show_spinner="Generating followup questions ...")
def generate_followup_cached(question, sql, df):
    vn = setup_vanna()
    return vn.generate_followup_questions(question=question, sql=sql, df=df)

@st.cache_data(show_spinner="Generating summary ...")
def generate_summary_cached(question, df):
    vn = setup_vanna()
    return vn.generate_summary(question=question, df=df)