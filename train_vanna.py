import streamlit as st
import os
from typing import Dict, List
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


class VannaTrainer:
    """Handles training Vanna AI with DDL, documentation, and sample queries using Weaviate Cloud."""
    
    def __init__(self):
        self.vn = None
        
    def setup_vanna_connection(self, db_path: str):
        """Setup Vanna connection with Weaviate Cloud and custom database."""
        try:
            # Get Weaviate configuration from environment or Streamlit secrets
            weaviate_url = os.getenv("WEAVIATE_URL") or st.secrets.get("WEAVIATE_URL")
            weaviate_api_key = os.getenv("WEAVIATE_API_KEY") or st.secrets.get("WEAVIATE_API_KEY")
            google_api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
            model_name = os.getenv("WEAVIATE_MODEL", "excel_data_model") or st.secrets.get("WEAVIATE_MODEL", "excel_data_model")
            
            if not weaviate_url or not weaviate_api_key:
                st.error("❌ Weaviate Cloud configuration missing. Please set WEAVIATE_URL and WEAVIATE_API_KEY in your .env file or Streamlit secrets.")
                return False
            
            if not google_api_key:
                st.error("❌ Google API key missing. Please set GOOGLE_API_KEY in your .env file or Streamlit secrets.")
                return False
            
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
            
            self.vn = VannaWeaviate(config=config)
            
            # Connect to SQLite database
            self.vn.connect_to_sqlite(db_path)
            
            st.success(f"✅ Connected to Weaviate Cloud at {weaviate_url}")
            return True
            
        except Exception as e:
            st.error(f"❌ Error setting up Weaviate connection: {str(e)}")
            st.info("💡 Make sure your Weaviate Cloud cluster is running and accessible.")
            return False
    
    def train_on_ddl(self, ddl: str) -> bool:
        """Train Vanna on DDL statement."""
        try:
            if self.vn:
                self.vn.train(ddl=ddl)
                return True
            return False
        except Exception as e:
            st.error(f"Error training on DDL: {str(e)}")
            return False
    
    def train_on_documentation(self, documentation: str) -> bool:
        """Train Vanna on documentation text."""
        try:
            if self.vn:
                self.vn.train(documentation=documentation)
                return True
            return False
        except Exception as e:
            st.error(f"Error training on documentation: {str(e)}")
            return False
    
    def train_on_question_sql_pair(self, question: str, sql: str) -> bool:
        """Train Vanna on question-SQL pairs."""
        try:
            if self.vn:
                self.vn.train(question=question, sql=sql)
                return True
            return False
        except Exception as e:
            st.error(f"Error training on question-SQL pair: {str(e)}")
            return False
    
    def train_on_sql_queries(self, queries: List[str]) -> int:
        """Train Vanna on a list of sample SQL queries."""
        trained_count = 0
        
        for query in queries:
            try:
                # Extract question from comment and SQL
                lines = query.split('\n')
                if len(lines) >= 2:
                    question = lines[0].replace('-- ', '').strip()
                    sql = '\n'.join(lines[1:]).strip()
                    
                    if sql and question and self.train_on_question_sql_pair(question, sql):
                        trained_count += 1
            except Exception as e:
                st.warning(f"Skipped query due to error: {str(e)}")
                continue
        
        return trained_count
    
    def train_comprehensive(self, 
                          ddls: List[str] = None,
                          documentation_files: Dict[str, str] = None,
                          predefined_docs: List[Dict] = None,
                          sample_queries: Dict[str, List[str]] = None) -> Dict[str, int]:
        """Comprehensive training on all provided data."""
        
        training_stats = {
            'ddl_count': 0,
            'doc_count': 0,
            'predefined_doc_count': 0,
            'query_count': 0,
            'total_count': 0
        }
        
        # Train on DDLs
        if ddls:
            for ddl in ddls:
                if self.train_on_ddl(ddl):
                    training_stats['ddl_count'] += 1
                    training_stats['total_count'] += 1
        
        # Train on predefined documentation
        if predefined_docs:
            for doc_entry in predefined_docs:
                if 'documentation' in doc_entry:
                    if self.train_on_documentation(doc_entry['documentation']):
                        training_stats['predefined_doc_count'] += 1
                        training_stats['total_count'] += 1
        
        # Train on generated documentation files
        if documentation_files:
            for sheet_name, doc_file in documentation_files.items():
                if os.path.exists(doc_file):
                    try:
                        with open(doc_file, 'r', encoding='utf-8') as f:
                            doc_content = f.read()
                        
                        if self.train_on_documentation(doc_content):
                            training_stats['doc_count'] += 1
                            training_stats['total_count'] += 1
                    except Exception as e:
                        st.warning(f"Error reading documentation file for {sheet_name}: {str(e)}")
        
        # Train on sample queries
        if sample_queries:
            for sheet_name, queries in sample_queries.items():
                query_count = self.train_on_sql_queries(queries)
                training_stats['query_count'] += query_count
                training_stats['total_count'] += query_count
        
        return training_stats


def create_training_interface():
    """Create the training interface for Vanna."""
    st.subheader("🤖 Train Vanna AI")
    st.markdown("Train Vanna AI on your data using **Weaviate Cloud** for vector storage and natural language querying.")
    
    # Weaviate Configuration Status
    with st.expander("🔧 Weaviate Cloud Configuration", expanded=False):
        weaviate_url = os.getenv("WEAVIATE_URL") or st.secrets.get("WEAVIATE_URL")
        weaviate_api_key = os.getenv("WEAVIATE_API_KEY") or st.secrets.get("WEAVIATE_API_KEY")
        google_api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
        model_name = os.getenv("WEAVIATE_MODEL", "excel_data_model") or st.secrets.get("WEAVIATE_MODEL", "excel_data_model")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Configuration Status:**")
            weaviate_status = "✅ Connected" if weaviate_url else "❌ Missing URL"
            api_key_status = "✅ Configured" if weaviate_api_key else "❌ Missing API Key"
            google_status = "✅ Configured" if google_api_key else "❌ Missing Google API Key"
            
            st.markdown(f"- Weaviate URL: {weaviate_status}")
            st.markdown(f"- Weaviate API Key: {api_key_status}")
            st.markdown(f"- Google API Key: {google_status}")
        
        with col2:
            st.markdown("**Configuration Details:**")
            if weaviate_url:
                st.markdown(f"- **Cluster**: {weaviate_url}")
            st.markdown(f"- **Model Class**: {model_name}")
            st.markdown(f"- **LLM Model**: gemini-pro")
        
        if not all([weaviate_url, weaviate_api_key, google_api_key]):
            st.error("❌ **Missing Configuration**: Please set up your Weaviate Cloud and Google API credentials in your `.env` file or Streamlit secrets.")
            st.markdown("""
            **Required Environment Variables:**
            ```
            WEAVIATE_URL=https://your-cluster-url.weaviate.network
            WEAVIATE_API_KEY=your-weaviate-api-key
            GOOGLE_API_KEY=your-google-api-key
            WEAVIATE_MODEL=excel_data_model  # optional
            ```
            """)
            return
    
    # Check if we have data to train on
    if 'db_path' not in st.session_state or not st.session_state.get('db_path'):
        st.warning("⚠️ No database found. Please generate DDL and documentation first in the 'Upload Excel Data' page.")
        return
    
    trainer = VannaTrainer()
    
    # Training options
    col1, col2 = st.columns(2)
    
    with col1:
        train_ddl = st.checkbox("Train on DDL statements", value=True)
        train_docs = st.checkbox("Train on generated documentation", value=True)
    
    with col2:
        train_queries = st.checkbox("Train on sample SQL queries", value=True)
    
    # Advanced options
    with st.expander("🔧 Advanced Training Options", expanded=False):
        st.markdown("**Training Configuration**")
        
        batch_size = st.slider("Batch size for training", 1, 50, 10, 
                              help="Number of items to train at once")
        
        show_progress = st.checkbox("Show detailed progress", value=True,
                                   help="Display detailed training progress")
    
    # Training button
    if st.button("🎯 Start Vanna Training", type="primary"):
        with st.spinner("Initializing Vanna connection..."):
            if not trainer.setup_vanna_connection(st.session_state['db_path']):
                st.error("Failed to connect to Vanna. Please check your API key.")
                return
        
        # Prepare training data
        ddls = []
        documentation_files = {}
        sample_queries = {}
        
        # Collect DDLs
        if train_ddl and 'generated_ddls' in st.session_state:
            ddls = list(st.session_state['generated_ddls'].values())
        
        # Collect documentation files
        if train_docs and 'doc_files' in st.session_state:
            documentation_files = st.session_state['doc_files']
        
        # Collect sample queries
        if train_queries and 'generated_docs_dict' in st.session_state:
            from excel_processor import ExcelProcessor
            processor = ExcelProcessor()
            
            for sheet_name, docs in st.session_state['generated_docs_dict'].items():
                sample_queries[sheet_name] = processor.generate_sample_sql_queries(docs)
        
        # No predefined documentation - training only on uploaded Excel data
        predefined_docs = None
        
        # Prepare all training data from session state
        progress_placeholder = st.empty()
        
        # Collect DDLs from generated content
        if train_ddl and 'generated_ddls' in st.session_state:
            ddls = list(st.session_state['generated_ddls'].values())
            if show_progress:
                progress_placeholder.info(f"📊 Found {len(ddls)} DDL statements to train on")
        
        # Collect documentation files from generated content
        if train_docs and 'doc_files' in st.session_state:
            documentation_files = st.session_state['doc_files']
            if show_progress:
                progress_placeholder.info(f"📄 Found {len(documentation_files)} documentation files to train on")
        
        # Generate and collect sample queries from generated documentation
        if train_queries and 'generated_docs_dict' in st.session_state:
            from excel_processor import ExcelProcessor
            processor = ExcelProcessor()
            
            sample_queries = {}
            total_queries = 0
            all_docs = st.session_state['generated_docs_dict']
            
            for sheet_name, docs in all_docs.items():
                # Pass all table docs for join query generation
                queries = processor.generate_sample_sql_queries(docs, all_docs)
                sample_queries[sheet_name] = queries
                total_queries += len(queries)
            
            if show_progress:
                progress_placeholder.info(f"🔍 Generated {total_queries} sample SQL queries for training (10 per table)")
        
        with st.spinner("Training Vanna AI on Weaviate Cloud..."):
            if show_progress:
                progress_placeholder.info("🚀 Starting comprehensive training on Weaviate Cloud...")
            
            training_stats = trainer.train_comprehensive(
                ddls=ddls,
                documentation_files=documentation_files,
                predefined_docs=predefined_docs,
                sample_queries=sample_queries
            )
        
        # Display results
        progress_placeholder.empty()
        
        if training_stats['total_count'] > 0:
            st.success(f"🎉 Vanna AI training completed successfully!")
            
            # Training statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("DDL Statements", training_stats['ddl_count'])
            with col2:
                st.metric("Documentation Files", training_stats['doc_count'])
            with col3:
                st.metric("Predefined Docs", training_stats['predefined_doc_count'])
            with col4:
                st.metric("SQL Queries", training_stats['query_count'])
            
            st.info(f"**Total Training Items**: {training_stats['total_count']}")
            st.info("✅ You can now ask questions about your data using natural language in the main chat interface.")
            
            # Sample questions
            st.markdown("**Sample questions you can now ask:**")
            
            # Generate dynamic sample questions based on actual trained data
            sample_questions = []
            
            # Add general questions
            sample_questions.extend([
                "How many records are in each table?",
                "What are the column names and data types?",
                "Show me the first 10 rows from each table",
                "Which table has the most records?",
                "What are the data types used in the database?"
            ])
            
            # Add specific questions based on generated DDLs and documentation
            if 'generated_ddls' in st.session_state and 'generated_docs_dict' in st.session_state:
                from excel_processor import ExcelProcessor
                processor = ExcelProcessor()
                
                for sheet_name, docs in st.session_state['generated_docs_dict'].items():
                    table_name = processor.clean_column_name(sheet_name)
                    
                    # Basic table questions
                    sample_questions.extend([
                        f"How many rows are in the {table_name} table?",
                        f"What columns does the {table_name} table have?",
                        f"Show me sample data from {table_name}",
                    ])
                    
                    # Questions based on column types
                    for col_name, col_info in docs['columns'].items():
                        data_type = col_info['data_type']
                        
                        if 'INTEGER' in data_type or 'REAL' in data_type:
                            sample_questions.extend([
                                f"What are the min, max, and average values for {col_name}?",
                                f"Show me the distribution of {col_name} values",
                            ])
                        
                        if 'VARCHAR' in data_type or 'TEXT' in data_type:
                            if col_info['unique_values'] < docs['total_rows'] * 0.8:  # Categorical-like
                                sample_questions.extend([
                                    f"What are the unique values in {col_name}?",
                                    f"Count records by {col_name}",
                                ])
                        
                        if 'DATE' in data_type:
                            sample_questions.extend([
                                f"What is the date range for {col_name}?",
                                f"Show me records grouped by year for {col_name}",
                            ])
            
            # Sample questions are now based only on uploaded Excel data
            
            # Display sample questions (limit to 15 for readability)
            unique_questions = list(dict.fromkeys(sample_questions))  # Remove duplicates
            for question in unique_questions[:15]:
                st.markdown(f"- {question}")
            
            if len(unique_questions) > 15:
                st.info(f"💡 And {len(unique_questions) - 15} more questions based on your specific data!")
            
        else:
            st.error("❌ No data was successfully trained. Please check your configuration and try again.")


def create_training_status_display():
    """Display current training status and model information."""
    st.subheader("📊 Training Data Status")
    
    if 'db_path' not in st.session_state:
        st.info("📋 No training session active. Upload Excel data and generate DDL first in the 'Upload Excel Data' page.")
        return
    
    # Display current session info
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📊 Generated Content:**")
        ddl_count = len(st.session_state.get('generated_ddls', {}))
        doc_count = len(st.session_state.get('doc_files', {}))
        
        st.metric("DDL Statements", ddl_count)
        st.metric("Documentation Files", doc_count)
    
    with col2:
        st.markdown("**🔍 Sample Queries:**")
        if 'generated_docs_dict' in st.session_state:
            # Each table gets exactly 10 queries
            table_count = len(st.session_state['generated_docs_dict'])
            total_queries = table_count * 10
            
            st.metric("SQL Query Samples", total_queries)
        else:
            st.metric("SQL Query Samples", 0)
        
        # No predefined documentation
        st.metric("Predefined Docs", 0)
    
    with col3:
        st.markdown("**💾 Database Info:**")
        if 'db_path' in st.session_state and os.path.exists(st.session_state['db_path']):
            db_size = os.path.getsize(st.session_state['db_path']) / 1024  # KB
            st.metric("Database Size", f"{db_size:.1f} KB")
            
            # Count tables in database
            import sqlite3
            try:
                conn = sqlite3.connect(st.session_state['db_path'])
                cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()
                conn.close()
                st.metric("Database Tables", len(tables))
            except:
                st.metric("Database Tables", "Error")
        else:
            st.metric("Database Size", "Not Found")
    
    # Show detailed training data breakdown
    with st.expander("🔍 Detailed Training Data Breakdown", expanded=False):
        if 'generated_ddls' in st.session_state and st.session_state['generated_ddls']:
            st.markdown("**📊 Available DDL Statements:**")
            for sheet_name, ddl in st.session_state['generated_ddls'].items():
                lines = ddl.count('\n') + 1
                st.markdown(f"- **{sheet_name}**: {lines} lines")
        
        if 'doc_files' in st.session_state and st.session_state['doc_files']:
            st.markdown("**📄 Available Documentation Files:**")
            for sheet_name, file_path in st.session_state['doc_files'].items():
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path) / 1024  # KB
                    st.markdown(f"- **{sheet_name}**: ✅ {file_size:.1f} KB")
                else:
                    st.markdown(f"- **{sheet_name}**: ❌ File not found")
        
        if 'generated_docs_dict' in st.session_state and st.session_state['generated_docs_dict']:
            st.markdown("**🔍 Sample Queries by Table:**")
            
            for sheet_name in st.session_state['generated_docs_dict'].keys():
                st.markdown(f"- **{sheet_name}**: 10 sample queries (3 Easy, 4 Medium, 3 Hard)")
    
    # Training readiness check
    ready_for_training = (
        'generated_ddls' in st.session_state and 
        'doc_files' in st.session_state and 
        'db_path' in st.session_state
    )
    
    if ready_for_training:
        st.success("✅ **Ready for Training**: All required data is available for Vanna AI training.")
    else:
        st.warning("⚠️ **Not Ready**: Please upload Excel data and generate DDL/documentation first.")
        
        missing_items = []
        if 'generated_ddls' not in st.session_state:
            missing_items.append("DDL statements")
        if 'doc_files' not in st.session_state:
            missing_items.append("Documentation files")
        if 'db_path' not in st.session_state:
            missing_items.append("Database connection")
        
        if missing_items:
            st.info(f"Missing: {', '.join(missing_items)}")
