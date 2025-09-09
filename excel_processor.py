import pandas as pd
import sqlite3
import streamlit as st
from typing import Dict, List, Tuple, Optional
import re
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.types import TypeDecorator
import tempfile
import os

class ExcelProcessor:
    """Handles Excel file processing, DDL generation, and database operations."""
    
    def __init__(self):
        self.df = None
        self.table_name = None
        self.column_mappings = {}
        self.data_types = {}
        
    def load_excel_file(self, uploaded_file) -> Dict[str, pd.DataFrame]:
        """Load Excel file and return dictionary of DataFrames (one per sheet)."""
        try:
            # Read Excel file with all sheets
            if uploaded_file.name.endswith('.xlsx') or uploaded_file.name.endswith('.xls'):
                engine = 'openpyxl' if uploaded_file.name.endswith('.xlsx') else 'xlrd'
                excel_data = pd.read_excel(uploaded_file, engine=engine, sheet_name=None)
                return excel_data
            else:
                raise ValueError("Unsupported file format. Please upload .xlsx or .xls files.")
            
        except Exception as e:
            st.error(f"Error loading Excel file: {str(e)}")
            return None
    
    def analyze_data_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """Analyze DataFrame and suggest SQL data types for each column."""
        type_mappings = {}
        
        for col in df.columns:
            col_data = df[col].dropna()
            
            if col_data.empty:
                type_mappings[col] = "TEXT"
                continue
                
            # Check if it's numeric
            if pd.api.types.is_numeric_dtype(col_data):
                if pd.api.types.is_integer_dtype(col_data):
                    type_mappings[col] = "INTEGER"
                else:
                    type_mappings[col] = "REAL"
            # Check if it's datetime
            elif pd.api.types.is_datetime64_any_dtype(col_data):
                type_mappings[col] = "DATETIME"
            # Check if it's boolean
            elif pd.api.types.is_bool_dtype(col_data):
                type_mappings[col] = "BOOLEAN"
            else:
                # For text data, determine if it should be TEXT or VARCHAR
                max_length = col_data.astype(str).str.len().max()
                if max_length <= 255:
                    type_mappings[col] = f"VARCHAR({max_length})"
                else:
                    type_mappings[col] = "TEXT"
        
        return type_mappings
    
    def clean_column_name(self, col_name: str) -> str:
        """Clean column name to be SQL-compatible."""
        # Remove special characters and replace spaces with underscores
        cleaned = re.sub(r'[^a-zA-Z0-9_]', '_', str(col_name))
        # Remove multiple underscores
        cleaned = re.sub(r'_+', '_', cleaned)
        # Remove leading/trailing underscores
        cleaned = cleaned.strip('_')
        # Ensure it doesn't start with a number
        if cleaned and cleaned[0].isdigit():
            cleaned = f"col_{cleaned}"
        # Ensure it's not empty
        if not cleaned:
            cleaned = "unnamed_column"
        return cleaned.lower()
    
    def generate_ddl(self, df: pd.DataFrame, table_name: str, data_types: Dict[str, str] = None) -> str:
        """Generate CREATE TABLE DDL statement."""
        if data_types is None:
            data_types = self.analyze_data_types(df)
        
        # Clean table name
        clean_table_name = self.clean_column_name(table_name)
        
        ddl_parts = [f"CREATE TABLE {clean_table_name} ("]
        
        column_definitions = []
        for col in df.columns:
            clean_col_name = self.clean_column_name(col)
            col_type = data_types.get(col, "TEXT")
            column_definitions.append(f"    {clean_col_name} {col_type}")
        
        ddl_parts.append(",\n".join(column_definitions))
        ddl_parts.append(");")
        
        return "\n".join(ddl_parts)
    
    def generate_documentation(self, df: pd.DataFrame, table_name: str) -> Dict[str, any]:
        """Generate comprehensive documentation for the table and columns."""
        doc = {
            "table_name": table_name,
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "columns": {}
        }
        
        for col in df.columns:
            col_data = df[col].dropna()
            clean_col_name = self.clean_column_name(col)
            
            col_doc = {
                "original_name": col,
                "clean_name": clean_col_name,
                "data_type": self.analyze_data_types(df)[col],
                "non_null_count": len(col_data),
                "null_count": df[col].isnull().sum(),
                "null_percentage": (df[col].isnull().sum() / len(df)) * 100,
                "unique_values": df[col].nunique(),
                "sample_values": []
            }
            
            # Add sample values (first 5 unique non-null values)
            if not col_data.empty:
                sample_values = col_data.drop_duplicates().head(5).tolist()
                col_doc["sample_values"] = [str(val) for val in sample_values]
            
            # Add statistics for numeric columns
            if pd.api.types.is_numeric_dtype(col_data):
                col_doc["statistics"] = {
                    "min": float(col_data.min()) if not col_data.empty else None,
                    "max": float(col_data.max()) if not col_data.empty else None,
                    "mean": float(col_data.mean()) if not col_data.empty else None,
                    "median": float(col_data.median()) if not col_data.empty else None,
                    "std": float(col_data.std()) if not col_data.empty else None
                }
            
            doc["columns"][clean_col_name] = col_doc
        
        return doc
    
    def create_sqlite_database(self, df: pd.DataFrame, table_name: str, db_path: str = None) -> str:
        """Create SQLite database with the DataFrame data."""
        if db_path is None:
            # Create database file in project directory
            project_dir = os.getcwd()
            db_path = os.path.join(project_dir, f"{table_name}.db")
        
        # Clean column names
        df_clean = df.copy()
        df_clean.columns = [self.clean_column_name(col) for col in df.columns]
        
        # Clean table name
        clean_table_name = self.clean_column_name(table_name)
        
        try:
            # Create connection and insert data
            conn = sqlite3.connect(db_path)
            df_clean.to_sql(clean_table_name, conn, if_exists='replace', index=False)
            conn.close()
            
            return db_path
        except Exception as e:
            st.error(f"Error creating database: {str(e)}")
            return None
    
    def create_combined_database(self, sheets_data: Dict[str, pd.DataFrame], db_path: str = None) -> str:
        """Create SQLite database with multiple sheets as separate tables."""
        if db_path is None:
            # Create database file in project directory
            project_dir = os.getcwd()
            db_path = os.path.join(project_dir, "excel_data.db")
        
        try:
            # Create connection
            conn = sqlite3.connect(db_path)
            
            # Insert each sheet as a separate table
            for sheet_name, df in sheets_data.items():
                # Clean column names
                df_clean = df.copy()
                df_clean.columns = [self.clean_column_name(col) for col in df.columns]
                
                # Clean table name
                clean_table_name = self.clean_column_name(sheet_name)
                
                # Insert data
                df_clean.to_sql(clean_table_name, conn, if_exists='replace', index=False)
            
            conn.close()
            return db_path
        except Exception as e:
            st.error(f"Error creating combined database: {str(e)}")
            return None
    
    def format_documentation_for_vanna(self, documentation: Dict) -> str:
        """Format documentation in a way that's useful for Vanna training."""
        doc_text = f"""
# Table: {documentation['table_name']}

## Table Overview
- Total Rows: {documentation['total_rows']:,}
- Total Columns: {documentation['total_columns']}

## Column Definitions

"""
        
        for col_name, col_info in documentation['columns'].items():
            doc_text += f"""### {col_name} ({col_info['original_name']})
- **Data Type**: {col_info['data_type']}
- **Non-null Values**: {col_info['non_null_count']:,} ({100 - col_info['null_percentage']:.1f}%)
- **Unique Values**: {col_info['unique_values']:,}
- **Sample Values**: {', '.join(col_info['sample_values'][:3])}"""
            
            if 'statistics' in col_info and col_info['statistics']:
                stats = col_info['statistics']
                if stats['mean'] is not None:
                    doc_text += f"""
- **Statistics**: Min: {stats['min']}, Max: {stats['max']}, Mean: {stats['mean']:.2f}, Median: {stats['median']:.2f}"""
                else:
                    doc_text += f"""
- **Statistics**: Min: {stats['min']}, Max: {stats['max']}"""
            
            doc_text += "\n\n"
        
        return doc_text
    
    def generate_sample_sql_queries(self, documentation: Dict, all_tables_docs: Dict = None) -> List[str]:
        """Generate exactly 10 sample SQL queries for training Vanna with easy, medium, hard difficulty levels."""
        table_name = documentation['table_name']
        columns = list(documentation['columns'].keys())
        
        # Get column types for intelligent query generation
        numeric_cols = [col for col, info in documentation['columns'].items() 
                       if 'INTEGER' in info['data_type'] or 'REAL' in info['data_type']]
        text_cols = [col for col, info in documentation['columns'].items() 
                    if 'VARCHAR' in info['data_type'] or 'TEXT' in info['data_type']]
        date_cols = [col for col, info in documentation['columns'].items() 
                    if 'DATE' in info['data_type']]
        
        sample_queries = []
        
        # EASY LEVEL (3 queries) - Basic operations
        sample_queries.extend([
            f"-- EASY: Get total count of records\nSELECT COUNT(*) FROM {table_name};",
            f"-- EASY: Show first 10 records\nSELECT * FROM {table_name} LIMIT 10;",
            f"-- EASY: Get specific columns\nSELECT {', '.join(columns[:3])} FROM {table_name} LIMIT 5;"
        ])
        
        # MEDIUM LEVEL (4 queries) - Aggregations and filtering
        if numeric_cols:
            sample_queries.append(f"-- MEDIUM: Statistics for numeric column\nSELECT MIN({numeric_cols[0]}), MAX({numeric_cols[0]}), AVG({numeric_cols[0]}), COUNT({numeric_cols[0]}) FROM {table_name};")
        else:
            sample_queries.append(f"-- MEDIUM: Count non-null values\nSELECT COUNT({columns[0]}) FROM {table_name} WHERE {columns[0]} IS NOT NULL;")
        
        if text_cols:
            sample_queries.append(f"-- MEDIUM: Group by categorical column\nSELECT {text_cols[0]}, COUNT(*) as count FROM {table_name} GROUP BY {text_cols[0]} ORDER BY count DESC LIMIT 10;")
        else:
            sample_queries.append(f"-- MEDIUM: Group by first column\nSELECT {columns[0]}, COUNT(*) FROM {table_name} GROUP BY {columns[0]} ORDER BY COUNT(*) DESC LIMIT 5;")
        
        if date_cols:
            sample_queries.append(f"-- MEDIUM: Date range analysis\nSELECT MIN({date_cols[0]}) as earliest, MAX({date_cols[0]}) as latest FROM {table_name};")
            sample_queries.append(f"-- MEDIUM: Records by year\nSELECT strftime('%Y', {date_cols[0]}) as year, COUNT(*) FROM {table_name} GROUP BY year ORDER BY year;")
        else:
            sample_queries.extend([
                f"-- MEDIUM: Filter with condition\nSELECT * FROM {table_name} WHERE {columns[0]} IS NOT NULL LIMIT 10;",
                f"-- MEDIUM: Distinct values\nSELECT DISTINCT {columns[0]} FROM {table_name} ORDER BY {columns[0]} LIMIT 10;"
            ])
        
        # HARD LEVEL (3 queries) - Complex operations and joins
        if len(numeric_cols) >= 2:
            sample_queries.append(f"-- HARD: Complex aggregation with multiple conditions\nSELECT {text_cols[0] if text_cols else columns[0]}, AVG({numeric_cols[0]}) as avg_val, COUNT(*) as count FROM {table_name} WHERE {numeric_cols[1]} > (SELECT AVG({numeric_cols[1]}) FROM {table_name}) GROUP BY {text_cols[0] if text_cols else columns[0]} HAVING COUNT(*) > 1 ORDER BY avg_val DESC;")
        else:
            sample_queries.append(f"-- HARD: Subquery with aggregation\nSELECT * FROM {table_name} WHERE {columns[0]} IN (SELECT {columns[0]} FROM {table_name} GROUP BY {columns[0]} HAVING COUNT(*) > 1);")
        
        # Try to create join queries if multiple tables exist
        if all_tables_docs and len(all_tables_docs) > 1:
            join_query = self._generate_join_query(table_name, documentation, all_tables_docs)
            if join_query:
                sample_queries.append(join_query)
            else:
                sample_queries.append(f"-- HARD: Window function\nSELECT {columns[0]}, {columns[1] if len(columns) > 1 else columns[0]}, ROW_NUMBER() OVER (ORDER BY {columns[0]}) as row_num FROM {table_name};")
        else:
            sample_queries.append(f"-- HARD: Window function\nSELECT {columns[0]}, {columns[1] if len(columns) > 1 else columns[0]}, ROW_NUMBER() OVER (ORDER BY {columns[0]}) as row_num FROM {table_name};")
        
        if len(sample_queries) < 10:
            # Add one more complex query to reach exactly 10
            if numeric_cols and text_cols:
                sample_queries.append(f"-- HARD: Percentile analysis\nSELECT {text_cols[0]}, {numeric_cols[0]}, NTILE(4) OVER (PARTITION BY {text_cols[0]} ORDER BY {numeric_cols[0]}) as quartile FROM {table_name};")
            else:
                sample_queries.append(f"-- HARD: Complex filtering\nSELECT * FROM {table_name} t1 WHERE EXISTS (SELECT 1 FROM {table_name} t2 WHERE t1.{columns[0]} = t2.{columns[0]} AND t1.rowid != t2.rowid);")
        
        # Ensure exactly 10 queries
        return sample_queries[:10]
    
    def _generate_join_query(self, current_table: str, current_docs: Dict, all_tables_docs: Dict) -> str:
        """Generate a JOIN query between tables if possible."""
        current_columns = set(current_docs['columns'].keys())
        
        # Look for potential join columns (common column names)
        for other_table, other_docs in all_tables_docs.items():
            if other_table == current_table:
                continue
                
            other_columns = set(other_docs['columns'].keys())
            common_columns = current_columns.intersection(other_columns)
            
            if common_columns:
                join_col = list(common_columns)[0]  # Use first common column
                current_cols = list(current_docs['columns'].keys())[:2]
                other_cols = list(other_docs['columns'].keys())[:2]
                
                return f"-- HARD: Join between tables\nSELECT a.{current_cols[0]}, a.{current_cols[1] if len(current_cols) > 1 else current_cols[0]}, b.{other_cols[0]}, b.{other_cols[1] if len(other_cols) > 1 else other_cols[0]} FROM {current_table} a INNER JOIN {other_table} b ON a.{join_col} = b.{join_col} LIMIT 10;"
        
        return None
    
    def export_documentation_to_file(self, documentation: Dict, file_path: str = None) -> str:
        """Export documentation to a text file."""
        if file_path is None:
            # Create file in project directory
            project_dir = os.getcwd()
            # Create a 'generated_files' subdirectory
            output_dir = os.path.join(project_dir, "generated_files")
            os.makedirs(output_dir, exist_ok=True)
            file_path = os.path.join(output_dir, f"{documentation['table_name']}_documentation.txt")
        
        doc_text = self.format_documentation_for_vanna(documentation)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(doc_text)
            return file_path
        except Exception as e:
            st.error(f"Error exporting documentation: {str(e)}")
            return None
    
    def export_sql_queries_to_file(self, queries: List[str], table_name: str, file_path: str = None) -> str:
        """Export sample SQL queries to a text file."""
        if file_path is None:
            # Create file in project directory
            project_dir = os.getcwd()
            # Create a 'generated_files' subdirectory
            output_dir = os.path.join(project_dir, "generated_files")
            os.makedirs(output_dir, exist_ok=True)
            file_path = os.path.join(output_dir, f"{table_name}_sample_queries.sql")
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"-- Sample SQL Queries for {table_name}\n")
                f.write(f"-- Generated on: {pd.Timestamp.now()}\n\n")
                
                for i, query in enumerate(queries, 1):
                    f.write(f"-- Query {i}\n{query}\n\n")
            
            return file_path
        except Exception as e:
            st.error(f"Error exporting SQL queries: {str(e)}")
            return None
    
    def export_ddl_to_file(self, ddl: str, table_name: str, file_path: str = None) -> str:
        """Export DDL to a SQL file."""
        if file_path is None:
            # Create file in project directory
            project_dir = os.getcwd()
            # Create a 'generated_files' subdirectory
            output_dir = os.path.join(project_dir, "generated_files")
            os.makedirs(output_dir, exist_ok=True)
            file_path = os.path.join(output_dir, f"{table_name}_ddl.sql")
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"-- DDL for {table_name}\n")
                f.write(f"-- Generated on: {pd.Timestamp.now()}\n\n")
                f.write(ddl)
                f.write("\n")
            
            return file_path
        except Exception as e:
            st.error(f"Error exporting DDL: {str(e)}")
            return None

def create_excel_upload_page():
    """Create the Excel upload page interface."""
    st.title("📊 Excel Data Upload & Training")
    st.markdown("Upload Excel files to automatically generate DDL, documentation, and train Vanna AI on your data.")
    
    # No predefined documentation - will use only uploaded Excel data
    
    processor = ExcelProcessor()
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose an Excel file",
        type=['xlsx', 'xls'],
        help="Upload .xlsx or .xls files to analyze and generate database schema"
    )
    
    if uploaded_file is not None:
        # Load and display basic info
        with st.spinner("Loading Excel file..."):
            excel_data = processor.load_excel_file(uploaded_file)
        
        if excel_data is not None:
            sheet_names = list(excel_data.keys())
            st.success(f"✅ File loaded successfully! Found {len(sheet_names)} sheet(s): {', '.join(sheet_names)}")
            
            # Sheet selection and preview
            st.subheader("📊 Sheet Selection & Preview")
            
            # Initialize session state for sheets data
            if 'sheets_data' not in st.session_state:
                st.session_state['sheets_data'] = {}
            if 'generated_ddls' not in st.session_state:
                st.session_state['generated_ddls'] = {}
            if 'generated_docs_dict' not in st.session_state:
                st.session_state['generated_docs_dict'] = {}
            
            # Show sheet tabs for preview
            selected_sheets = st.multiselect(
                "Select sheets to process",
                options=sheet_names,
                default=sheet_names,
                help="Select which sheets you want to generate DDL and documentation for"
            )
            
            if selected_sheets:
                # Preview selected sheets
                for sheet_name in selected_sheets:
                    df = excel_data[sheet_name]
                    with st.expander(f"📋 {sheet_name} Preview ({df.shape[0]} rows × {df.shape[1]} columns)", expanded=False):
                        st.dataframe(df.head(10))
                
                # Configuration section
                st.subheader("⚙️ Configuration")
                
                auto_detect_types = st.checkbox(
                    "Auto-detect data types",
                    value=True,
                    help="Automatically detect appropriate SQL data types"
                )
                
                use_context_docs = st.checkbox(
                    "Use predefined documentation context",
                    value=True,
                    help="Use predefined table descriptions and column documentation"
                )
                
                # Generate DDL and Documentation for all selected sheets
                if st.button("🚀 Generate DDL & Documentation for All Sheets", type="primary"):
                    with st.spinner("Generating DDL and documentation for all sheets..."):
                        all_ddls = []
                        all_docs = []
                        db_path = None
                        
                        for sheet_name in selected_sheets:
                            df = excel_data[sheet_name]
                            
                            # Clean sheet name for table name
                            table_name = processor.clean_column_name(sheet_name)
                            
                            # Analyze data types
                            if auto_detect_types:
                                data_types = processor.analyze_data_types(df)
                            else:
                                data_types = {col: "TEXT" for col in df.columns}
                            
                            # Generate DDL
                            ddl = processor.generate_ddl(df, table_name, data_types)
                            all_ddls.append(ddl)
                            
                            # Generate documentation
                            documentation = processor.generate_documentation(df, table_name)
                            all_docs.append(documentation)
                            
                            # Store in session state
                            st.session_state['sheets_data'][sheet_name] = df
                            st.session_state['generated_ddls'][sheet_name] = ddl
                            st.session_state['generated_docs_dict'][sheet_name] = documentation
                        
                        # Create combined database with all sheets
                        if selected_sheets:
                            db_path = processor.create_combined_database(st.session_state['sheets_data'])
                            if db_path:
                                st.session_state['db_path'] = db_path
                                st.session_state['all_ddls'] = all_ddls
                                st.session_state['all_docs'] = all_docs
                                
                                # Generate and export documentation files
                                st.session_state['doc_files'] = {}
                                st.session_state['sql_files'] = {}
                                st.session_state['ddl_files'] = {}
                                
                                for sheet_name in selected_sheets:
                                    if sheet_name in st.session_state['generated_docs_dict']:
                                        docs = st.session_state['generated_docs_dict'][sheet_name]
                                        ddl = st.session_state['generated_ddls'][sheet_name]
                                        
                                        # Export DDL to SQL file
                                        ddl_file = processor.export_ddl_to_file(ddl, docs['table_name'])
                                        if ddl_file:
                                            st.session_state['ddl_files'][sheet_name] = ddl_file
                                        
                                        # Export documentation to text file
                                        doc_file = processor.export_documentation_to_file(docs)
                                        if doc_file:
                                            st.session_state['doc_files'][sheet_name] = doc_file
                                        
                                        # Generate and export sample SQL queries
                                        sample_queries = processor.generate_sample_sql_queries(docs)
                                        sql_file = processor.export_sql_queries_to_file(sample_queries, docs['table_name'])
                                        if sql_file:
                                            st.session_state['sql_files'][sheet_name] = sql_file
                                
                                st.success(f"✅ DDL, documentation, and sample queries generated successfully for {len(selected_sheets)} sheet(s)!")
                
                # Display results
                if 'generated_ddls' in st.session_state and st.session_state['generated_ddls']:
                    st.subheader("📝 Generated DDLs")
                    
                    # Show DDL for each sheet
                    for sheet_name in selected_sheets:
                        if sheet_name in st.session_state['generated_ddls']:
                            with st.expander(f"🗄️ {sheet_name} DDL", expanded=False):
                                st.code(st.session_state['generated_ddls'][sheet_name], language='sql')
                    
                    # File Downloads Section
                    st.subheader("📁 Download Files")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**🗄️ DDL Files**")
                        if 'ddl_files' in st.session_state:
                            for sheet_name, ddl_file in st.session_state['ddl_files'].items():
                                if os.path.exists(ddl_file):
                                    with open(ddl_file, 'r', encoding='utf-8') as f:
                                        ddl_content = f.read()
                                    
                                    st.download_button(
                                        label=f"🗄️ {sheet_name}_ddl.sql",
                                        data=ddl_content,
                                        file_name=f"{sheet_name}_ddl.sql",
                                        mime="text/plain"
                                    )
                    
                    with col2:
                        st.markdown("**📄 Documentation Files**")
                        if 'doc_files' in st.session_state:
                            for sheet_name, doc_file in st.session_state['doc_files'].items():
                                if os.path.exists(doc_file):
                                    with open(doc_file, 'r', encoding='utf-8') as f:
                                        doc_content = f.read()
                                    
                                    st.download_button(
                                        label=f"📄 {sheet_name}_documentation.txt",
                                        data=doc_content,
                                        file_name=f"{sheet_name}_documentation.txt",
                                        mime="text/plain"
                                    )
                    
                    with col3:
                        st.markdown("**🗂️ Sample SQL Queries**")
                        if 'sql_files' in st.session_state:
                            for sheet_name, sql_file in st.session_state['sql_files'].items():
                                if os.path.exists(sql_file):
                                    with open(sql_file, 'r', encoding='utf-8') as f:
                                        sql_content = f.read()
                                    
                                    st.download_button(
                                        label=f"🗂️ {sheet_name}_queries.sql",
                                        data=sql_content,
                                        file_name=f"{sheet_name}_sample_queries.sql",
                                        mime="text/plain"
                                    )
                    
                    # Sample Queries Preview
                    st.subheader("🔍 Sample SQL Queries Preview")
                    
                    for sheet_name in selected_sheets:
                        if sheet_name in st.session_state['generated_docs_dict']:
                            docs = st.session_state['generated_docs_dict'][sheet_name]
                            sample_queries = processor.generate_sample_sql_queries(docs)
                            
                            with st.expander(f"🔍 {sheet_name} Sample Queries", expanded=False):
                                for i, query in enumerate(sample_queries[:10], 1):  # Show first 10 queries
                                    st.code(query, language='sql')
                    
                    st.subheader("📖 Table Documentation")
                    
                    # Show documentation for each sheet
                    for sheet_name in selected_sheets:
                        if sheet_name in st.session_state['generated_docs_dict']:
                            docs = st.session_state['generated_docs_dict'][sheet_name]
                            
                            with st.expander(f"📊 {sheet_name} Documentation", expanded=False):
                                # Table overview
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Total Rows", f"{docs['total_rows']:,}")
                                with col2:
                                    st.metric("Total Columns", docs['total_columns'])
                                with col3:
                                    if 'db_path' in st.session_state and os.path.exists(st.session_state['db_path']):
                                        st.metric("Database Size", f"{os.path.getsize(st.session_state['db_path']) / 1024:.1f} KB")
                                
                                # Column details
                                for col_name, col_info in docs['columns'].items():
                                    with st.container():
                                        st.markdown(f"**{col_name}** _{col_info['original_name']}_")
                                        
                                        col1, col2, col3, col4 = st.columns(4)
                                        with col1:
                                            st.metric("Data Type", col_info['data_type'])
                                        with col2:
                                            st.metric("Non-null", f"{col_info['non_null_count']:,}")
                                        with col3:
                                            st.metric("Unique Values", f"{col_info['unique_values']:,}")
                                        with col4:
                                            st.metric("Null %", f"{col_info['null_percentage']:.1f}%")
                                        
                                        if col_info['sample_values']:
                                            st.markdown(f"*Sample values: {', '.join(col_info['sample_values'][:5])}*")
                                        
                                        if 'statistics' in col_info and col_info['statistics']:
                                            stats = col_info['statistics']
                                            if stats['mean'] is not None:
                                                st.markdown(f"*Stats: Min: {stats['min']}, Max: {stats['max']}, Mean: {stats['mean']:.2f}*")
                                            else:
                                                st.markdown(f"*Stats: Min: {stats['min']}, Max: {stats['max']}*")
                                        
                                        st.divider()
                    
                    # Vanna Training Section
                    from train_vanna import create_training_interface, create_training_status_display
                    
                    # Training status display
                    create_training_status_display()
                    
                    # Training interface
                    create_training_interface()
