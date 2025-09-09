import pandas as pd
import streamlit as st
from typing import Dict, List
import os
from vanna_calls import setup_vanna
import time

class BatchSQLGenerator:
    """Handles batch SQL generation from Excel files with natural language statements."""
    
    def __init__(self):
        self.vn = None
        
    def setup_vanna_connection(self):
        """Setup Vanna connection for SQL generation."""
        try:
            self.vn = setup_vanna()
            return self.vn is not None
        except Exception as e:
            st.error(f"❌ Error setting up Vanna: {str(e)}")
            return False
    
    def load_statements_file(self, uploaded_file) -> pd.DataFrame:
        """Load Excel file containing natural language statements."""
        try:
            if uploaded_file.name.endswith('.xlsx') or uploaded_file.name.endswith('.xls'):
                engine = 'openpyxl' if uploaded_file.name.endswith('.xlsx') else 'xlrd'
                df = pd.read_excel(uploaded_file, engine=engine)
                return df
            else:
                raise ValueError("Unsupported file format. Please upload .xlsx or .xls files.")
        except Exception as e:
            st.error(f"Error loading Excel file: {str(e)}")
            return None
    
    def validate_statements_file(self, df: pd.DataFrame) -> bool:
        """Validate that the Excel file has the required 'statements' column."""
        if df is None:
            return False
        
        if 'statements' not in df.columns:
            st.error("❌ The Excel file must contain a column named 'statements'")
            st.info("💡 Please ensure your Excel file has a column called 'statements' containing natural language queries.")
            return False
        
        # Check for empty statements
        empty_statements = df['statements'].isna().sum()
        if empty_statements > 0:
            st.warning(f"⚠️ Found {empty_statements} empty statements. These will be skipped.")
        
        return True
    
    def generate_sql_for_statement(self, statement: str) -> str:
        """Generate SQL for a single natural language statement."""
        try:
            if not statement or pd.isna(statement):
                return "-- Empty statement"
            
            sql = self.vn.generate_sql(question=statement.strip())
            
            # Clean the SQL
            if sql:
                sql = sql.strip()
                if sql.startswith('```'):
                    lines = sql.split('\n')
                    cleaned_lines = [line for line in lines if not line.strip().startswith('```')]
                    sql = '\n'.join(cleaned_lines).strip()
                
                if sql.lower().startswith('sqlite'):
                    sql = sql[6:].strip()
                
                return sql
            else:
                return "-- Unable to generate SQL"
                
        except Exception as e:
            return f"-- Error: {str(e)}"
    
    def generate_batch_sql(self, df: pd.DataFrame, overwrite_existing=False, progress_callback=None) -> pd.DataFrame:
        """Generate SQL for statements that don't have SQL yet."""
        if not self.setup_vanna_connection():
            return None
        
        # Create a copy of the DataFrame
        result_df = df.copy()
        
        # Add SQL column if it doesn't exist
        if 'sql' not in result_df.columns:
            result_df['sql'] = ""
        
        # Identify rows that need SQL generation
        if overwrite_existing:
            # Generate for all rows
            rows_to_process = result_df.index.tolist()
        else:
            # Only generate for rows with empty/missing SQL
            rows_to_process = []
            for idx, row in result_df.iterrows():
                sql_value = row.get('sql', '')
                # Check if SQL is empty, NaN, or contains only whitespace/comments
                if (pd.isna(sql_value) or 
                    str(sql_value).strip() == '' or 
                    str(sql_value).strip().startswith('-- Empty') or
                    str(sql_value).strip().startswith('-- Unable') or
                    str(sql_value).strip().startswith('-- Error')):
                    rows_to_process.append(idx)
        
        total_to_process = len(rows_to_process)
        
        if total_to_process == 0:
            st.info("ℹ️ All statements already have SQL generated. Enable 'Overwrite existing SQL' to regenerate.")
            return result_df
        
        st.info(f"🔄 Processing {total_to_process} statements (skipping {len(result_df) - total_to_process} with existing SQL)")
        
        # Generate SQL for rows that need it
        for process_idx, row_idx in enumerate(rows_to_process):
            statement = result_df.at[row_idx, 'statements']
            
            if progress_callback:
                progress_callback(process_idx + 1, total_to_process, statement, row_idx + 1)
            
            # Generate SQL
            sql = self.generate_sql_for_statement(statement)
            result_df.at[row_idx, 'sql'] = sql
            
            # Small delay to avoid overwhelming the API
            time.sleep(0.1)
        
        return result_df
    
    def save_results_to_file(self, df: pd.DataFrame, original_filename: str) -> str:
        """Save the results to a new Excel file."""
        try:
            # Create output filename
            base_name = os.path.splitext(original_filename)[0]
            output_filename = f"{base_name}_with_sql.xlsx"
            output_path = os.path.join(os.getcwd(), "generated_files", output_filename)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save to Excel
            df.to_excel(output_path, index=False, engine='openpyxl')
            
            return output_path
            
        except Exception as e:
            st.error(f"Error saving file: {str(e)}")
            return None


def create_batch_sql_page():
    """Create the batch SQL generation page interface."""
    st.title("🔄 Batch SQL Generation")
    st.markdown("Upload an Excel file with natural language statements to generate SQL queries in bulk.")
    
    generator = BatchSQLGenerator()
    
    # Sample template download
    template_path = os.path.join(os.getcwd(), "generated_files", "sample_statements_template.xlsx")
    if os.path.exists(template_path):
        with open(template_path, 'rb') as template_file:
            st.download_button(
                label="📥 Download Sample Template",
                data=template_file.read(),
                file_name="sample_statements_template.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="Download a sample Excel template to understand the required format"
            )
    
    # Instructions
    with st.expander("📖 How to Use", expanded=True):
        st.markdown("""
        **Steps:**
        1. **Prepare Excel File**: Create an Excel file with a column named `statements`
        2. **Add Statements**: Fill the `statements` column with natural language queries
        3. **Upload File**: Upload your Excel file using the file uploader below
        4. **Generate SQL**: Click the generate button to create SQL for all statements
        5. **Download Results**: Download the updated Excel file with SQL queries
        
        **Example Excel Structure:**
        | statements | other_columns |
        |------------|---------------|
        | How many hotels are in California? | ... |
        | Show me all luxury hotels with more than 200 rooms | ... |
        | What's the average room count by state? | ... |
        """)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose an Excel file with statements",
        type=['xlsx', 'xls'],
        help="Upload .xlsx or .xls files containing a 'statements' column"
    )
    
    if uploaded_file is not None:
        # Load and validate file
        with st.spinner("Loading Excel file..."):
            df = generator.load_statements_file(uploaded_file)
        
        if df is not None and generator.validate_statements_file(df):
            # Display file info
            st.success(f"✅ File loaded successfully!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", len(df))
            with col2:
                st.metric("Total Columns", len(df.columns))
            with col3:
                valid_statements = df['statements'].notna().sum()
                st.metric("Valid Statements", valid_statements)
            
            # Preview data
            st.subheader("📊 Data Preview")
            st.dataframe(df.head(10))
            
            # Show statements and SQL status
            st.subheader("🗣️ Statements & SQL Status")
            
            # Check existing SQL status
            has_sql_column = 'sql' in df.columns
            if has_sql_column:
                existing_sql_count = 0
                for idx, row in df.iterrows():
                    sql_value = row.get('sql', '')
                    if not (pd.isna(sql_value) or 
                            str(sql_value).strip() == '' or 
                            str(sql_value).strip().startswith('-- Empty') or
                            str(sql_value).strip().startswith('-- Unable') or
                            str(sql_value).strip().startswith('-- Error')):
                        existing_sql_count += 1
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Statements with existing SQL", existing_sql_count)
                with col2:
                    st.metric("Statements needing SQL", len(df) - existing_sql_count)
                
                if existing_sql_count > 0:
                    st.info(f"ℹ️ Found {existing_sql_count} statements with existing SQL. These will be skipped unless you enable 'Overwrite existing SQL'.")
            
            # Preview statements
            statements_preview = df['statements'].dropna().head(5).tolist()
            for i, statement in enumerate(statements_preview, 1):
                # Show SQL status for this statement
                if has_sql_column and i <= len(df):
                    sql_value = df.iloc[i-1].get('sql', '')
                    has_existing_sql = not (pd.isna(sql_value) or 
                                          str(sql_value).strip() == '' or 
                                          str(sql_value).strip().startswith('-- Empty') or
                                          str(sql_value).strip().startswith('-- Unable') or
                                          str(sql_value).strip().startswith('-- Error'))
                    status_icon = "✅" if has_existing_sql else "⚪"
                    st.write(f"{status_icon} {i}. {statement}")
                else:
                    st.write(f"⚪ {i}. {statement}")
            
            if len(df['statements'].dropna()) > 5:
                st.write(f"... and {len(df['statements'].dropna()) - 5} more statements")
            
            # Generation settings
            st.subheader("⚙️ Generation Settings")
            
            col1, col2 = st.columns(2)
            with col1:
                overwrite_existing = st.checkbox(
                    "Overwrite existing SQL column",
                    value=True,
                    help="If checked, will overwrite any existing 'sql' column"
                )
            
            with col2:
                show_progress = st.checkbox(
                    "Show detailed progress",
                    value=True,
                    help="Display progress for each statement being processed"
                )
            
            # Generate SQL button
            if st.button("🚀 Generate SQL for All Statements", type="primary", use_container_width=True):
                if not generator.setup_vanna_connection():
                    st.error("❌ Could not connect to Vanna. Please check your configuration.")
                    st.stop()
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                results_container = st.empty()
                
                def progress_callback(current, total, statement, row_num=None):
                    progress = current / total
                    progress_bar.progress(progress)
                    row_info = f" (Row {row_num})" if row_num else ""
                    status_text.text(f"Processing {current}/{total}{row_info}: {statement[:50]}...")
                    
                    if show_progress:
                        with results_container.container():
                            st.write(f"**{current}/{total}**{row_info}: {statement}")
                
                # Generate SQL
                with st.spinner("Generating SQL queries..."):
                    result_df = generator.generate_batch_sql(
                        df, 
                        overwrite_existing=overwrite_existing,
                        progress_callback=progress_callback if show_progress else None
                    )
                
                if result_df is not None:
                    progress_bar.progress(1.0)
                    status_text.text("✅ SQL generation completed!")
                    
                    # Count how many were actually processed
                    total_processed = 0
                    if overwrite_existing:
                        total_processed = len(result_df)
                    else:
                        # Count rows that needed processing
                        for idx, row in df.iterrows():
                            sql_value = row.get('sql', '')
                            if (pd.isna(sql_value) or 
                                str(sql_value).strip() == '' or 
                                str(sql_value).strip().startswith('-- Empty') or
                                str(sql_value).strip().startswith('-- Unable') or
                                str(sql_value).strip().startswith('-- Error')):
                                total_processed += 1
                    
                    # Display results
                    if total_processed > 0:
                        st.success(f"🎉 Successfully generated SQL for {total_processed} statements!")
                    else:
                        st.info("ℹ️ No new SQL generated - all statements already had valid SQL.")
                    
                    # Results preview
                    st.subheader("📝 Results Preview")
                    
                    # Show sample results
                    preview_df = result_df[['statements', 'sql']].head(5)
                    for idx, row in preview_df.iterrows():
                        with st.expander(f"Statement {idx + 1}: {row['statements'][:50]}..."):
                            st.write("**Natural Language:**")
                            st.write(row['statements'])
                            st.write("**Generated SQL:**")
                            st.code(row['sql'], language='sql')
                    
                    # Save results
                    output_path = generator.save_results_to_file(result_df, uploaded_file.name)
                    
                    if output_path:
                        st.success(f"✅ Results saved to: `{output_path}`")
                        
                        # Download button
                        with open(output_path, 'rb') as file:
                            st.download_button(
                                label="📥 Download Results Excel File",
                                data=file.read(),
                                file_name=os.path.basename(output_path),
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True
                            )
                    
                    # Store results in session state for further use
                    st.session_state['batch_results'] = result_df
                    
                else:
                    st.error("❌ Failed to generate SQL queries. Please try again.")
            
            # Show existing results if available
            if 'batch_results' in st.session_state:
                st.divider()
                st.subheader("📊 Full Results")
                st.dataframe(st.session_state['batch_results'])


if __name__ == "__main__":
    create_batch_sql_page()
