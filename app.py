import time
import streamlit as st
from vanna_calls import (
    generate_questions_cached,
    generate_sql_cached,
    generate_sql_with_context,
    run_sql_cached,
    generate_plotly_code_cached,
    generate_plot_cached,
    generate_followup_cached,
    should_generate_chart_cached,
    is_sql_valid_cached,
    generate_summary_cached,
    clean_sql_formatting
)
from excel_processor import create_excel_upload_page

avatar_url = "https://vanna.ai/img/vanna.svg"

st.set_page_config(layout="wide", page_title="Vanna AI - Data Analysis & Training")

def set_question(question):
    st.session_state["my_question"] = question

def reset_chat():
    """Reset the entire chat session"""
    st.session_state["my_question"] = None
    st.session_state["chat_history"] = []
    st.session_state["generated_sql"] = ""
    st.session_state["df"] = None
    st.session_state["last_sidebar_sql"] = ""

# Navigation
page = st.sidebar.selectbox(
    "Choose a page",
    ["💬 Chat with Data", "📊 Upload Excel Data", "🔄 Batch SQL Generation", "🎯 Train Vanna AI"],
    help="Navigate between chat interface, Excel upload, batch SQL generation, and Vanna training"
)

if page == "📊 Upload Excel Data":
    create_excel_upload_page()
    st.stop()

elif page == "🔄 Batch SQL Generation":
    from batch_sql_generator import create_batch_sql_page
    create_batch_sql_page()
    st.stop()

elif page == "🎯 Train Vanna AI":
    from train_vanna import create_training_interface, create_training_status_display
    
    st.title("🎯 Vanna AI Training Center")
    st.markdown("Advanced training interface for Vanna AI with comprehensive options and monitoring.")
    
    # Training status and interface
    create_training_status_display()
    st.divider()
    create_training_interface()
    st.stop()

# Original chat interface
st.sidebar.title("Output Settings")
st.sidebar.checkbox("Show SQL", value=True, key="show_sql")
st.sidebar.checkbox("Auto-Execute SQL", value=False, key="auto_execute_sql", help="Automatically run SQL queries without clicking the button")
st.sidebar.checkbox("Contextual Q&A", value=True, key="use_context", help="Use conversation history for better contextual understanding")
st.sidebar.divider()
st.sidebar.checkbox("Show Table", value=True, key="show_table")
st.sidebar.checkbox("Show Plotly Code", value=True, key="show_plotly_code")
st.sidebar.checkbox("Show Chart", value=True, key="show_chart")
st.sidebar.checkbox("Show Summary", value=True, key="show_summary")
st.sidebar.checkbox("Show Follow-up Questions", value=True, key="show_followup")
st.sidebar.divider()

# Manual SQL execution
st.sidebar.subheader("🔧 Manual SQL Execution")

# Check if there's new SQL generated
if st.session_state.get("generated_sql") and st.session_state.get("generated_sql") != st.session_state.get("last_sidebar_sql", ""):
    st.sidebar.info("🆕 New SQL generated! Click 'Refresh SQL' to load it.")

# Get the latest generated SQL and clean it
latest_sql = st.session_state.get("generated_sql", "")
latest_sql = clean_sql_formatting(latest_sql)

manual_sql = st.sidebar.text_area(
    "Enter SQL Query:",
    value=latest_sql,
    height=100,
    help="Enter or modify SQL query to execute manually",
    key="manual_sql_input"
)

col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("🔄 Refresh SQL", help="Load the latest generated SQL"):
        if st.session_state.get("generated_sql"):
            # Mark as loaded in sidebar
            st.session_state["last_sidebar_sql"] = st.session_state["generated_sql"]
            st.rerun()
        
with col2:
    execute_manual_sql = st.button("🚀 Execute SQL")

if execute_manual_sql:
    if manual_sql.strip():
        try:
            df_manual = run_sql_cached(sql=manual_sql)
            if df_manual is not None:
                st.session_state["df"] = df_manual
                st.session_state["manual_execution"] = True
                # Add manual execution to chat history
                st.session_state["chat_history"].append({
                    "type": "assistant",
                    "content": f"Manual SQL Execution: {manual_sql}",
                    "is_sql": True
                })
                st.session_state["chat_history"].append({
                    "type": "results",
                    "df": df_manual.copy(),
                    "question": "Manual SQL Execution"
                })
                st.sidebar.success(f"✅ Query executed! Found {len(df_manual)} rows.")
            else:
                st.sidebar.error("❌ Query returned no results")
        except Exception as e:
            st.sidebar.error(f"❌ Error: {str(e)}")
    else:
        st.sidebar.warning("⚠️ Please enter a SQL query")

st.sidebar.button("Reset Chat", on_click=reset_chat, use_container_width=True)

st.title("💬 Vanna AI - Chat with Your Data")
# st.sidebar.write(st.session_state)


assistant_message_suggested = st.chat_message(
    "assistant", avatar=avatar_url
)
if assistant_message_suggested.button("Click to show suggested questions"):
    st.session_state["my_question"] = None
    questions = generate_questions_cached()
    for i, question in enumerate(questions):
        time.sleep(0.05)
        button = st.button(
            question,
            on_click=set_question,
            args=(question,),
            key=f"suggested_question_{i}"
        )

my_question = st.session_state.get("my_question", default=None)

if my_question is None:
    my_question = st.chat_input(
        "Ask me a question about your data",
    )


# Initialize chat history if it doesn't exist
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Display chat history
for chat_item in st.session_state["chat_history"]:
    if chat_item["type"] == "user":
        user_msg = st.chat_message("user")
        user_msg.write(chat_item["content"])
    elif chat_item["type"] == "assistant":
        assistant_msg = st.chat_message("assistant", avatar=avatar_url)
        if chat_item.get("is_sql"):
            assistant_msg.code(chat_item["content"], language="sql", line_numbers=True)
        else:
            assistant_msg.write(chat_item["content"])
    elif chat_item["type"] == "results":
        # Display results from previous queries
        if chat_item.get("df") is not None:
            st.subheader("📊 Query Results")
            st.dataframe(chat_item["df"])

if my_question:
    # Add user question to chat history
    st.session_state["chat_history"].append({
        "type": "user",
        "content": my_question
    })
    
    # Display the new user message
    user_message = st.chat_message("user")
    user_message.write(f"{my_question}")

    # Generate SQL with or without context
    use_context = st.session_state.get("use_context", True)
    has_history = len(st.session_state.get("chat_history", [])) > 0
    
    if use_context and has_history:
        # Use contextual generation with chat history
        with st.spinner("🧠 Generating SQL with conversation context..."):
            sql = generate_sql_with_context(question=my_question, chat_history=st.session_state["chat_history"])
        # Show context indicator
        st.info(f"🧠 **Contextual Q&A Active**: Using {len([item for item in st.session_state['chat_history'] if item['type'] in ['user', 'assistant', 'results']])} previous conversation items for context.")
    else:
        # Use standard generation without context
        sql = generate_sql_cached(question=my_question)
        if use_context and not has_history:
            st.info("💡 **First Question**: Contextual Q&A will activate after your first query.")
        elif not use_context:
            st.info("⚙️ **Standard Mode**: Contextual Q&A is disabled in sidebar settings.")

    if sql:
        # Store the generated SQL in session state
        st.session_state["generated_sql"] = sql
        
        # Add SQL to chat history
        st.session_state["chat_history"].append({
            "type": "assistant",
            "content": sql,
            "is_sql": True
        })
        
        if is_sql_valid_cached(sql=sql):
            if st.session_state.get("show_sql", True):
                assistant_message_sql = st.chat_message(
                    "assistant", avatar=avatar_url
                )
                assistant_message_sql.code(sql, language="sql", line_numbers=True)
                assistant_message_sql.info("💡 Use the sidebar to execute this SQL query or edit it before running.")
                
                # No inline execution button - use sidebar only
                should_execute = False
        else:
            assistant_message = st.chat_message(
                "assistant", avatar=avatar_url
            )
            assistant_message.write(sql)
            
            # Add error message to chat history
            st.session_state["chat_history"].append({
                "type": "assistant",
                "content": f"Invalid SQL generated: {sql}",
                "is_sql": False
            })
            should_execute = False

        # SQL execution is now handled only through the sidebar
        # No automatic execution from chat interface

    else:
        assistant_message_error = st.chat_message(
            "assistant", avatar=avatar_url
        )
        assistant_message_error.error("I wasn't able to generate SQL for that question")
        
        # Add error to chat history
        st.session_state["chat_history"].append({
            "type": "assistant",
            "content": "I wasn't able to generate SQL for that question",
            "is_sql": False
        })
    
    # Clear the current question to allow new input
    st.session_state["my_question"] = None
