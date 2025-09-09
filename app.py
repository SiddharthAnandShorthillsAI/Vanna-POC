import time
import streamlit as st
from vanna_calls import (
    generate_questions_cached,
    generate_sql_cached,
    run_sql_cached,
    generate_plotly_code_cached,
    generate_plot_cached,
    generate_followup_cached,
    should_generate_chart_cached,
    is_sql_valid_cached,
    generate_summary_cached
)
from excel_processor import create_excel_upload_page

avatar_url = "https://vanna.ai/img/vanna.svg"

st.set_page_config(layout="wide", page_title="Vanna AI - Data Analysis & Training")

# Navigation
page = st.sidebar.selectbox(
    "Choose a page",
    ["💬 Chat with Data", "📊 Upload Excel Data", "🎯 Train Vanna AI"],
    help="Navigate between chat interface, Excel upload functionality, and Vanna training"
)

if page == "📊 Upload Excel Data":
    create_excel_upload_page()
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

# Get the latest generated SQL
latest_sql = st.session_state.get("generated_sql", "")
if latest_sql and latest_sql.startswith('```'):
    # Clean the SQL for display in text area
    lines = latest_sql.split('\n')
    cleaned_lines = [line for line in lines if not line.strip().startswith('```')]
    latest_sql = '\n'.join(cleaned_lines).strip()
    if latest_sql.lower().startswith('sqlite'):
        latest_sql = latest_sql[6:].strip()

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
                st.session_state["my_question"] = None  # Clear question to show input
                st.sidebar.success(f"✅ Query executed! Found {len(df_manual)} rows.")
            else:
                st.sidebar.error("❌ Query returned no results")
        except Exception as e:
            st.sidebar.error(f"❌ Error: {str(e)}")
    else:
        st.sidebar.warning("⚠️ Please enter a SQL query")

st.sidebar.button("Reset", on_click=lambda: set_question(None), use_container_width=True)

st.title("💬 Vanna AI - Chat with Your Data")
# st.sidebar.write(st.session_state)


def set_question(question):
    st.session_state["my_question"] = question


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


if my_question:
    st.session_state["my_question"] = my_question
    user_message = st.chat_message("user")
    user_message.write(f"{my_question}")

    sql = generate_sql_cached(question=my_question)

    if sql:
        # Store the generated SQL in session state
        st.session_state["generated_sql"] = sql
        
        if is_sql_valid_cached(sql=sql):
            if st.session_state.get("show_sql", True):
                assistant_message_sql = st.chat_message(
                    "assistant", avatar=avatar_url
                )
                assistant_message_sql.code(sql, language="sql", line_numbers=True)
                
                # Add manual execution button
                col1, col2 = assistant_message_sql.columns([1, 4])
                with col1:
                    execute_sql = st.button("▶️ Run SQL", key=f"run_sql_{hash(sql)}", help="Click to execute the generated SQL query")
                with col2:
                    st.write("Click the button to execute this SQL query")
                
                # Execute SQL if button is clicked or auto-execute is enabled
                should_execute = execute_sql or st.session_state.get("auto_execute_sql", False)
        else:
            assistant_message = st.chat_message(
                "assistant", avatar=avatar_url
            )
            assistant_message.write(sql)
            should_execute = False

        # Execute SQL only if requested
        if should_execute and sql:
            df = run_sql_cached(sql=sql)
        else:
            df = None

        if df is not None:
            st.session_state["df"] = df
        
        # Clear the question after SQL execution attempt to show input again
        if should_execute:
            st.session_state["my_question"] = None

        if st.session_state.get("df") is not None:
            if st.session_state.get("show_table", True):
                df = st.session_state.get("df")
                assistant_message_table = st.chat_message(
                    "assistant",
                    avatar=avatar_url,
                )
                if len(df) > 10:
                    assistant_message_table.text("First 10 rows of data")
                    assistant_message_table.dataframe(df.head(10))
                else:
                    assistant_message_table.dataframe(df)

            if should_generate_chart_cached(question=my_question, sql=sql, df=df):

                code = generate_plotly_code_cached(question=my_question, sql=sql, df=df)

                if st.session_state.get("show_plotly_code", False):
                    assistant_message_plotly_code = st.chat_message(
                        "assistant",
                        avatar=avatar_url,
                    )
                    assistant_message_plotly_code.code(
                        code, language="python", line_numbers=True
                    )

                if code is not None and code != "":
                    if st.session_state.get("show_chart", True):
                        assistant_message_chart = st.chat_message(
                            "assistant",
                            avatar=avatar_url,
                        )
                        fig = generate_plot_cached(code=code, df=df)
                        if fig is not None:
                            assistant_message_chart.plotly_chart(fig)
                        else:
                            assistant_message_chart.error("I couldn't generate a chart")

            if st.session_state.get("show_summary", True):
                assistant_message_summary = st.chat_message(
                    "assistant",
                    avatar=avatar_url,
                )
                summary = generate_summary_cached(question=my_question, df=df)
                if summary is not None:
                    assistant_message_summary.text(summary)

            if st.session_state.get("show_followup", True):
                assistant_message_followup = st.chat_message(
                    "assistant",
                    avatar=avatar_url,
                )
                followup_questions = generate_followup_cached(
                    question=my_question, sql=sql, df=df
                )
                st.session_state["df"] = None

                if len(followup_questions) > 0:
                    assistant_message_followup.text(
                        "Here are some possible follow-up questions"
                    )
                    # Print the first 5 follow-up questions
                    for i, question in enumerate(followup_questions[:5]):
                        assistant_message_followup.button(question, on_click=set_question, args=(question,), key=f"followup_question_{i}")
                
                # Clear the question after processing to show input again
                st.session_state["my_question"] = None

    else:
        assistant_message_error = st.chat_message(
            "assistant", avatar=avatar_url
        )
        assistant_message_error.error("I wasn't able to generate SQL for that question")
        # Clear the question to show input again
        st.session_state["my_question"] = None
