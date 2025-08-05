import streamlit as st
import pandas as pd
import os
import re
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from langchain_experimental.agents import create_csv_agent
from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set the API key from the environment (manage via st.secrets or .env file)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Unique custom CSS for a modern, soft, and distinctive look
custom_css = """
<style>
body {
    background: linear-gradient(135deg, #f9fafc 0%, #e6ecfa 100%);
    font-family: 'Segoe UI', 'Roboto', sans-serif;
}
h1 {
    font-size: 2.6rem;
    font-weight: bold;
    letter-spacing: -2px;
    background: linear-gradient(90deg, #3a8dde 40%, #27c4f5 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.st-emotion-cache-18ni7ap {
    background: #ffffffcc !important;
    border-radius: 28px !important;
    box-shadow: 0 4px 24px #005fa335, 0 1.5px 6px #b5b8e433;
    margin: 0.8rem 0.8rem 2rem 0.8rem !important;
    padding: 2.2rem 2rem 2.2rem 2rem !important;
}
.stButton>button {
    background: linear-gradient(90deg, #3a8dde 60%, #27c4f5 100%);
    color: #fff;
    font-weight: 600;
    font-size: 1.1rem;
    border: none;
    padding: 0.7em 2em;
    border-radius: 30px;
    box-shadow: 0 2px 8px #b5d1fa33;
    transition: background 0.3s;
}
.stButton>button:hover {
    background: linear-gradient(90deg, #27c4f5 60%, #3a8dde 100%);
    color: #fff;
}
.stTextInput>div>div>input {
    border-radius: 18px;
    background: #f2f6fa;
    font-size: 1.12rem;
    padding: 0.7em 1em;
}
.stTextInput>label {
    font-weight: 500;
    color: #3494e6;
}
.stDataFrame, .stTable {
    background: #f5faff;
    border-radius: 20px;
    margin-bottom: 1.2rem;
    box-shadow: 0 1px 6px #b5b8e433;
}
.stFileUploader>div {
    border-radius: 24px;
    background: #eaf3fc;
    padding: 1.5em 1em;
    box-shadow: 0 2px 12px #a6c3e633;
    border: 1.5px solid #7cc0f7;
}
.st-expander {
    border-radius: 16px !important;
    background: #f7f9fc !important;
}
hr {
    border: none;
    border-top: 1.5px solid #c1d2ef;
    margin: 1.5rem 0;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

def extract_code_from_response(response):
    """Extracts the python code from the response"""
    code = None
    code_match = re.search(r"```(?:python)?\n(.*?)```", response, re.DOTALL)
    if code_match:
        code = code_match.group(1)
    return code

def csv_agent_func(file_path, user_message):
    """Creates an agent that can analyze a csv file and return a response"""
    agent = create_csv_agent(
        ChatOpenAI(temperature=0, model="gpt-4o", openai_api_key=OPENAI_API_KEY),
        file_path, 
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        allow_dangerous_code=True
    )

    try:
        tool_input = {
            "input": {
                "name": "python",
                "arguments": user_message
            }
        }
        response = agent.run(tool_input)
        return response
    except Exception as e:
        st.write(f"Error: {e}")
        return None

def display_content_from_json(json_response):
    """
    Displays the answer in the Streamlit app based on the JSON response.
    """
    if "answer" in json_response:
        st.write(json_response["answer"])

    if "bar" in json_response:
        data = json_response["bar"]
        df = pd.DataFrame(data)
        df.set_index("columns", inplace=True)
        st.bar_chart(df)

    if "table" in json_response:
        data = json_response["table"]
        df = pd.DataFrame(data["data"], columns=data["columns"])
        st.table(df)

def csv_analyzer_app():
    """The streamlit app that allows you to upload a csv file and ask questions about it."""
    st.set_page_config(page_title="Your Data Analyst AI Agent", page_icon="📊", layout="wide")

    # Sidebar: File Upload + Info
    with st.sidebar:
        st.markdown("<h2 style='color:#3a8dde;'>Upload Your CSV</h2>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Select CSV file", type="csv", key="file-uploader-custom")
        st.info(
            "💡 **Tip:** You can upload any CSV — from sales data to grades to finance. "
            "Ask natural questions like 'Show sales trends' or 'What are top categories?'",
            icon="ℹ️"
        )
        st.markdown("---")
        st.markdown(
            "<small style='color:#8a96b8;'>Built with 🤖 <b>LangChain + OpenAI</b> &mdash; by [Your Name]</small>",
            unsafe_allow_html=True
        )
    
    # Main area: Hero section and interaction
    st.markdown(
        """
        <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;margin-top:-2.2rem;">
            <h1>Your Data Analyst AI Agent</h1>
            <h4 style="font-weight:400;color:#2176bd;margin-bottom:2.5rem;">
                Upload a CSV, ask questions in plain English,<br>and get instant analysis, visualizations, and answers!
            </h4>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Query history in session state
    if "history" not in st.session_state:
        st.session_state.history = []
    
    if 'uploaded_file' not in st.session_state:
        st.session_state['uploaded_file'] = None

    # Use uploaded_file from sidebar
    if uploaded_file is not None:
        st.session_state['uploaded_file'] = uploaded_file

    uploaded_file = st.session_state['uploaded_file']
    
    if uploaded_file is not None:
        file_details = {
            "File Name": uploaded_file.name,
            "File Type": uploaded_file.type,
            "Size": uploaded_file.size
        }
        st.markdown(f"<div style='background:#eaf6ff;padding:1rem 1.5rem;border-radius:16px;'><b>File Details:</b> {file_details}</div>", unsafe_allow_html=True)
        
        file_path = os.path.join("/tmp", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        try:
            df = pd.read_csv(file_path)
            st.dataframe(df)
        except Exception as e:
            st.warning(f"Error reading CSV: {e}")
            return
        
        # Place user input & button in a card
        st.markdown("<div style='background:#fff;box-shadow:0 1.5px 8px #7ec5fa1a;padding:1.5rem 2rem 2rem 2rem;border-radius:22px;margin-top:2rem;margin-bottom:2rem;'>", unsafe_allow_html=True)
        user_input = st.text_input("Ask Your Data a Question", key="question-input", help="E.g. 'Show me average sales per month'")
        execute_col, spacer, clear_col = st.columns([1, 0.1, 1])
        if execute_col.button('🔎 Analyze', use_container_width=True):
            with st.spinner("Reasoning..."):
                response = csv_agent_func(file_path, user_input)
            if response is None:
                st.error("No response received.")
                return
            
            # Save query and response
            st.session_state.history.append({"query": user_input, "response": response})
            code_to_execute = extract_code_from_response(response)
            if code_to_execute:
                try:
                    local_vars = {"df": df, "plt": plt}
                    exec(code_to_execute, globals(), local_vars)
                    fig = plt.gcf()  
                    st.pyplot(fig)  
                except Exception as e:
                    st.warning(f"Error during code execution: {e}")
            else:
                st.success(response)
        if clear_col.button("🧹 Clear History", use_container_width=True):
            st.session_state.history = []
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.divider()
    with st.expander("🕓 Query History", expanded=False):
        if st.session_state.history:
            for idx, entry in enumerate(st.session_state.history):
                st.markdown(
                    f"<div style='padding:0.7em 1em 0.7em 1.1em;margin-bottom:0.9em;background:#f7fafd;border-radius:12px;'><b>Query {idx+1}:</b> {entry['query']}<br><span style='color:#578ab5;'><b>Response:</b> {entry['response']}</span></div>",
                    unsafe_allow_html=True
                )
        else:
            st.info("No history available.")

if __name__ == "__main__":
    csv_analyzer_app()