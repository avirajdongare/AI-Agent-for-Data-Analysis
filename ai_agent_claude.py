import streamlit as st
import pandas as pd
import os
import re
import tempfile
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from langchain_experimental.agents import create_csv_agent
from langchain_anthropic import ChatAnthropic
from langchain.agents.agent_types import AgentType

#  Load environment variables 
load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY") or st.secrets.get("ANTHROPIC_API_KEY")

#  Custom CSS for clean UI 
custom_css = """
<style>
body {
    background-color: #f0f4f8;
}
h1 {
    color: #2c3e50;
}
.stButton>button {
    background-color: #007acc;
    color: white;
    border: none;
    padding: 10px 24px;
    font-size: 16px;
    border-radius: 5px;
}
.stButton>button:hover {
    background-color: #005fa3;
}
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

#  Helper to extract code blocks 
def extract_code_from_response(response: str):
    match = re.search(r"```(?:python)?\n(.*?)```", response, re.DOTALL)
    return match.group(1) if match else None

#  Claude 3.5 Sonnet CSV Agent Wrapper 
def csv_agent_func(file_path, user_message):
    agent = create_csv_agent(
        ChatAnthropic(
            model="claude-3-5-sonnet-20240620",  # current model name!
            anthropic_api_key=ANTHROPIC_API_KEY,
            temperature=0
        ),
        file_path,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,   # Specially for NON-OPEAI models!
        allow_dangerous_code=True # LangChain requires this opt-in for any agent that can run arbitrary Python code
    )
    try:
        return agent.run(user_message)
    except Exception as e:
        return f"Error: {e}"

#  Main Streamlit App 
def csv_analyzer_app():
    st.title('Claude 3.5 Sonnet: CSV Data Analyst Agent')
    st.write('Upload your CSV file and ask anything! (e.g. "What is the average of column X?", "Plot the distribution of column Y", etc.)')

    # Session state for query/response history and file
    if "history" not in st.session_state:
        st.session_state.history = []
    if "last_file" not in st.session_state:
        st.session_state.last_file = None

    uploaded_file = st.file_uploader("Select a CSV file", type="csv")

    if uploaded_file is not None:
        # Reset history if file changes
        if st.session_state.last_file != uploaded_file.name:
            st.session_state.history = []
            st.session_state.last_file = uploaded_file.name

        st.write({
            "File Name": uploaded_file.name,
            "Type": uploaded_file.type,
            "Size": uploaded_file.size
        })

        # Store uploaded CSV as a temp file for agent use
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(uploaded_file.getbuffer())
            file_path = tmp.name

        try:
            df = pd.read_csv(file_path)
            st.dataframe(df)
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            return

        user_input = st.text_input("Ask your data a question:")
        if st.button('Execute') and user_input.strip():
            with st.spinner("Claude is reasoning..."):
                response = csv_agent_func(file_path, user_input)
            if not response:
                st.write("No response received.")
                return

            st.session_state.history.append({"query": user_input, "response": response})

            code_to_execute = extract_code_from_response(response)
            if code_to_execute:
                try:
                    # Provide df and plt to exec context
                    local_vars = {"df": df, "plt": plt}
                    exec(code_to_execute, {}, local_vars)
                    st.pyplot(plt.gcf())
                    plt.clf()
                except Exception as e:
                    st.error(f"Error during code execution: {e}")
            else:
                st.write(response)

    st.divider()
    with st.expander("Query History"):
        if st.session_state.history:
            for idx, entry in enumerate(st.session_state.history):
                st.write(f"**Query {idx+1}:** {entry['query']}")
                st.write(f"**Response:** {entry['response']}")
                st.write("---")
        else:
            st.write("No history available.")

#  Entrypoint 
if __name__ == "__main__":
    csv_analyzer_app()
