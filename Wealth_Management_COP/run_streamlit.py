import sys
import streamlit.web.cli as stcli

if __name__ == "__main__":
    # We pass the arguments exactly as if we ran `streamlit run ...` from the terminal
    sys.argv = ["streamlit", "run", "app/streamlit_app.py", "--server.port", "8501"]
    sys.exit(stcli.main())
