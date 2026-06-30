import streamlit as st
import subprocess
import json
import os

def render_evaluation_view():
    st.title("🧪 Automated Evaluation (LLM-as-a-Judge)")
    st.markdown("Run the automated RAGAS evaluation pipeline against the Golden Set to test the Copilot's resilience against complex prompt injections, compliance traps, and data isolation attempts.")
    
    st.divider()
    
    if st.button("🚀 Start Evaluation"):
        st.info("Evaluation started. This may take a few minutes. Please wait...")
        
        # Container for live streaming logs
        output_container = st.empty()
        log_text = ""
        
        # Run the evaluation script as a subprocess
        process = subprocess.Popen(
            ["python", "run_ragas_eval.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Stream the output line by line
        for line in iter(process.stdout.readline, ''):
            log_text += line
            # Keep only the last 50 lines to prevent UI lag if output is massive
            display_text = "".join(log_text.splitlines(True)[-50:])
            output_container.code(display_text, language="text")
            
        process.stdout.close()
        process.wait()
        
        if process.returncode == 0:
            st.success("Evaluation completed successfully!")
            
            # Load and display the JSON beautifully
            if os.path.exists("ragas_evaluation_results.json"):
                with open("ragas_evaluation_results.json", "r") as f:
                    results = json.load(f)
                    
                st.subheader("📊 Final Metrics")
                col1, col2 = st.columns(2)
                col1.metric("Average Faithfulness", results.get("average_faithfulness_percent", "N/A"))
                col2.metric("Average Relevancy", results.get("average_relevancy_percent", "N/A"))
                
                with st.expander("View Detailed Breakdown"):
                    st.json(results.get("detailed_results", []))
        else:
            st.error("Evaluation failed. Check the logs above.")
