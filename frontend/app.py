import os
import streamlit as st
import pandas as pd
from io import BytesIO
import requests
import json

# Prefer the modern graph_objects import, fallback to graph_objs for older installs
try:
    import plotly.graph_objects as go
except Exception:
    import plotly.graph_objs as go

import plotly.express as px


def _ensure_running_under_streamlit():
    """If not already running under Streamlit, re-launch with `streamlit run`.

    This avoids the repeated "missing ScriptRunContext" warnings when users
    accidentally run `python app.py` instead of `streamlit run app.py`.
    """
    try:
        # Newer Streamlit exposes get_script_run_ctx in runtime.scriptrunner
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is not None:
            return
    except Exception:
        # If import fails or context is None, assume not running under streamlit
        pass

    # If we're in an interactive session that isn't streamlit, spawn streamlit
    # and exit the current process so the Streamlit runtime handles the app.
    import sys
    import subprocess
    script = os.path.abspath(__file__)
    # Avoid infinite relaunch loops by checking an env var
    if os.environ.get("_PREDICTIFLOW_LAUNCHED_WITH_STREAMLIT") != "1":
        print("Re-launching with: streamlit run", script)
        env = os.environ.copy()
        env["_PREDICTIFLOW_LAUNCHED_WITH_STREAMLIT"] = "1"
        subprocess.Popen([sys.executable, "-m", "streamlit", "run", script], env=env)
        sys.exit(0)


# Ensure we're running under streamlit runtime (or re-launch)
_ensure_running_under_streamlit()

# Page config
st.set_page_config(
    page_title="PredictiFlow",
    page_icon="🔮",
    layout="wide"
)

# Title
st.title("🔮 PredictiFlow")
st.subheader("Automated Time Series Analysis & Forecasting")

# File upload
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
tabs = st.tabs(["Forecast", "Models"]) 

# Forecast tab
with tabs[0]:
    uploaded_file = st.file_uploader("Upload your time series data (CSV)", type="csv")
    periods = st.number_input('Forecast horizon (periods)', min_value=1, max_value=365, value=30, step=1)
    mode = st.radio('Mode', options=['Synchronous', 'Asynchronous (background job)'])

    if uploaded_file is not None:
        try:
            raw = uploaded_file.getvalue()
            df = pd.read_csv(BytesIO(raw))
            st.subheader("📊 Data Preview")
            st.dataframe(df.head())

            if st.button("Generate Forecast"):
                if df.shape[1] < 2:
                    st.error("CSV must contain at least two columns: date and value.")
                else:
                    with st.spinner("Analyzing your data..."):
                        try:
                            files = {"file": ("data.csv", raw, "text/csv")}
                            if mode == 'Synchronous':
                                response = requests.post(f"{BACKEND_URL}/predict?periods={int(periods)}", files=files, timeout=120)
                                response.raise_for_status()
                                results = response.json()
                            else:
                                response = requests.post(f"{BACKEND_URL}/predict-async?periods={int(periods)}", files=files, timeout=30)
                                response.raise_for_status()
                                job = response.json()
                                job_id = job.get('job_id')
                                st.success(f"Job submitted (id={job_id}). Polling status...")
                                status = None
                                import time
                                for _ in range(60):
                                    r = requests.get(f"{BACKEND_URL}/jobs/{job_id}")
                                    if r.status_code != 200:
                                        st.error('Failed to fetch job status')
                                        status = 'error'
                                        break
                                    j = r.json()
                                    status = j.get('status')
                                    if status == 'done':
                                        results = j.get('result')
                                        break
                                    if status == 'error':
                                        st.error(f"Job error: {j.get('error')}")
                                        results = None
                                        break
                                    st.info(f"Job {job_id} status: {status}")
                                    time.sleep(2)
                                else:
                                    st.warning('Job timed out while polling. You can check /jobs/{id} later in the Jobs tab.')
                                    results = None

                            # Plot results if present
                            if results:
                                st.subheader("🎯 Forecast Results")
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(x=results['historical_data']['timestamp'], y=results['historical_data']['value'], name='Historical Data', line=dict(color='blue')))
                                fig.add_trace(go.Scatter(x=results['forecast_data']['timestamp'], y=results['forecast_data']['value'], name='Forecast', line=dict(color='red', dash='dash')))
                                fig.update_layout(title='Time Series Forecast', xaxis_title='Date', yaxis_title='Value', hovermode='x unified')
                                st.plotly_chart(fig, use_container_width=True)

                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Best Model", results['best_model'].upper())
                                with col2:
                                    mae = results.get('metrics', {}).get('mae')
                                    st.metric("MAE", f"{mae:.2f}" if mae is not None else "N/A")
                                with col3:
                                    rmse = results.get('metrics', {}).get('rmse')
                                    st.metric("RMSE", f"{rmse:.2f}" if rmse is not None else "N/A")

                                # Decomposition
                                st.subheader("📈 Time Series Decomposition")
                                fig_decomp = go.Figure()
                                components = ['trend', 'seasonal', 'residual']
                                colors = ['blue', 'green', 'red']
                                for component, color in zip(components, colors):
                                    vals = results['decomposition'].get(component, [])
                                    fig_decomp.add_trace(go.Scatter(y=vals, name=component.capitalize(), line=dict(color=color)))
                                fig_decomp.update_layout(title='Time Series Components', xaxis_title='Time', yaxis_title='Value', height=800)
                                st.plotly_chart(fig_decomp, use_container_width=True)

                                csv_buf = BytesIO()
                                out_df = pd.DataFrame({'ds': results['forecast_data']['timestamp'], 'yhat': results['forecast_data']['value']})
                                out_df.to_csv(csv_buf, index=False)
                                csv_buf.seek(0)
                                st.download_button("Download forecast CSV", data=csv_buf, file_name="forecast.csv", mime="text/csv")

                        except requests.exceptions.RequestException as err:
                            st.error(f"Request error: {err}")
                            st.stop()
        except Exception as e:
            st.error(f"Error: {str(e)}")


# Models tab
with tabs[1]:
    st.header('Model Registry')
    st.markdown('List and upload model artifacts to the backend model registry.')

    try:
        r = requests.get(f"{BACKEND_URL}/models")
        models = r.json() if r.status_code == 200 else []
    except Exception:
        models = []

    if models:
        st.write('Registered models:')
        for m in models:
            cols = st.columns([1,3,2,1])
            cols[0].write(m['id'])
            cols[1].write(m['name'])
            cols[2].write(m['uploaded_at'])
            cols[3].markdown(f"[download]({BACKEND_URL}/models/{m['id']}/download)")
    else:
        st.info('No models registered yet.')

    st.markdown('---')
    st.subheader('Upload a model artifact')
    model_file = st.file_uploader('Choose a model file (pickle, tar.gz, etc)', type=None, key='model_upload')
    model_name = st.text_input('Model name (optional)')
    if st.button('Upload model') and model_file is not None:
        try:
            files = {'file': (model_file.name, model_file.getvalue(), 'application/octet-stream')}
            params = {'name': model_name} if model_name else {}
            r = requests.post(f"{BACKEND_URL}/models", files=files, params=params)
            if r.status_code == 200:
                st.success('Model uploaded')
            else:
                st.error(f'Upload failed: {r.status_code} {r.text}')
        except Exception as e:
            st.error(f'Upload error: {e}')

# sample
df = pd.DataFrame({'ds': pd.date_range('2024-01-01', periods=10), 'y': range(10)})

fig = px.line(df, x='ds', y='y', title='Sample')
st.plotly_chart(fig)

# Sidebar info
with st.sidebar:
    st.header('Jobs')
    st.markdown('Open the Jobs tab in the main UI to list past async submissions.')
    st.header("ℹ️ About")
    st.markdown("""
    **PredictiFlow** is an automated time series analysis and forecasting platform.
    
    Upload your CSV file with:
    - First column: Date/Timestamp
    - Second column: Values
    
    The platform will:
    1. Analyze your data
    2. Select the best forecasting model
    3. Generate predictions
    4. Visualize results
    """)
    
    st.header("🛠️ Models Used")
    st.markdown("""
    - Prophet (Facebook)
    - SARIMA
    - Exponential Smoothing
    """)
    st.markdown("---")
    st.markdown("Use the sample CSV to test quickly or upload your own. If you see an error about installing packages, try running with Docker as described in README.")