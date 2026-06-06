import sys
import os

# Suppress some warnings
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
os.environ['STREAMLIT_SERVER_PORT'] = '8501'
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'

from streamlit.web.bootstrap import run

app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
run(app_path, False, [], {})
