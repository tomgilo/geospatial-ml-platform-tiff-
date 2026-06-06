import sys
import traceback
import os

log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "error_log.txt")

try:
    import streamlit
    print("streamlit imported OK")
    print(f"streamlit version: {streamlit.__version__}")
    print(f"streamlit path: {streamlit.__file__}")
except Exception as e:
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"Import streamlit failed:\n{traceback.format_exc()}\n")
    print(f"Error logged to {log_file}")
    sys.exit(1)

try:
    from streamlit.web.bootstrap import run
    print("streamlit.web.bootstrap imported OK")
except Exception as e:
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"Import bootstrap failed:\n{traceback.format_exc()}\n")
    print(f"Error logged to {log_file}")
    sys.exit(1)

print("All imports OK")
