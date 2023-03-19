import sys
import os
from streamlit.web import cli as stcli

if __name__ == '__main__':
    paths = os.path.abspath(__file__).split('/')[:-1]
    path = '/'.join(paths)
    sys.argv = ["streamlit", "run", f"{path}/clean_data_gui.py"]
    sys.exit(stcli.main())
