import subprocess
import os

from src.core.exception.bad_request_exceptions import BadReqException

# def sub_processing(survey):
script_dir = os.path.dirname(os.path.abspath(__file__))
dest_script_path = os.path.join(script_dir, "dest_preprocessing.py")

try:
    subprocess.run(["python", dest_script_path], check=True)
    print("Executed")
except subprocess.CalledProcessError as e:
    raise BadReqException("sub processing failure")