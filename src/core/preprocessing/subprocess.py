import subprocess
import os

# Define the script path
script_dir = os.path.dirname(os.path.abspath(__file__))
dest_script_path = os.path.join(script_dir, "dest_preprocessing.py")
user_script_path = os.path.join(script_dir, "user_preprocessing.py")

# Run the script
try:
    subprocess.run(["python", dest_script_path], check=True)
    subprocess.run(["python", user_script_path], check=True)
    print("Script executed successfully.")
except subprocess.CalledProcessError as e:
    print(f"Error occurred while executing the script: {e}")
