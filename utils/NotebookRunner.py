import papermill as pm
import os

def run_notebook(dir_name, name):
    notebook_path = os.path.join(dir_name, name)
    # Run notebook
    print(f"Running {name}...")
    pm.execute_notebook(
        notebook_path,
        notebook_path.replace(".ipynb", "_output.ipynb")
    )
    print(f"{name} completed.")