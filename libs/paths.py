from pathlib import Path

# Get the project root dynamically
project_root = Path(__file__).resolve().parent.parent

# Define key folder paths
data_folder = project_root / "data"
libs_folder = project_root / "libs"
scripts_folder = project_root / "scripts"
results_folder = project_root / "results"
notebooks_folder = project_root / "notebooks"
models_folder = project_root / "models"  # New models folder

# Ensure necessary directories exist
results_folder.mkdir(parents=True, exist_ok=True)
models_folder.mkdir(parents=True, exist_ok=True)  # Ensure models folder exists
