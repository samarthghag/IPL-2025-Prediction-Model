import os
import sys
import logging
import time
import subprocess

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define required packages for each module
MODULE_DEPENDENCIES = {
    "src/models/train_model.py": ["xgboost", "scikit-learn", "pandas", "numpy", "lightgbm", "seaborn", "matplotlib"],
    "src/data/collect_data.py": ["requests", "pandas"],
    "src/data/process_data.py": ["pandas", "numpy"],
    "src/visualization/visualize.py": ["matplotlib", "seaborn", "pandas"]
}

def check_and_install_dependencies():
    """
    Check if all required packages for all modules are installed
    Install any missing packages
    """
    # Get unique packages from all modules
    required_packages = set()
    for packages in MODULE_DEPENDENCIES.values():
        required_packages.update(packages)
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.info(f"Installing missing packages: {', '.join(missing_packages)}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
            logger.info("Successfully installed all missing packages")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install packages: {e}")
            return False
    
    return True

def run_module(module_path, module_name):
    """
    Run a Python module and handle any errors
    """
    logger.info(f"Running {module_name}...")
    start_time = time.time()
    
    try:
        # Get the absolute path to the module
        current_dir = os.path.dirname(os.path.abspath(__file__))
        absolute_module_path = os.path.join(current_dir, module_path)
        
        # Add src to Python path
        src_path = os.path.join(current_dir, 'src')
        if src_path not in sys.path:
            sys.path.append(src_path)
        
        # Use subprocess instead of os.system to better handle paths with spaces
        result = subprocess.run([sys.executable, absolute_module_path], 
                               capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"{module_name} failed with error: {result.stderr}")
            return False
        
        # Log the output
        for line in result.stdout.splitlines():
            logger.info(line)
        
        elapsed_time = time.time() - start_time
        logger.info(f"{module_name} completed successfully in {elapsed_time:.2f} seconds")
        return True
    
    except Exception as e:
        logger.error(f"Error running {module_name}: {str(e)}")
        return False

def main():
    """
    Main function to run the entire prediction pipeline
    """
    logger.info("Starting IPL prediction pipeline")
    
    # Install all dependencies first
    if not check_and_install_dependencies():
        logger.error("Failed to install dependencies, stopping pipeline")
        return
    
    # Step 1: Collect data
    if not run_module("src/data/collect_data.py", "Data Collection"):
        logger.error("Data collection failed, stopping pipeline")
        return
    
    # Step 2: Process data
    if not run_module("src/data/process_data.py", "Data Processing"):
        logger.error("Data processing failed, stopping pipeline")
        return
    
    # Step 3: Train models
    if not run_module("src/models/train_model.py", "Model Training"):
        logger.error("Model training failed, stopping pipeline")
        return
    
    # Step 4: Generate visualizations
    if not run_module("src/visualization/visualize.py", "Visualization Generation"):
        logger.error("Visualization generation failed, stopping pipeline")
        return
    
    logger.info("IPL prediction pipeline completed successfully")
    
    # Display the prediction results
    try:
        import json
        current_dir = os.path.dirname(os.path.abspath(__file__))
        results_file = os.path.join(current_dir, "models", "results", "tournament_summary.json")
        
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            logger.info("=== IPL 2025 Tournament Prediction Results ===")
            logger.info(f"Predicted Champion: {results.get('Champion', 'Not available')}")
            logger.info(f"Predicted Runner-up: {results.get('Runner-up', 'Not available')}")
            
            if 'Playoff Teams' in results:
                logger.info(f"Predicted Playoff Teams: {', '.join(results['Playoff Teams'])}")
            
            logger.info("==============================================")
            logger.info("Full visualization dashboard available in the 'visualizations' directory")
        else:
            logger.warning(f"Results file not found at: {results_file}")
            logger.warning("Please check if the models directory and results file were created correctly")
    
    except Exception as e:
        logger.error(f"Error displaying results: {str(e)}")

if __name__ == "__main__":
    main()