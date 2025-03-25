import logging
import kagglehub
import sys
import subprocess


# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s',  stream=sys.stdout  )


def load_data(dataset):
    try:
        path = kagglehub.dataset_download(dataset)
        logging.info("Dataset downloaded successfully!")
        logging.info("Path to dataset files: %s", path)
        result = subprocess.run(["ls", path], capture_output=True, text=True)
            
            # Print actual file list
        if result.stdout:
                logging.info("📄 Dataset files:\n%s", result.stdout)
        if result.stderr:
                logging.error("⚠️ Error listing files:\n%s", result.stderr)
        return path
    except Exception as e:
        logging.exception("Error downloading dataset: %s", e)
    

dataset = "olistbr/brazilian-ecommerce"
load_data(dataset)
