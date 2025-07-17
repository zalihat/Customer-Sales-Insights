import os
import kagglehub
import logging
import subprocess
import sys
from pathlib import Path
import pandas as pd



logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s',  stream=sys.stdout  )

class DataIngestion:
  def __init__(self, dataset, output_path):
    self.dataset = dataset
    self.output_path = output_path
    os.makedirs(self.output_path, exist_ok=True)
  def download_dataset(self):
    try:
        path = Path(kagglehub.dataset_download(self.dataset))
        logging.info("Dataset downloaded successfully!")
        logging.info("Path to dataset files: %s", path)
        print(path)
        files = [f for f in path.iterdir()]
        if not files:
            logging.info("Dataset files:\n%s", files)
        else:
            logging.error("Error listing files")
        return path

    except Exception as e:
          logging.exception("Error downloading dataset: %s", e)

  def load_data(self):  
    try:
      # List all CSV files in the directory
      path = self.download_dataset()
      files = [f for f in path.iterdir()]
      csv_files = [f.name for f in files if f.suffix.endswith(".csv")]
      if not csv_files:
        logging.warning("No CSV files found in the dataset folder.")
        return
      logging.info("Found CSV files: %s", ", ".join(csv_files))

      # Dictionary to store DataFrames
      dataframes = {}

        # Loop through each CSV file and create a DataFrame
      for file in csv_files:
        # print(file)
        file_path = os.path.join(path, file)
        df_name = file.replace(".csv", "")  # Remove .csv to create variable-friendly names
        dataframes[df_name] = pd.read_csv(file_path)
        logging.info("Loaded DataFrame: %s (Shape: %s)", df_name, dataframes[df_name].shape)
        # print(f"Loaded DataFrame: {df_name} (Shape: {dataframes[df_name].shape})")

      return dataframes  # Return all DataFrames as a dictionary

    except Exception as e:
      logging.exception("Error processing dataset: %s", e)
      print('exception')
      return None
  def save_ingested_data(self):
     dfs = self.load_data()