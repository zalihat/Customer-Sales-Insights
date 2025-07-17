# from preprocess import DataCleaner
dataset = "olistbr/brazilian-ecommerce"
output_folder_path = "data/silver"
# data_cleaner = DataCleaner(dataset, output_folder_path)
# clean_merged_df = data_cleaner.clean_all_tables()
# print(clean_merged_df.head())

from data_ingestion import DataIngestion
ingest_data = DataIngestion(dataset, output_folder_path)
ingest_data.download_dataset()
dfs = ingest_data.load_data()
print(dfs)