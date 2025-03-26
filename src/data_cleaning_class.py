import logging
import kagglehub
import sys
import subprocess
import os
import pandas as pd


# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s',  stream=sys.stdout  )

class DataCleaner:
  def __init__(self, dataset, output_path):
    self.dataset = dataset
    self.output_path = output_path
  def download_dataset(self):
      try:
          path = kagglehub.dataset_download(self.dataset)
          logging.info("Dataset downloaded successfully!")
          logging.info("Path to dataset files: %s", path)
          result = subprocess.run(["ls", path], capture_output=True, text=True)
          if result.stdout:
                  logging.info("Dataset files:\n%s", result.stdout)
          if result.stderr:
                  logging.error("Error listing files:\n%s", result.stderr)
          return path
      except Exception as e:
          logging.exception("Error downloading dataset: %s", e)

  def load_data(self):
    try:
      # List all CSV files in the directory
      path = self.download_dataset()
      csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]
      if not csv_files:
        logging.warning("No CSV files found in the dataset folder.")
        # print("no files")
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
  @staticmethod
  def get_percentage_of_missing_values(df):
        # Calculate the percentage of missing values in each column
    missing_percentage = (df.isnull().sum() / len(df)) * 100

    missing_df = missing_percentage.reset_index()
    # Rename columns
    missing_df.columns = ['Column', 'Missing Percentage']

    # Format percentages
    missing_df['Missing Percentage'] = missing_df['Missing Percentage'].apply(lambda x: f'{x:.2f}%')
    return missing_df
  @staticmethod
  def remove_duplicates(df, columns, keep='first'):
      """
      Removes duplicates from a DataFrame based on specified columns.

      Parameters:
      df (pd.DataFrame): The input DataFrame.
      columns (str or list): Column(s) to check for duplicates.
      keep (str): 'first' (keep first occurrence) or 'last' (keep last occurrence).

      Returns:
      pd.DataFrame: A DataFrame with duplicates removed.
      """
      before = df.shape[0]
      duplicate_count = df.duplicated(subset=columns, keep=False).sum()

      if duplicate_count > 0:
          print(f"⚠️ Found {duplicate_count} duplicate rows based on {columns}. Removing them...")
      else:
          print(f"✅ No duplicates found in {columns}.")

      df = df.drop_duplicates(subset=columns, keep=keep)
      after = df.shape[0]
      print(f"✅ {before - after} rows removed. New shape: {df.shape}")

      return df
  @staticmethod
  def check_outliers(df):
      """
      Checks for outliers in the numeric columns in the dataframe.

      Parameters:
      df (pd.DataFrame): The input DataFrame.
      """
      print("\n📌 **Outlier Detection (IQR Method) for Numerical Columns**")
      print("=" * 50)

      num_cols = df.select_dtypes(include=['number']).columns
      for col in num_cols:
          Q1 = df[col].quantile(0.25)
          Q3 = df[col].quantile(0.75)
          IQR = Q3 - Q1
          lower_bound = Q1 - 1.5 * IQR
          upper_bound = Q3 + 1.5 * IQR

          outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
          if not outliers.empty:
              print(f"\n🚨 **Outliers detected in '{col}'**")
              print(outliers[[col]])
          else:
              print(f"\n✅ No significant outliers detected in '{col}'.")

  def clean_product_table(self, dfs):
    product_table_translated = dfs['olist_products_dataset'].merge(dfs['product_category_name_translation'], on='product_category_name', how='left')

    # convert the nan values to Null
    product_table_translated = product_table_translated.map(lambda x: None if pd.isna(x) else x)

    # Check for null values in the denormalized product table
    null_values = self.get_percentage_of_missing_values(product_table_translated)
    logging.info("Missing Values in the denormalized product table:\n%s\n", null_values)

    # If the product category is not translated in English use the original product category name
    no_translation = product_table_translated[product_table_translated['product_category_name_english'].isnull()]
    # print(f"**Product categories without translation in English: {no_translation['product_category_name'].unique()}**\n")
    logging.info("**Product categories without translation in English: %s**\n", no_translation['product_category_name'].unique())
    product_table_translated['product_category_name_english'] = product_table_translated['product_category_name_english'].combine_first(product_table_translated['product_category_name'])

    # Replace Null values in the product table with 'Unknown'
    product_table_translated['product_category_name_english'] = product_table_translated['product_category_name_english'].fillna('Unknown')

    # Drop columns that are not needed for the analysis
    cols_to_drop = ['product_category_name', 'product_name_lenght', 'product_description_lenght', 'product_photos_qty', 'product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm']
    product_table_translated.drop(columns=cols_to_drop, inplace=True)
    product_table_translated = self.remove_duplicates(product_table_translated, 'product_id')
    # convert product category name to title to avoid redundancy
    product_table_translated['product_category_name_english'] = product_table_translated['product_category_name_english'].map(lambda x: x.title())
    null_values = self.get_percentage_of_missing_values(product_table_translated)
    # print(f"Check cleaned denormalized Product table for missing values:\n{null_values}")
    logging.info("Check cleaned denormalized Product table for missing values:\n%s", null_values)

    #Save processed product table as parquet

    logging.info("Checking datatypes of the columns:\n%s", product_table_translated.dtypes)
    output_file_path = os.path.join(self.output_path, 'cleaned_product_table.parquet')
    product_table_translated.to_parquet(output_file_path)

    return product_table_translated


  def clean_customer_table(self, dfs):
    customer_table_df = dfs['olist_customers_dataset']

    # convert na to None
    customer_table_df = customer_table_df.map(lambda x: None if pd.isna(x) else x)
    customer_table_df = self.remove_duplicates(customer_table_df, 'customer_id')

    # Check for null values in the customer table
    null_values = self.get_percentage_of_missing_values(customer_table_df)
    logging.info("Percentage of Null values for each column in the Customers table:\n%s\n", null_values)

    # Check datatypes of columns
    logging.info("Checking datatypes of the columns in the customers table:\n%s", customer_table_df.dtypes)

    # Convert the columns to title case to avoid redundancy during analysis
    customer_table_df['customer_city']= customer_table_df['customer_city'].map(lambda x: x.title())
    customer_table_df['customer_state']= customer_table_df['customer_state'].map(lambda x: x.upper())
    output_file_path = os.path.join(self.output_path, 'cleaned_customer_table.parquet')
    customer_table_df.to_parquet(output_file_path)
    return customer_table_df


  def clean_geolocation_table(self, dfs,):
    geolocation_table_df = dfs['olist_geolocation_dataset']
    #convert na to None
    geolocation_table_df = geolocation_table_df.map(lambda x: None if pd.isna(x) else x)

    #check and remove duplicate rows
    geolocation_table_df = self.remove_duplicates(geolocation_table_df, geolocation_table_df.columns)
    # Check for null values in the geolocation table
    null_values = self.get_percentage_of_missing_values(geolocation_table_df)
    logging.info("Percentage of Null values for each column in the Geolocation table:\n%s\n", null_values)

    #Check datatypes of columns
    logging.info("Checking datatypes of the columns in the geolocation table:\n%s", geolocation_table_df.dtypes)

    #convert the columns to uniform case to avoid redundancy during analysis
    geolocation_table_df['geolocation_city']= geolocation_table_df['geolocation_city'].map(lambda x: x.title())
    geolocation_table_df['geolocation_state']= geolocation_table_df['geolocation_state'].map(lambda x: x.upper())
    output_file_path = os.path.join(self.output_path, 'cleaned_geolocation_table.parquet')
    geolocation_table_df.to_parquet(output_file_path)
    return geolocation_table_df


  def clean_order_items_table(self, dfs):
    order_items_df = dfs['olist_order_items_dataset']
    # Convert na to None
    order_items_df = order_items_df.map(lambda x: None if pd.isna(x) else x)
    # Check and remove duplicates
    order_items_df = self.remove_duplicates(order_items_df, order_items_df.columns)

    # Check for null values in the order items table
    null_values = self.get_percentage_of_missing_values(order_items_df)
    print(null_values)
    logging.info("Percentage of Null values for each column in the order items table:\n%s\n", null_values)
    # Check datatypes of columns
    logging.info("Checking datatypes of the columns in the customers table:\n%s", order_items_df.dtypes)
    print(order_items_df.dtypes)
    # Convert the shipping limit date to timestamp
    order_items_df['shipping_limit_date'] = pd.to_datetime(order_items_df['shipping_limit_date'])
    output_file_path = os.path.join(self.output_path, 'cleaned_order_items_table.parquet')
    order_items_df.to_parquet(output_file_path)
    return order_items_df


  def clean_order_payments(self, dfs):
    order_payments_df = dfs['olist_order_payments_dataset']
    # Convert na to None
    order_payments_df = order_payments_df.map(lambda x: None if pd.isna(x) else x)
    # Check and remove duplicates
    order_payments_df = self.remove_duplicates(order_payments_df, order_payments_df.columns)
    # Check for null values in the order payments table
    null_values = self.get_percentage_of_missing_values(order_payments_df)
    print(null_values)
    logging.info("Percentage of Null values for each column in the order payments table:\n%s\n", null_values)
    # Check datatypes of columns
    logging.info("Checking datatypes of the columns in the order payments table:\n%s", order_payments_df.dtypes)
    print(order_payments_df.dtypes)
    # Convert the columns to title case to avoid redundancy during analysis
    order_payments_df['payment_type']= order_payments_df['payment_type'].map(lambda x: x.title())
    output_file_path = os.path.join(self.output_path, 'cleaned_order_payments_table.parquet')
    order_payments_df.to_parquet(output_file_path)
    return order_payments_df


  def clean_reviews(self, dfs):
    reviews_df = dfs['olist_order_reviews_dataset']
    # Convert na to None
    reviews_df = reviews_df.map(lambda x: None if pd.isna(x) else x)
    # Check and remove duplicates
    reviews_df = self.remove_duplicates(reviews_df,reviews_df.columns)
    # Check for null values in the reviews table
    null_values = self.get_percentage_of_missing_values(reviews_df)
    logging.info("Percentage of Null values for each column in the reviews table:\n%s\n", null_values)
    print(null_values)
    # handle missing values in review comment title and message
    reviews_df['review_comment_title'] = reviews_df['review_comment_title'].fillna('No Title')
    reviews_df['review_comment_message'] = reviews_df['review_comment_message'].fillna('No Message')

    # Check datatypes of Columns
    logging.info("Checking datatypes of the columns in the reviews table:\n%s", reviews_df.dtypes)

    output_file_path = os.path.join(self.output_path, 'cleaned_reviews_table.parquet')
    reviews_df.to_parquet(output_file_path)

    print(reviews_df.dtypes)
    return reviews_df


  def clean_sellers_table(self, dfs):
    sellers_df = dfs['olist_sellers_dataset']
    # Convert na to None
    sellers_df = sellers_df.map(lambda x: None if pd.isna(x) else x)
    # Check and remove duplicates
    sellers_df = self.remove_duplicates(sellers_df, 'seller_id')
    # Check for null values in the sellers table
    null_values = self.get_percentage_of_missing_values(sellers_df)
    logging.info("Percentage of Null values for each column in the sellers table:\n%s\n", null_values)
    print(null_values)
    # Convert the columns to title case to avoid redundancy during analysis
    sellers_df['seller_city']= sellers_df['seller_city'].map(lambda x: x.title())
    sellers_df['seller_state']= sellers_df['seller_state'].map(lambda x: x.upper())
    output_file_path = os.path.join(self.output_path, 'cleaned_sellers_table.parquet')
    sellers_df.to_parquet(output_file_path)
    return sellers_df


  def clean_orders_table(self, dfs):
    orders_df = dfs['olist_orders_dataset']
    # Convert na to None
    orders_df = orders_df.map(lambda x: None if pd.isna(x) else x)
    # Check and remove duplicates
    orders_df = self.remove_duplicates(orders_df, 'order_id')
    # Check datatypes of Columns
    logging.info("Checking datatypes of the columns in the orders table:\n%s", orders_df.dtypes)
    #convert timestamp column to appropriate datatype
    orders_df['order_purchase_timestamp'] = pd.to_datetime(orders_df['order_purchase_timestamp'])
    orders_df['order_approved_at'] = pd.to_datetime(orders_df['order_approved_at'])
    orders_df['order_delivered_carrier_date'] = pd.to_datetime(orders_df['order_delivered_carrier_date'])
    orders_df['order_delivered_customer_date'] = pd.to_datetime(orders_df['order_delivered_customer_date'])
    orders_df['order_estimated_delivery_date'] = pd.to_datetime(orders_df['order_estimated_delivery_date'])
    print(orders_df.dtypes)
    # Check for null values in the orders table
    null_values = self.get_percentage_of_missing_values(orders_df)
    logging.info("Percentage of Null values for each column in the orders table:\n%s\n", null_values)
    print(null_values)
    # Fill timestamps with logical estimations
    orders_df['order_approved_at'].fillna(orders_df['order_purchase_timestamp'], inplace=True)

    # Fill 'order_delivered_carrier_date' with median carrier delivery time for the same order_status
    orders_df['order_delivered_carrier_date'] = orders_df.groupby('order_status')['order_delivered_carrier_date'].transform(
        lambda x: x.fillna(x.median())
    )

    # Fill 'order_delivered_customer_date' with median customer delivery time for the same order_status
    orders_df['order_delivered_customer_date'] = orders_df.groupby('order_status')['order_delivered_customer_date'].transform(
        lambda x: x.fillna(x.median())
    )

    null_values = self.get_percentage_of_missing_values(orders_df)
    logging.info("Percentage of Null values for each column in the orders table:\n%s\n", null_values)
    print(null_values)
    output_file_path = os.path.join(self.output_path, 'cleaned_orders_table.parquet')
    orders_df.to_parquet(output_file_path)
  
  
  def clean_all_tables(self):
    dfs = self.load_data()
    # loop through the dataframes and create a csv 
    self.clean_product_table(dfs)
    self.clean_customer_table(dfs)
    self.clean_geolocation_table(dfs)
    self.clean_order_items_table(dfs)
    self.clean_order_payments(dfs)
    self.clean_reviews(dfs)
    self.clean_sellers_table(dfs)
    self.clean_orders_table(dfs)


if __name__ == "__main__":
    dataset = "olistbr/brazilian-ecommerce"
    os.makedirs('data/processed', exist_ok=True)
    output_path = 'data/processed'
    clean_df=DataCleaner(dataset, output_path)
    clean_df.clean_all_tables()

