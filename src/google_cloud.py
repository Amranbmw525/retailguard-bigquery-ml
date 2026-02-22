from google.cloud import bigquery
import google.api_core.exceptions

from src.model_type.kmeans import kmeans_anomaly_model_create_query, create_table_anomaly_score_query
from src.model_type.logistic_regression import create_model_log_reg_query, evaluate_logistic_reg_model_query
from src.queries import create_table_features_query, hybrid_detection_create_table_query, fetch_hybrid_detection_query, \
    defect_timeseries_query, store_risk_query

client = bigquery.Client()

dataset_id = 'retail_data'
transactions_table_id = 'transactions'
transactions_table_id_labeled = 'labeled_transactions'
full_dataset_id = f"{client.project}.{dataset_id}"
transactions_table_full_id = f"{client.project}.{dataset_id}.{transactions_table_id}"
full_table_id_labeled = f"{client.project}.{dataset_id}.{transactions_table_id_labeled}"
logistic_reg_model_id = 'logistic_reg_fraud_model'
full_logistic_reg_model_id = f"{client.project}.{dataset_id}.{logistic_reg_model_id}"
kmeans_model_id = 'kmeans_anomaly_model'
full_kmeans_model_id = f"{client.project}.{dataset_id}.{kmeans_model_id}"
features_table_id = 'features'
full_features_table_id = f"{client.project}.{dataset_id}.{features_table_id}"


def run_query(query, job_config=None):
    """Run a BigQuery SQL query and wait for completion.

    Args:
        query (str): SQL to execute.
        job_config (google.cloud.bigquery.job.QueryJobConfig | None): Optional job config.

    Returns:
        google.cloud.bigquery.table.RowIterator: Query results iterator.
    """
    job = client.query(query, job_config=job_config)
    return job.result()


def get_df(query):
    """Run a BigQuery SQL query and return a pandas dataframe.

    Args:
        query (str): SQL to execute.

    Returns:
        pandas.DataFrame: Query results as a dataframe.
    """
    return client.query(query).to_dataframe()


""" Infrastructure / IO """


def ensure_dataset(location='US'):
    """Create the BigQuery dataset if it does not already exist.

    The dataset ID is derived from the module-level `dataset_id` and current
    project configured in the BigQuery client.

    Args:
        location (str): Geographic location for the dataset. Defaults to "US".

    Returns:
        None
    """
    dataset = bigquery.Dataset(full_dataset_id)
    dataset.location = location

    try:
        client.create_dataset(dataset, timeout=30)
        print(f"Created dataset {full_dataset_id}")
    except google.api_core.exceptions.Conflict:
        # This will raise a Conflict exception if the dataset already exists.
        print(f"Dataset {full_dataset_id} already exists.")
    except Exception as e:
        print(f"An error occurred: {e}")


def ensure_transactions_loaded():
    """Ensure the transactions table exists by loading CSV data if needed.

    Returns:
        None
    """
    transactions_file_path = "./data/retail_transactions_simulated.csv"
    try:
        table = client.get_table(transactions_table_full_id)
        print(f"Table exists {table.num_rows} rows and {len(table.schema)} columns to {transactions_table_full_id}")
    except google.api_core.exceptions.NotFound:
        # Configure the load job
        job_config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.CSV,
            skip_leading_rows=1,  # Assuming your CSV has a header row
            autodetect=True,  # This enables schema auto-detection
        )
        # Load the CSV data into BigQuery
        with open(transactions_file_path, "rb") as source_file:
            job = client.load_table_from_file(source_file, transactions_table_full_id, job_config=job_config)
        # Wait for the job to complete
        job.result()
        table = client.get_table(transactions_table_full_id)  # Make an API request to get the table details
        print(
            f"Loaded {table.num_rows} rows and {len(table.schema)} columns to {transactions_table_full_id}"
            f" with auto-detected schema."
        )


""" ML pipeline steps"""


def build_features():
    """Create or replace the features table used for model training."""
    # Source features from the table actually loaded by `ensure_transactions_loaded`.
    run_query(create_table_features_query(full_features_table_id, transactions_table_full_id))


def train_logreg():
    """Create or replace the logistic regression model."""
    run_query(create_model_log_reg_query(full_dataset_id, full_logistic_reg_model_id))


def eval_logreg():
    """Evaluate the logistic regression model with ML.EVALUATE."""
    run_query(evaluate_logistic_reg_model_query(full_logistic_reg_model_id))


def train_kmeans():
    """Create or replace the k-means anomaly model."""
    run_query(
        kmeans_anomaly_model_create_query(full_table_id=f"{full_dataset_id}.features",
                                          full_kmeans_model_id=full_kmeans_model_id))


def score_kmeans():
    """Create or replace the k-means scored table."""
    run_query(create_table_anomaly_score_query(full_dataset_id))


def build_hybrid():
    """Create or replace the hybrid detection table."""
    run_query(hybrid_detection_create_table_query(full_dataset_id, full_logistic_reg_model_id))


def get_df_hybrid_detection():
    """Fetch hybrid detection rows as a dataframe."""
    return get_df(fetch_hybrid_detection_query(full_dataset_id))


def get_df_store_risk():
    """Fetch store risk aggregates as a dataframe."""
    return get_df(store_risk_query(full_dataset_id))


def get_df_defect_timeseries():
    """Fetch defect time-series aggregates as a dataframe."""
    return get_df(defect_timeseries_query(transactions_table_full_id))


def get_df_anomalies():
    """Fetch anomaly-focused fields from the hybrid detection table."""
    query = f"""
    SELECT
      product_id,
      transaction_amount,
      return_reason,
      fraud_flag,
      anomaly_score,
      hybrid_risk_label,
      anomaly_score > 3 AS is_anomaly
    FROM `{full_dataset_id}.hybrid_detection`
    """
    return get_df(query)
