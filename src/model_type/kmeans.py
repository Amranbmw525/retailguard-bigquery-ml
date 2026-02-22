def kmeans_anomaly_model_create_query(num_clusters=5, full_table_id=None,
                                      full_kmeans_model_id="kmeans_anomaly_model"):
    """Build a query to create a k-means model for anomaly detection.

    Args:
        num_clusters (int): Number of clusters to fit.
        full_table_id (str | None): Fully qualified source table ID.
        full_kmeans_model_id (str): Fully qualified model ID to create.

    Returns:
        str: SQL query that creates or replaces the k-means model.
    """
    return f"""
        CREATE OR REPLACE MODEL `{full_kmeans_model_id}`
        OPTIONS(model_type='kmeans', num_clusters={num_clusters}, standardize_features=TRUE) AS
        SELECT
          transaction_amount,
          product_id,
          store_location,
          CASE WHEN return_reason = 'No Return' THEN 0 ELSE 1 END AS is_return,
          txn_30d,
          returns_30d,
          avg_price_30d
        FROM `{full_table_id}`;
        """


def detect_anomalies_query(contamination=0.023, full_dataset_id=None, full_table_id=None,
                           kmeans_model_id="kmeans_anomaly_model"):
    """Build a query to detect anomalies using ML.DETECT_ANOMALIES.

    Args:
        contamination (float): Expected anomaly proportion between 0 and 1.
        full_dataset_id (str | None): Dataset containing the model.
        full_table_id (str | None): Fully qualified source table ID.
        kmeans_model_id (str): Model ID within the dataset.

    Returns:
        str: SQL query that returns only rows flagged as anomalies.
    """
    return f"""
    SELECT* 
    FROM
        ML.DETECT_ANOMALIES(
            MODEL `{full_dataset_id}.{kmeans_model_id}`,
            STRUCT({contamination} AS contamination),
            (
              SELECT 
                *, 
                CASE WHEN return_reason = 'No Return' THEN 0 ELSE 1 END as is_return 
              FROM `{full_table_id}`
            )
          )
    WHERE is_anomaly = TRUE;
    """


def predict_fraud_query(full_dataset_id=None, kmeans_model_id="kmeans_anomaly_model"):
    """Build a query to run ML.PREDICT for k-means clustering output.

    Args:
        full_dataset_id (str | None): Dataset containing the features table.
        kmeans_model_id (str): Model ID within the dataset.

    Returns:
        str: SQL query that predicts clusters and distances.
    """
    return f"""
    SELECT*,
    transaction_id,
    transaction_amount,
    product_id,
    product_name,
    store_location,
    return_reason
    FROM ML.PREDICT(
    MODEL {full_dataset_id}.{kmeans_model_id},
    (SELECT * FROM {full_dataset_id}.features));
    """


def create_table_anomaly_score_query(full_dataset_id=None):
    """Build a query to create a table with anomaly scores from k-means.

    Args:
        full_dataset_id (str | None): Fully qualified dataset identifier.

    Returns:
        str: SQL query to create or replace `{full_dataset_id}.kmeans_scored`.
    """
    return f"""
    CREATE OR REPLACE TABLE `{full_dataset_id}.kmeans_scored` AS
        SELECT
          *,
          nearest_centroids_distance[OFFSET(0)].distance AS anomaly_score
        FROM ML.PREDICT(
          MODEL `{full_dataset_id}.kmeans_anomaly_model`,
          (
            SELECT
              transaction_id,
              transaction_amount,
              product_id,
              product_name,
              product_category,
              store_location,
              return_reason,
              timestamp,
              fraud_flag,
              txn_30d,
              returns_30d,
              avg_price_30d,
              CASE WHEN return_reason = 'No Return' THEN 0 ELSE 1 END AS is_return
            FROM `{full_dataset_id}.features`
          )
        );
    """


def create_table_anomaly_query(full_dataset_id=None):
    """Build a query to fetch rows with high anomaly scores.

    The threshold is computed as AVG + 3 * STDDEV of `anomaly_score`.

    Args:
        full_dataset_id (str | None): Fully qualified dataset identifier.

    Returns:
        str: SQL query string to select anomalous entries.
    """
    return f"""
        SELECT*
        FROM `{full_dataset_id}.kmeans_scored`
        WHERE anomaly_score > (
          SELECT AVG(anomaly_score) + 3 * STDDEV(anomaly_score)
          FROM `{full_dataset_id}.kmeans_scored`
        );
    """
