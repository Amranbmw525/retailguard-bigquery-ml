def kmeans_anomaly_model_create_query(num_clusters=5, full_table_id=None,
                                      full_kmeans_model_id="kmeans_anomaly_model"):
    """Creates a new kmeans model in the dataset.Detect unknown / emerging anomalies.
        ✔ Unsupervised
        ✔ No labels required
        ✔ Finds clusters, not fraud probabilities
        ✔ You want anomaly detection
        ✔ You want behavioral segmentation
        ✔ Fraud patterns are unknown
        Output:
         cluster_id
         distance_from_centroid
         :param full_table_id:
         :param full_kmeans_model_id:
         :param num_clusters:
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
    """
    Constructs a SQL query string for detecting anomalies using a machine learning model (KMeans)
    and BigQuery's `ML.DETECT_ANOMALIES` function. This query extracts data, transforms it to include
    a derived field `is_return`, and filters the result to only include rows marked as anomalies.

    :param kmeans_model_id:
    :param full_table_id:
    :param full_dataset_id:
    :param contamination: A float value specifying the proportion of the data expected to be
       anomalous. Must be between 0 and 1.
    :return: A formatted SQL query string for anomaly detection.
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
    """
    Generate a SQL query to Predict Cluster & Distance using a kmeans machine learning model and input features.
    The query selects all columns from the prediction, including the predicted fraud flag and
    probabilities, using a machine learning model in BigQuery to perform the predictions.
    :param full_dataset_id:
    :param kmeans_model_id:
    :return: A formatted SQL query string for fraud prediction.
    :rtype: str
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
    """
    Generates a SQL query for creating or replacing a table including anomaly scores
    calculated using the k-means clustering model applied to transaction data.

    This function constructs a query that applies a model to a dataset to predict
    anomaly scores (`nearest_centroids_distance`) for provided features and stores
    the results in a new table.

    :param full_dataset_id: Fully qualified dataset identifier where the table
        will be created or replaced. If not provided, an error might occur
        during execution.
    :type full_dataset_id: Optional[str]
    :return: A formatted string containing the SQL query for creating or replacing
        the table with anomaly scores.
    :rtype: str
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
    """
    Generates a SQL query string to retrieve rows from the `kmeans_scored` table
    where the anomaly score exceeds a calculated threshold. The threshold is determined
    as the average anomaly score plus three times the standard deviation of the anomaly
    score in the table.
    **Transactions far from cluster centroids = unusual behavior.**
    :param full_dataset_id:
    :return: SQL query string to select anomalous entries
    :rtype: str
    """
    return f"""
        SELECT*
        FROM `{full_dataset_id}.kmeans_scored`
        WHERE anomaly_score > (
          SELECT AVG(anomaly_score) + 3 * STDDEV(anomaly_score)
          FROM `{full_dataset_id}.kmeans_scored`
        );
    """
