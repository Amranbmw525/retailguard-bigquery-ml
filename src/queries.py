def hybrid_detection_create_table_query(full_dataset_id=None, full_logistic_reg_model_id=None):
    """
    Constructs a SQL query to create or replace a table combining predictions from a supervised
    logistic regression model and anomaly scores from an unsupervised k-means clustering model. The
    resulting table includes a hybrid risk label indicating the type of risk (fraud, anomaly, or normal).

    :param full_dataset_id: Fully qualified identifier for the dataset containing the required tables
                            (features, kmeans_scored) in BigQuery.
    :type full_dataset_id: str
    :param full_logistic_reg_model_id: Fully qualified identifier for the logistic regression model
                                       used for supervised prediction.
    :type full_logistic_reg_model_id: str
    :return: A formatted SQL query string to create the hybrid detection table.
    :rtype: str
    """
    return f"""
    CREATE OR REPLACE TABLE `{full_dataset_id}.hybrid_detection` AS
    WITH lr AS (
      SELECT
        transaction_id,
        predicted_fraud_flag,
        predicted_fraud_flag_probs
      FROM ML.PREDICT(
        MODEL `{full_logistic_reg_model_id}`,
        (SELECT * FROM `{full_dataset_id}.features`)
      )
    )
    SELECT
      f.*,
      lr.predicted_fraud_flag,
      lr.predicted_fraud_flag_probs,
      km.anomaly_score,
      CASE
        WHEN lr.predicted_fraud_flag = 1 THEN 'Likely Fraud (Supervised)'
        WHEN km.anomaly_score > 3 THEN 'Anomalous (Unsupervised)'
        ELSE 'Normal'
      END AS hybrid_risk_label,
      CASE
          WHEN predicted_fraud_flag = 1 AND anomaly_score > 4 THEN 'HIGH'
          WHEN predicted_fraud_flag = 1 THEN 'MEDIUM'
          WHEN anomaly_score > 4 THEN 'MEDIUM'
          ELSE 'LOW'
        END AS risk_bucket
    FROM `{full_dataset_id}.features` f
    LEFT JOIN lr
      ON f.transaction_id = lr.transaction_id
    LEFT JOIN `{full_dataset_id}.kmeans_scored` km
      ON f.transaction_id = km.transaction_id;
    """


def fraud_detection_query(full_dataset_id=None):
    """
    Generates a SQL query to retrieve potential fraud detection records based
    on certain conditions such as a predicted fraud flag or anomaly score threshold.

    :param full_dataset_id: The ID of the dataset containing the hybrid_detection
        table. Defaults to None.
    :type full_dataset_id: str or None
    :return: A formatted SQL query string that retrieves records from the
        hybrid_detection table where either the predicted_fraud_flag is 1
        or the anomaly_score is greater than 4.
    :rtype: str
    """
    return f"""
        SELECT *
        FROM `{full_dataset_id}.hybrid_detection`
        WHERE
            predicted_fraud_flag = 1
         OR anomaly_score > 4;
    """


def clustered_fraud_detection_query(full_table_id=None):
    return f"""
    SELECT
      product_id,
      store_location,
      COUNT(*) AS suspicious_returns
    FROM `{full_table_id}` -- labeled_transactions
    WHERE anomaly_type = 'Return goFraud'
    GROUP BY product_id, store_location
    ORDER BY suspicious_returns DESC;
    """


def defect_timeseries_query(retail_transactions_table=None):
    return f"""
    SELECT
      product_id,
      DATE(timestamp) AS date,
      COUNTIF(return_reason = 'Damaged') AS damaged_returns,
      AVG(transaction_amount) AS avg_price,
      COUNT(*) AS total_txns
    FROM `{retail_transactions_table}`
    GROUP BY product_id, date
    ORDER BY product_id, date;
    """


def fetch_anomalies_query(full_table_id=None):
    return f"""SELECT* FROM `{full_table_id}`;"""


def create_table_features_query(full_features_table_id=None, full_table_id_labeled=None):
    return f"""
    CREATE OR REPLACE TABLE `{full_features_table_id}` AS
    SELECT
      product_id,
      transaction_id, 
      product_name, 
      product_category, 
      store_location,
      transaction_amount,
      return_reason,
      timestamp,
    
      -- Behavioral metrics
      COUNT(*) OVER (
        PARTITION BY product_id, store_location
        ORDER BY UNIX_SECONDS(timestamp)
        RANGE BETWEEN 2592000 PRECEDING AND CURRENT ROW
      ) AS txn_30d,
    
      SUM(CASE WHEN return_reason != 'No Return' THEN 1 ELSE 0 END)
        OVER (
          PARTITION BY product_id, store_location
          ORDER BY UNIX_SECONDS(timestamp)
          RANGE BETWEEN 2592000 PRECEDING AND CURRENT ROW
        ) AS returns_30d,
    
      AVG(transaction_amount)
        OVER (
          PARTITION BY product_id
          ORDER BY UNIX_SECONDS(timestamp)
          RANGE BETWEEN 2592000 PRECEDING AND CURRENT ROW
        ) AS avg_price_30d,
      fraud_flag,
      CAST(NULL AS STRING) AS anomaly_type
    FROM `{full_table_id_labeled}`;
"""

def fetch_hybrid_detection_query(full_dataset_id=None):
    return f"""
    SELECT
      transaction_id,
      timestamp,
      product_id,
      product_name,
      product_category,
      store_location,
      transaction_amount,
      return_reason,
      fraud_flag,
      predicted_fraud_flag,
      (
        SELECT prob
        FROM UNNEST(predicted_fraud_flag_probs)
        WHERE label = 1
      ) AS fraud_prob,

      anomaly_score,
      hybrid_risk_label

    FROM `{full_dataset_id}.hybrid_detection`;
    """

def store_risk_query(full_dataset_id=None):
    return f"""
    SELECT
      store_location,
      product_category,
      COUNT(*) AS transactions,
      COUNTIF(return_reason != 'No Return') AS returns,
      SAFE_DIVIDE(COUNTIF(return_reason != 'No Return'), COUNT(*)) AS return_rate,

      AVG((
        SELECT p.prob
        FROM UNNEST(predicted_fraud_flag_probs) AS p
        WHERE p.label = 1
        LIMIT 1
      )) AS avg_fraud_prob,
      AVG(anomaly_score) AS avg_anomaly_score,

      COUNTIF(hybrid_risk_label != 'Normal') AS flagged_txns
    FROM `{full_dataset_id}.hybrid_detection`
    GROUP BY store_location, product_category
    ORDER BY avg_fraud_prob DESC, avg_anomaly_score DESC;
    """
