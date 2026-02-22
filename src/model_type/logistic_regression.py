def create_model_log_reg_query(full_dataset_id=None,full_logistic_reg_model_id=None):
    """Build a query to create a logistic regression model in BigQuery ML.

    Args:
        full_dataset_id (str | None): Fully qualified dataset ID containing `features`.
        full_logistic_reg_model_id (str | None): Fully qualified model ID to create.

    Returns:
        str: SQL query that creates or replaces the logistic regression model.
    """
    return f"""
        CREATE OR REPLACE MODEL `{full_logistic_reg_model_id}`
        OPTIONS (
          model_type = 'logistic_reg',
          input_label_cols = ['fraud_flag']
        ) AS

        SELECT
          txn_30d,
          returns_30d,
          avg_price_30d,
          transaction_amount,
          product_category,
          store_location,
          return_reason,
          fraud_flag
        FROM `{full_dataset_id}.features`;
        """


def evaluate_logistic_reg_model_query(full_logistic_reg_model_id=None):
    """Build a query to evaluate a logistic regression model with ML.EVALUATE.

    Args:
        full_logistic_reg_model_id (str | None): Fully qualified model ID.

    Returns:
        str: SQL query that evaluates the specified model.
    """
    return f"""
    SELECT*
    FROM ML.EVALUATE(MODEL `{full_logistic_reg_model_id}`);
    """


def predict_fraud_risk_query(full_dataset_id=None, full_logistic_reg_model_id=None):
    """Build a query to generate fraud risk predictions.

    Args:
        full_dataset_id (str | None): Fully qualified dataset ID containing `features`.
        full_logistic_reg_model_id (str | None): Fully qualified model ID.

    Returns:
        str: SQL query that runs ML.PREDICT against the features table.
    """
    return f"""
    SELECT*,
      predicted_fraud_flag,
      predicted_fraud_flag_probs
    FROM ML.PREDICT(
      MODEL `{full_logistic_reg_model_id}`,
      (SELECT * FROM `{full_dataset_id}.features`)
    );
"""
