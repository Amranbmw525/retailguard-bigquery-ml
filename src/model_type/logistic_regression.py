def create_model_log_reg_query(full_dataset_id=None,full_logistic_reg_model_id=None):
    """Creates a new Logistic Regression model in the dataset.Detect known fraud patterns.
        ✔ You know which transactions are fraud
        ✔ You can train a classifier
        ✔ You can evaluate accuracy
        Predict:Probability (transaction is fraud)

        :param full_dataset_id:
        :param full_logistic_reg_model_id:

        Output:
         predicted_fraud_flag
         predicted_probability
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
    """
    Generates and returns a SQL query string to evaluate a machine learning model.
    The query uses the `ML.EVALUATE` function to assess the performance of the
    `logistic_reg_fraud_model` of the specified dataset.

    :param full_logistic_reg_model_id: The fully qualified dataset ID from which the
        machine learning model resides. Defaults to None.
    :type full_logistic_reg_model_id: str or None
    :return: A SQL query string that evaluates the specified machine learning model.
    :rtype: str
    """
    return f"""
    SELECT*
    FROM ML.EVALUATE(MODEL `{full_logistic_reg_model_id}`);
    """


def predict_fraud_risk_query(full_dataset_id=None, full_logistic_reg_model_id=None):
    return f"""
    SELECT*,
      predicted_fraud_flag,
      predicted_fraud_flag_probs
    FROM ML.PREDICT(
      MODEL `{full_logistic_reg_model_id}`,
      (SELECT * FROM `{full_dataset_id}.features`)
    );
"""
