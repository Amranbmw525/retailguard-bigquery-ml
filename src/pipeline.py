from src.chart_generator import generate_charts
from src.google_cloud import run_query, ensure_dataset, ensure_transactions_loaded, build_features, train_kmeans, \
    eval_logreg, train_logreg, score_kmeans, build_hybrid

def _run_step(step_name, fn):
    print(f"{step_name}...")
    try:
        fn()
    except Exception as exc:
        print(f"[FAILED] {step_name}: {type(exc).__name__}: {exc}")
        raise
    print(f"[OK] {step_name}")


def run_pipeline():
    _run_step("Creating dataset", ensure_dataset)
    _run_step("Creating transactions table", ensure_transactions_loaded)
    _run_step("Building features", build_features)
    _run_step("Training Logistic Regression", train_logreg)
    _run_step("Evaluating Logistic Regression", eval_logreg)
    _run_step("Training KMeans", train_kmeans)
    _run_step("Scoring anomalies", score_kmeans)
    _run_step("Building hybrid table", build_hybrid)
    _run_step("Generating charts", generate_charts)

    print("Pipeline complete ✅")
