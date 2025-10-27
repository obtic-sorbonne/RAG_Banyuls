import mlflow
import yaml
from datetime import datetime

class ExperimentTracker:
    def __init__(self, tracking_uri="http://localhost:5000"):
        mlflow.set_tracking_uri(tracking_uri)
        self.client = mlflow.tracking.MlflowClient()
    
    def start_run(self, experiment_name: str, config: dict):
        """Start a new experiment run"""
        mlflow.set_experiment(experiment_name)
        self.run = mlflow.start_run()
        
        # Log configuration
        mlflow.log_params(config)
        mlflow.log_dict(config, "config.yaml")
        
        # Record start time
        self.start_time = datetime.now()
    
    def log_metrics(self, metrics: dict, step: int = None):
        """Log evaluation metrics"""
        mlflow.log_metrics(metrics, step=step)
    
    def log_artifact(self, file_path: str):
        """Log an artifact file"""
        mlflow.log_artifact(file_path)
    
    def log_model(self, model, artifact_path: str):
        """Log a model artifact"""
        mlflow.pyfunc.log_model(artifact_path, python_model=model)
    
    def end_run(self, status: str = "FINISHED"):
        """End the current run"""
        # Calculate duration
        duration = datetime.now() - self.start_time
        mlflow.log_metric("duration_seconds", duration.total_seconds())
        
        # Set status
        mlflow.end_run(status=status)
    
    def compare_runs(self, experiment_name: str, metric: str = "f1@k"):
        """Compare runs for a given metric"""
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if not experiment:
            return {}
        
        runs = mlflow.search_runs(experiment.experiment_id)
        if runs.empty:
            return {}
        
        # Find best run for the metric
        best_run = runs.loc[runs[f"metrics.{metric}"].idxmax()]
        return {
            "best_run_id": best_run.run_id,
            "best_score": best_run[f"metrics.{metric}"],
            "all_runs": runs[[f"metrics.{metric}", "run_id"]].to_dict("records")
        }
