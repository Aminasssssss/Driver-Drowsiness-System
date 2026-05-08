import mlflow
from typing import Dict, Any, Optional
from .config import settings


class MLOpsTracker:

    def __init__(self, experiment_name: Optional[str] = None):
        self.experiment_name = experiment_name or settings.mlflow_experiment
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        self._run_id: Optional[str] = None

    def start_session(self, params: Dict[str, Any]) -> None:
        if mlflow.active_run():
            mlflow.end_run()
        run = mlflow.start_run()
        self._run_id = run.info.run_id
        mlflow.log_params(params)
        print(f"[MLflow] Session started — run_id: {self._run_id}")

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        if not mlflow.active_run():
            return
        mlflow.log_metrics(metrics, step=step)

    def log_event(self, event_name: str, value: int, step: int) -> None:
        if not mlflow.active_run():
            return
        mlflow.log_metric(event_name, value, step=step)

    def end_session(self) -> None:
        if mlflow.active_run():
            mlflow.end_run()
            print(f"[MLflow] Session ended — run_id: {self._run_id}")
        self._run_id = None