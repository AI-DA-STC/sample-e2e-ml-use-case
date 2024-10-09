"""Script for testing of logging to MLflow server.
"""

import sys
import time
import random
import mlflow


def main():
    """Main function for testing of logging to MLflow server.

    This function:
    - logs parameters
    - logs metrics
    - logs an artifact
    - prints the current tracking URI and artifact location
    """
    mlflow.set_experiment(sys.argv[1])
    dummy_param_1 = 64
    dummy_param_2 = 1.0

    tracking_uri = mlflow.get_tracking_uri()
    experiment = mlflow.get_experiment_by_name(sys.argv[1])

    print("Current tracking URI: ", tracking_uri)
    print("Current unique experiment ID: ", experiment.experiment_id)
    print("Current location of artifacts: ", experiment.artifact_location)

    mlflow.start_run()

    print("Logging parameters...")
    # Log two dummy parameters
    mlflow.log_param("dummy_param_1", dummy_param_1)
    mlflow.log_param("dummy_param_2", dummy_param_2)

    print("Logging metrics...")
    # Log two dummy metrics for 4 steps
    for step in range(1, 5):
        mlflow.log_metric("dummy_metric_1", time.time(), step=step)
        mlflow.log_metric("dummy_metric_2", random.uniform(0.1, 0.5), step=step)
        time.sleep(2)

    print("Logging artifact...")
    # Create a text artifact and log it
    dummy_text_content = "This text content should be uploaded to the MinIO bucket."
    with open("text_artifact.txt", "w", encoding="utf-8") as file:
        file.write(dummy_text_content)
    mlflow.log_artifact("text_artifact.txt")

    artifact_uri = mlflow.get_artifact_uri()
    print("Current artifact URI: ", artifact_uri)

    mlflow.end_run()

    print("Test script has run successfully!")


if __name__ == "__main__":
    main()
