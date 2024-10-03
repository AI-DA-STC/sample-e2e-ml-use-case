"""
This module contains general utility functions for the package.
"""

import os
import sys
import logging
import typing
import yaml
import mlflow
import torch
import omegaconf

logger = logging.getLogger(__name__)


def setup_logging(
    logging_config_path: str = "./conf/logging.yaml",
    default_level: int = logging.INFO,
    exclude_handlers: list = None,
    use_log_filename_prefix: bool = False,
    log_filename_prefix: str = "",
):
    """Load a specified custom configuration for logging.

    Parameters
    ----------
    logging_config_path : str, optional
        Path to the logging YAML configuration file, by default "./conf/logging.yaml"
    default_level : int, optional
        Default logging level to use if the configuration file is not found,
        by default logging.INFO
    """
    try:
        with open(logging_config_path, "rt", encoding="utf-8") as file:
            log_config = yaml.safe_load(file.read())

        if use_log_filename_prefix:
            for handler in log_config["handlers"]:
                if "filename" in log_config["handlers"][handler]:
                    curr_log_filename = log_config["handlers"][handler]["filename"]
                    log_config["handlers"][handler]["filename"] = os.path.join(
                        log_filename_prefix, curr_log_filename
                    )

        logging_handlers = log_config["root"]["handlers"]
        if exclude_handlers:
            log_config["root"]["handlers"] = [
                handler
                for handler in logging_handlers
                if handler not in exclude_handlers
            ]

        logging.config.dictConfig(log_config)
        logger.info("Successfully loaded custom logging configuration.")

    except FileNotFoundError as error:
        logging.basicConfig(
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            level=default_level,
        )
        logger.info(error)
        logger.info("Logging config file is not found. Basic config is being used.")


def init_mlflow_run(args: dict) -> mlflow.tracking.MlflowClient:
    """Initialize MLflow run.

    Parameters
    ----------
    args : dict
        A dictionary containing configuration options for the MLflow run.

    Returns
    -------
    mlflow.MlflowClient
        An instance of the MLflow client that can be used to interact with the
        MLflow server.
    """
    mlflow_envvars = [
        "MLFLOW_TRACKING_URI",
        "MLFLOW_TRACKING_USERNAME",
        "MLFLOW_TRACKING_PASSWORD",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
    ]

    missing_envvars = []
    for envvar in mlflow_envvars:
        if not envvar in os.environ:
            missing_envvars.append(envvar)

    if missing_envvars:
        logger.error(
            "Missing environment variables for MLflow Tracking: %s",
            ", ".join(missing_envvars),
        )
        sys.exit(1)

    mlflow_tracking_uri = mlflow.get_tracking_uri()
    mlflow.set_experiment(args["mlflow_exp_name"])
    mlflow_experiment = mlflow.get_experiment_by_name(args["mlflow_exp_name"])

    mlflow.start_run()
    mlflow_run = mlflow.active_run()

    logger.info(
        "Logging experiment to MLflow Tracking server at %s", mlflow_tracking_uri
    )
    logger.info("MLflow experiment ID: %s", mlflow_experiment.experiment_id)
    logger.info("UUID for MLflow run: %s", mlflow_run.info.run_id)
    logger.info("Artifact location: %s", mlflow_experiment.artifact_location)

    return mlflow_run


def get_accelerator_device(
    args: omegaconf.DictConfig,
) -> typing.Tuple[bool, torch.device]:
    """Returns the device type to be used for PyTorch operations.

    Parameters
    ----------
    args : omegaconf.DictConfig
        An omegaconf.DictConfig object containing arguments from the main function.

    Returns
    -------
    use_cuda : bool
        Whether or not to use CUDA acceleration.
    device : torch.device
        The device type to be used for PyTorch operations.
    """
    use_cuda = not args["no_cuda"] and torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda")
    elif not args["no_mps"] and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    return use_cuda, device
