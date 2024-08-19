"""
This script is for training a model on the MNIST dataset.
"""

import os
import sys
import logging
import omegaconf
import hydra
import mlflow
import torch
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import image_classifier as imgc


# pylint: disable = no-value-for-parameter
@hydra.main(version_base=None, config_path="../conf", config_name="pipelines.yaml")
def main(args):
    """This is the main function for training the model.

    Parameters
    ----------
    args : omegaconf.DictConfig
        An omegaconf.DictConfig object containing arguments for the main function.
    """
    args = args["train_model"]

    logger = logging.getLogger(__name__)
    logger.info("Setting up logging configuration.")
    imgc.general_utils.setup_logging(
        logging_config_path=os.path.join(
            hydra.utils.get_original_cwd(), "conf/logging.yaml"
        )
    )

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
            "Missing environment variables for MLflow Tracking: " + str(missing_envvars)
        )
        exit(1)

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

    mlflow.log_params(
        params={
            "learning_rate": args["lr"],
            "gamma": args["gamma"],
            "seed": args["seed"],
            "epochs": args["epochs"],
        }
    )

    torch.manual_seed(args["seed"])

    use_cuda = not args["no_cuda"] and torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda")
    elif not args["no_mps"] and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {"batch_size": args["train_bs"]}
    test_kwargs = {"batch_size": args["test_bs"]}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    train_dataset = imgc.data_prep.datasets.MNISTDataset(
        args["data_dir_path"], "train.csv", to_grayscale=True, to_tensor=True
    )
    test_dataset = imgc.data_prep.datasets.MNISTDataset(
        args["data_dir_path"], "test.csv", to_grayscale=True, to_tensor=True
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    model = imgc.modeling.models.Net().to(device)
    optimiser = torch.optim.Adadelta(model.parameters(), lr=args["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimiser, step_size=1, gamma=args["gamma"]
    )

    for epoch in range(1, args["epochs"] + 1):
        curr_train_loss = imgc.modeling.utils.train(
            args, model, device, train_loader, optimiser, epoch
        )
        curr_test_loss, curr_test_accuracy = imgc.modeling.utils.test(
            model, device, test_loader, epoch
        )

        if epoch % args["model_checkpoint_interval"] == 0:
            logger.info("Exporting the model for epoch %s.", epoch)

            model_checkpoint_path = os.path.join(
                args["model_checkpoint_dir_path"], "model.pt"
            )
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "optimiser_state_dict": optimiser.state_dict(),
                    "train_loss": curr_train_loss,
                    "test_loss": curr_test_loss,
                },
                model_checkpoint_path,
            )

            mlflow.log_artifact(model_checkpoint_path, "model")

        scheduler.step()

    mlflow.log_dict(
        dictionary=omegaconf.OmegaConf.to_container(args, resolve=True),
        artifact_file="train_model_config.json",
    )

    artifact_uri = mlflow.get_artifact_uri()
    logger.info("Artifact URI: %s", artifact_uri)
    mlflow.log_params(params={"artifact_uri": artifact_uri})
    logger.info(
        "Model training with MLflow run ID %s has completed.",
        mlflow_run.info.run_id,
    )
    mlflow.end_run()

    logger.info("Model training has completed.")

    return curr_test_loss, curr_test_accuracy


if __name__ == "__main__":
    main()
