"""This script processes raw MNIST images and saves them in the processed data directory.
"""

import os
import sys
import shutil
import logging
import hydra
import torchvision
from pathlib import Path
import pandas as pd
import torch.utils.data as torch_data

sys.path.append(str(Path(__file__).resolve().parent.parent))

import image_classifier as imgc


# pylint: disable = no-value-for-parameter
@hydra.main(version_base=None, config_path="../conf", config_name="pipelines.yaml")
def main(args):
    """This function processes raw MNIST images and saves them in the processed data directory.

    Parameters
    ----------
    args : omegaconf.DictConfig
        An omegaconf.DictConfig object containing arguments for the main function.
    """
    args = args["process_data"]

    logger = logging.getLogger(__name__)
    logger.info("Setting up logging configuration.")
    imgc.general_utils.setup_logging(
        logging_config_path=os.path.join(
            hydra.utils.get_original_cwd(), "conf/logging.yaml"
        )
    )

    raw_data_dir_path = args["raw_data_dir_path"]
    raw_data_subdir_list = os.listdir(raw_data_dir_path)
    raw_data_subdir_list = [
        item for item in raw_data_subdir_list if not item.startswith(".")
    ]

    processed_data_dir_path = args["processed_data_dir_path"]
    os.makedirs(args["processed_data_dir_path"], exist_ok=True)

    combined_anno_train_df_list = []
    for raw_data_subdir_name in raw_data_subdir_list:
        curr_raw_data_subdir_path = os.path.join(
            raw_data_dir_path, raw_data_subdir_name
        )
        curr_dir_files = os.listdir(curr_raw_data_subdir_path)

        logger.info("Processing raw data for directory: %s", curr_raw_data_subdir_path)

        curr_anno_train_df = pd.read_csv(
            os.path.join(curr_raw_data_subdir_path, "train.csv")
        )
        combined_anno_train_df_list.append(curr_anno_train_df)
        curr_train_dataset = imgc.data_prep.datasets.MNISTDataset(
            curr_raw_data_subdir_path,
            "train.csv",
            to_tensor=False,
            transform=imgc.data_prep.transforms.MNIST_TRANSFORM_STEPS["train"],
        )
        curr_train_dataloader = torch_data.DataLoader(curr_train_dataset)

        for batch in curr_train_dataloader:
            train_image_file_name, train_image, _ = batch
            train_image_file_dest_path = os.path.join(
                processed_data_dir_path, train_image_file_name[0]
            )
            os.makedirs(os.path.dirname(train_image_file_dest_path), exist_ok=True)
            torchvision.utils.save_image(train_image[0], train_image_file_dest_path)

        if "test.csv" in curr_dir_files:
            test_dataset = imgc.data_prep.datasets.MNISTDataset(
                curr_raw_data_subdir_path,
                "test.csv",
                to_tensor=False,
                transform=imgc.data_prep.transforms.MNIST_TRANSFORM_STEPS["test"],
            )
            test_dataloader = torch_data.DataLoader(test_dataset)

            for batch in test_dataloader:
                test_image_file_name, test_image, _ = batch
                test_image_file_dest_path = os.path.join(
                    processed_data_dir_path, test_image_file_name[0]
                )
                os.makedirs(os.path.dirname(test_image_file_dest_path), exist_ok=True)
                torchvision.utils.save_image(test_image[0], test_image_file_dest_path)
                shutil.copy(
                    os.path.join(curr_raw_data_subdir_path, "test.csv"),
                    os.path.join(processed_data_dir_path, "test.csv"),
                )

    combined_anno_train_df = pd.concat(combined_anno_train_df_list)
    combined_anno_train_df.to_csv(
        os.path.join(processed_data_dir_path, "train.csv"), index=False
    )

    logger.info("All raw data has been processed.")


if __name__ == "__main__":
    main()
