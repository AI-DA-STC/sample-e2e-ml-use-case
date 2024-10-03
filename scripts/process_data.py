"""
This script processes raw MNIST images and saves them in the processed data directory.
"""

import os
import sys
import logging
from pathlib import Path
import hydra
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

import image_classifier as imgc # pylint: disable = wrong-import-position


# pylint: disable = no-value-for-parameter
@hydra.main(version_base=None, config_path="../conf", config_name="pipelines.yaml")
def main(args):
    """
    This function processes raw MNIST images and saves them in the processed data
    directory.

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
    combined_test_df_list = []
    for raw_data_subdir_name in raw_data_subdir_list:
        raw_data_dir_fullpath = os.path.join(raw_data_dir_path, raw_data_subdir_name)
        curr_anno_train_df, curr_test_df = imgc.data_prep.utils.transform_raw_dataset(
            raw_data_dir_fullpath, processed_data_dir_path
        )
        combined_anno_train_df_list.append(curr_anno_train_df)
        combined_test_df_list.append(curr_test_df)

    combined_anno_train_df = pd.concat(combined_anno_train_df_list)
    combined_anno_train_df.to_csv(
        os.path.join(processed_data_dir_path, "train.csv"), index=False
    )
    combined_test_df = pd.concat(combined_test_df_list)
    combined_test_df.to_csv(
        os.path.join(processed_data_dir_path, "test.csv"), index=False
    )

    logger.info("All raw data has been processed.")


if __name__ == "__main__":
    main()
