"""Functions for facilitating transformation of datasets.
"""

import os
import logging
import pandas as pd
import torch.utils.data as torch_data
import torchvision

import image_classifier as imgc

logger = logging.getLogger(__name__)


def save_batch_images(batch, dest_dir_path):
    """
    Saves an image tensor from a batch to the specified directory.

    Parameters
    ----------
    batch : tuple
        A tuple containing the image file name, the image tensor, and its label.
    dest_dir_path : str
        The path to the output directory.
    """
    image_file_name, image, _ = batch
    image_file_dest_path = os.path.join(dest_dir_path, image_file_name[0])
    os.makedirs(os.path.dirname(image_file_dest_path), exist_ok=True)
    torchvision.utils.save_image(image[0], image_file_dest_path)


def transform_raw_dataset(raw_data_dir_path, processed_data_dir_path):
    """
    Transforms the raw dataset into the processed dataset.

    Parameters
    ----------
    raw_data_dir_path : str
        The path to the directory containing the raw dataset.
    processed_data_dir_path : str
        The path to the directory to save the processed dataset.

    Returns
    -------
    anno_train_df : pd.DataFrame
        The annotation dataframe for the processed training dataset.
    test_df : pd.DataFrame or None
        The annotation dataframe for the processed test dataset, or None if the test dataset is not present.
    """
    dir_files = os.listdir(raw_data_dir_path)
    logger.info("Processing raw training data for directory: %s", raw_data_dir_path)
    anno_train_df = pd.read_csv(os.path.join(raw_data_dir_path, "train.csv"))
    train_dataset = imgc.data_prep.datasets.MNISTDataset(
        raw_data_dir_path,
        "train.csv",
        to_tensor=False,
        transform=imgc.data_prep.transforms.MNIST_TRANSFORM_STEPS["train"],
    )
    train_dataloader = torch_data.DataLoader(train_dataset)

    for batch in train_dataloader:
        save_batch_images(batch, processed_data_dir_path)

    if "test.csv" in dir_files:
        logger.info("Processing raw test data for directory: %s", raw_data_dir_path)
        test_df = pd.read_csv(os.path.join(raw_data_dir_path, "test.csv"))
        test_dataset = imgc.data_prep.datasets.MNISTDataset(
            raw_data_dir_path,
            "test.csv",
            to_tensor=False,
            transform=imgc.data_prep.transforms.MNIST_TRANSFORM_STEPS["test"],
        )
        test_dataloader = torch_data.DataLoader(test_dataset)
        for batch in test_dataloader:
            save_batch_images(batch, processed_data_dir_path)
    else:
        test_df = None

    return anno_train_df, test_df
