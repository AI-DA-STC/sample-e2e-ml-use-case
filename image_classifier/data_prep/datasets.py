"""Dataset classes for defining how datasets are to be loaded.
"""

import os
import pandas as pd
import torchvision
from PIL import Image
import torch.utils.data as torch_data


class MNISTDataset(torch_data.Dataset):
    """MNIST dataset class."""

    def __init__(
        self,
        data_dir_path,
        anno_file_name,
        to_grayscale=False,
        to_tensor=False,
        transform=None,
    ):
        """
        Parameters
        ----------
        data_dir_path : str
            Path to the directory that contains the dataset.
        anno_file_name : str
            Name of the CSV file containing the annotations.
        to_grayscale : bool, optional
            Whether to convert the images to grayscale by default. Defaults to False.
        to_tensor : bool, optional
            Whether to convert the images to PyTorch tensors by default.
            Defaults to False.
        transform : callable, optional
            A PyTorch transform sequence to apply to the images. Defaults to None.
        """
        self.data_dir_path = data_dir_path
        self.anno_df = pd.read_csv(os.path.join(data_dir_path, anno_file_name))
        self.to_grayscale = to_grayscale
        self.to_tensor = to_tensor
        self.transform = transform

    def __len__(self):
        """Get the length of the dataset.

        Returns
        -------
        int
            The number of samples in the dataset.
        """
        return len(self.anno_df)

    def __getitem__(self, index):
        """Get a sample from the dataset.

        Parameters
        ----------
        index : int
            Index of the sample to retrieve.

        Returns
        -------
        image_file_name : str
            Name of the image file.
        image : PIL.Image or torch.Tensor
            The image.
        label : int
            The label of the image.
        """
        image_file_name = self.anno_df["filepath"][index]
        image_path = os.path.join(self.data_dir_path, self.anno_df["filepath"][index])
        image = Image.open(image_path)
        if self.to_grayscale:
            image = torchvision.transforms.functional.to_grayscale(image)
        if self.to_tensor:
            image = torchvision.transforms.functional.to_tensor(image)
        if self.transform:
            image = self.transform(image)

        label = self.anno_df["label"][index]

        return image_file_name, image, label
