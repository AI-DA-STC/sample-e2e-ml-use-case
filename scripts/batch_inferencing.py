"""Script to conduct batch inferencing on a directory of images.
"""

import os
import sys
import datetime
import logging
import glob
from pathlib import Path
import hydra
import jsonlines
import torchvision
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))

import image_classifier as imgc  # pylint: disable = wrong-import-position


# pylint: disable = no-value-for-parameter
@hydra.main(version_base=None, config_path="../conf", config_name="pipelines.yaml")
def main(args):
    """This main function does the following:
    - load logging config
    - gets list of files to be loaded for inferencing
    - loads trained model
    - conducts inferencing on data
    - outputs prediction results to a jsonline file
    """
    args = args["batch_infer"]

    logger = logging.getLogger(__name__)
    logger.info("Setting up logging configuration.")
    logger_config_path = os.path.join(
        hydra.utils.get_original_cwd(), "conf/logging.yaml"
    )
    imgc.general_utils.setup_logging(logger_config_path)

    logger.info("Loading the model...")
    loaded_model, device = imgc.modeling.utils.load_model(
        args["model_path"], args["use_cuda"], args["use_mps"]
    )

    glob_expr = args["input_data_dir"] + "/*.png"
    logger.info("Conducting inferencing on image files...")

    for image_file in glob.glob(glob_expr):
        image = Image.open(image_file)
        image = torchvision.transforms.functional.to_grayscale(image)
        image = torchvision.transforms.functional.to_tensor(image)
        output = loaded_model(image.unsqueeze(0).to(device))
        pred = output.argmax(dim=1, keepdim=True)
        pred_str = str(int(pred[0]))

        curr_time = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S%z"
        )
        curr_res_jsonl = {
            "time": curr_time,
            "image_filepath": image_file,
            "prediction": pred_str,
        }

        with jsonlines.open("batch-infer-res.jsonl", mode="a") as writer:
            writer.write(curr_res_jsonl)
            writer.close()

    logger.info("Batch inferencing has completed.")
    logger.info("Output result location: %s/batch-infer-res.jsonl", os.getcwd())


if __name__ == "__main__":
    main()
