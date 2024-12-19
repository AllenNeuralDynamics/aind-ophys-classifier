import argparse
import json
import os
from datetime import datetime as dt
from datetime import timezone as tz
from pathlib import Path

import h5py
import numpy as np
import roicat
import sparse
from pint import UnitRegistry


def find_file(root_folder, target_file):
    """
    Recursively searches for a file with a specific name within a directory tree.

    Parameters
    ----------
    root_folder : str
        The root directory to start searching from.
    target_file : str
        The name of the file to search for.

    Returns
    -------
    str or None
        The full path to the file if found, or None if the file is not found.

    Examples
    --------
    >>> find_file('/path/to/search', 'target_file.txt')
    '/path/to/search/subfolder/target_file.txt'
    """
    for dirpath, dirnames, filenames in os.walk(root_folder):
        if target_file in filenames:
            return os.path.join(dirpath, target_file)
    return None


def make_output_directory(output_dir: Path, experiment_id: str) -> str:
    """Creates the output directory if it does not exist

    Parameters
    ----------
    output_dir: Path
        output directory
    experiment_id: str
        experiment_id number

    Returns
    -------
    output_dir: str
        output directory
    """
    output_dir = output_dir / experiment_id
    output_dir.mkdir(exist_ok=True)
    output_dir = output_dir / "classification"
    output_dir.mkdir(exist_ok=True)

    return output_dir

def classify_plane(rois, soma_classifier, dendrite_classifier):
    # ROInet embedding
    roinet = roicat.ROInet.ROInet_embedder(
        device=roicat.helpers.set_device(),  ## Which torch device to use ('cpu', 'cuda', etc.)
        dir_networkFiles=args.tmp_dir,  ## Directory to download the pretrained network to
        download_method="check_local_first",  ## Check to see if a model has already been downloaded to the location (will skip if hash matches)
        download_url="https://osf.io/c8m3b/download",  ## URL of the model
        download_hash="357a8d9b630ec79f3e015d0056a4c2d5",  ## Hash of the model file
        forward_pass_version="head",  ## How the data is passed through the network
        verbose=False,  ## Whether to print updates
    )

    roinet.generate_dataloader(
        ROI_images=[rois],  # data.ROI_images,  ## Input images of ROIs
        um_per_pixel=um_per_pixel,  # data.um_per_pixel,  ## Resolution of FOV
        pref_plot=False,  ## Whether or not to plot the ROI sizes
    )
    roinet.generate_latents()

    soma_predictions, soma_probabilities = soma_classifier(roinet.latents)
    dendrite_predictions, dendrite_probabilities = dendrite_classifier(roinet.latents)

    return soma_predictions, soma_probabilities, dendrite_predictions, dendrite_probabilities


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input-dir", type=str, help="Input directory", default="/data"
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, help="Output directory", default="/results"
    )
    parser.add_argument(
        "--tmp-dir",
        type=str,
        default="/scratch",
        help="Directory into which to write temporary files",
    )
    parser.add_argument(
        "--soma-classifier-path",
        type=str,
        default="/data/2p_roi_classifier/soma.classification_training.autoclassifier.onnx",
        help="Path of the classifier model, comma-delimited",
    )
    parser.add_argument(
        "--dendrite-classifier-path",
        type=str,
        default="/data/2p_roi_classifier/dendrite.classification_training.autoclassifier.onnx",
        help="Path of the classifier model, comma-delimited",
    )
    start_time = dt.now(tz.utc)
    args = parser.parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    soma_classifier = (
        roicat.classification.classifier.Load_ONNX_model_sklearnLogisticRegression(
            args.soma_classifier_path
        )
    )

    dendrite_classifier = (
        roicat.classification.classifier.Load_ONNX_model_sklearnLogisticRegression(
            args.dendrite_classifier_path
        )
    )
    
    # get um_per_pixel and dims (FOV size) from session.json
    try:
        session_json_fp = next(input_dir.glob("session.json"))
    except StopIteration:
        session_json_fp = next(input_dir.glob("*/session.json"))
    with open(session_json_fp, "r") as j:
        session_data = json.load(j)
    for data_stream in session_data["data_streams"]:
        if data_stream.get("ophys_fovs"):
            fov = data_stream["ophys_fovs"][0]
            um_per_pixel = (
                float(fov["fov_scale_factor"])
                * (UnitRegistry().parse_expression(fov["fov_scale_factor_unit"]))
                .to("um/pixel")
                .magnitude
            )
            dims = (fov["fov_height"], fov["fov_width"])

    # get ROIs
    for path in Path('/data').rglob('*extraction.h5'):
        plane_id = path.parts[-2]
        with h5py.File(path) as f:
            rois = sparse.COO(
                f["rois/coords"], f["rois/data"], f["rois/shape"]
            ).todense()

        soma_predictions, soma_probabilities, dendrite_predictions, dendrite_probabilities = classify_plane(rois, dendrite_classifier, soma_classifier)
    
        # save results
        with h5py.File(output_dir / "classification.h5", "w") as f:
            grp = f.create_group(str(plane_id))
            grp.create_dataset("soma_predictions", data=soma_predictions)
            grp.create_dataset("soma_probabilities", data=soma_probabilities)
            grp.create_dataset("dendrite_predictions", data=dendrite_predictions)
            grp.create_dataset("dendrite_probabilities", data=dendrite_probabilities)
