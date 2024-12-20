import argparse
import json
import os
from datetime import datetime as dt
from datetime import timezone as tz
from pathlib import Path
from dataclasses import dataclass
import matplotlib.pyplot as plt

import h5py
import numpy as np
import roicat
import sparse
from pint import UnitRegistry
from typing import List
from plots import plot_probabilities, plot_predictions, plot_border_rois


@dataclass
class Plane:
    name: str
    input_extraction_file: Path
    output_dir: Path
    output_classification_file: Path
    rois: np.array


def classify_plane(
    rois: np.array, um_per_pixel: float, soma_classifier, dendrite_classifier
):
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

    data = roicat.data_importing.Data_roicat()
    data.set_spatialFootprints(rois, um_per_pixel=um_per_pixel)
    data.transform_spatialFootprints_to_ROIImages(out_height_width=(36, 36))

    roinet.generate_dataloader(
        ROI_images=data.ROI_images,  ## Input images of ROIs
        um_per_pixel=data.um_per_pixel,  ## Resolution of FOV
        pref_plot=False,  ## Whether or not to plot the ROI sizes
    )
    roinet.generate_latents()

    soma_predictions, soma_probabilities = soma_classifier(roinet.latents)
    dendrite_predictions, dendrite_probabilities = dendrite_classifier(roinet.latents)

    return (
        soma_predictions,
        soma_probabilities,
        dendrite_predictions,
        dendrite_probabilities,
    )


def find_um_per_pixel(input_dir: Path) -> float:
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

    return um_per_pixel


def prepare_plane(extraction_file: Path, output_dir: Path) -> List[Plane]:
    plane_name = path.parts[-3]
    plane_dir = output_dir / "classification" / plane_name
    plane = Plane(
        name=plane_name,
        input_extraction_file=path,
        output_dir=plane_dir,
        output_classification_file=plane_dir / "classification.h5",
        rois=load_extraction_rois(extraction_file),
    )

    plane.output_dir.mkdir(parents=True, exist_ok=True)

    return plane


def load_extraction_rois(extraction_file: Path) -> np.array:
    with h5py.File(extraction_file) as f:
        return sparse.COO(f["rois/coords"], f["rois/data"], f["rois/shape"]).todense()


def classify_border_rois(rois: np.array, border_size: int) -> np.array:
    masks = np.where(rois > 0)
    border_rois = set()
    border_rois.update(masks[0][np.where(masks[1] < border_size)[0]].tolist())
    border_rois.update(masks[0][np.where(masks[2] < border_size)[0]].tolist())
    border_rois.update(
        masks[0][np.where(masks[1] >= rois.shape[1] - border_size)[0]].tolist()
    )
    border_rois.update(
        masks[0][np.where(masks[2] >= rois.shape[2] - border_size)[0]].tolist()
    )

    border_rois_bool = np.zeros(shape=rois.shape[0], dtype=bool)
    border_rois_bool[list(border_rois)] = True

    return border_rois_bool


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
    parser.add_argument(
        "--border-size",
        type=int,
        default=10,
        help="Distance from border for ROI to be classified on the border",
    )
    start_time = dt.now(tz.utc)
    args = parser.parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    um_per_pixel = find_um_per_pixel(input_dir)

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

    for path in Path(input_dir).rglob("*extraction.h5"):
        plane = prepare_plane(path, output_dir)

        (
            soma_predictions,
            soma_probabilities,
            dendrite_predictions,
            dendrite_probabilities,
        ) = classify_plane(
            rois=plane.rois,
            um_per_pixel=um_per_pixel,
            soma_classifier=soma_classifier,
            dendrite_classifier=dendrite_classifier,
        )

        border_rois = classify_border_rois(
            rois=plane.rois, border_size=args.border_size
        )

        ax = plot_probabilities(plane.rois, soma_probabilities, "soma probabilities")
        plt.savefig(str(plane.output_dir / "soma_probabilities.png"))

        ax = plot_probabilities(
            plane.rois, dendrite_probabilities, "dendrite probabilities"
        )
        plt.savefig(str(plane.output_dir / "dendrite_probabilities.png"))

        ax = plot_predictions(plane.rois, soma_predictions, "soma predictions")
        plt.savefig(str(plane.output_dir / "soma_predictions.png"))

        ax = plot_predictions(plane.rois, dendrite_predictions, "dendrite predictions")
        plt.savefig(str(plane.output_dir / "dendrite_predictions.png"))

        ax = plot_border_rois(plane.rois, border_rois, "border ROIs")
        plt.savefig(str(plane.output_dir / "border_rois.png"))

        # save results
        with h5py.File(plane.output_classification_file, "w") as f:
            g = f.create_group("soma")
            g.create_dataset("predictions", data=soma_predictions)
            g.create_dataset("probabilities", data=soma_probabilities)

            g = f.create_group("dendrites")
            g.create_dataset("predictions", data=dendrite_predictions)
            g.create_dataset("probabilities", data=dendrite_probabilities)

            g = f.create_group("border")
            g.create_dataset("labels", data=border_rois)
