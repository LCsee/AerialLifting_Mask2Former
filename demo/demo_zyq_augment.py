# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import argparse
import glob
import multiprocessing as mp
import os
import gzip

# fmt: off
import sys

from matplotlib import cm

from tta_handler import TTAHandler

sys.path.insert(1, os.path.join(sys.path[0], '..'))
sys.path.insert(1, os.path.join(sys.path[0], '.'))

# fmt: on

import tempfile
import time
import warnings
import torch
import cv2
import albumentations as A
import numpy as np
from tqdm import tqdm
from pathlib import Path
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from mask2former import add_maskformer2_config
from predictor import VisualizationDemo
from PIL import Image
import torchvision.transforms as T
import math

from detectron2.utils.file_io import PathManager
from detectron2.data.detection_utils import _apply_exif_orientation, convert_PIL_to_numpy

ori_w = 2000
ori_h = 1333

resize_scale = 1

extensions = {".jpg", ".png", ".tif", ".jpeg", ".tiff"}

label_color = {
    0: [0, 0, 0],  # cluster
    1: [128, 0, 0],  # building
    2: [192, 192, 192],  # road
    3: [192, 0, 192],  # car
    4: [0, 128, 0],  # tree
    5: [128, 128, 0],  # vegetation
    6: [255, 255, 0],  # human
    7: [135, 206, 250],  # sky
    8: [0, 0, 128],  # water
    9: [252, 230, 201],  # ground
    10: [128, 64, 128]  # mountain
}


def label2rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    for i in range(10):
        mask_rgb[np.all(mask_convert == i, axis=0)] = label_color[i]
    return mask_rgb


def visualize_mask_folder(path_to_folder, offset=0):

    (path_to_folder.parent /
     f"visualized_{path_to_folder.stem}").mkdir(exist_ok=True)
    (path_to_folder.parent / "labels_m2f").mkdir(exist_ok=True)

    used_files = [
        str(file.relative_to(path_to_folder))
        for file in path_to_folder.rglob('*')
        if file.suffix.lower() in extensions
    ]

    for f in tqdm(used_files, desc='visualizing masks'):
        label_img = Image.open(path_to_folder / f)
        rgbs = label2rgb(np.array(label_img))
        rgb_save_path = Path(path_to_folder.parent /
                             f"visualized_{path_to_folder.stem}" / f)
        rgb_save_path.parent.mkdir(exist_ok=True, parents=True)
        Image.fromarray(rgbs).save(rgb_save_path)

        label_img = label_img.resize((ori_w, ori_h), Image.Resampling.NEAREST)
        resize_save_path = Path(path_to_folder.parent / "labels_m2f" / f)
        resize_save_path.parent.mkdir(exist_ok=True, parents=True)
        label_img.save(resize_save_path)


# constants
WINDOW_NAME = "mask2former demo"


def convert_from_mask_to_semantics_and_instances_no_remap(
        original_mask, segments):
    id_to_class = torch.zeros(1024).int()
    original_mask = original_mask.cpu()
    for s in segments:
        id_to_class[s['id']] = s['category_id']
    return id_to_class[original_mask.flatten().numpy().tolist()].reshape(
        original_mask.shape)


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.zyq_code = args.zyq_code
    cfg.zyq_mapping = args.zyq_mapping
    cfg.freeze()
    return cfg


def visualize_tensor(tensor, minval=0.000, maxval=1.00, use_global_norm=True):
    x = tensor.cpu().numpy()
    x = np.nan_to_num(x)  # change nan to 0
    if use_global_norm:
        mi = minval
        ma = maxval
    else:
        mi = np.min(x)  # get minimum depth
        ma = np.max(x)
    x = (x - mi) / (ma - mi + 1e-8)  # normalize to 0~1
    x_ = Image.fromarray((cm.get_cmap('jet')(x) * 255).astype(np.uint8))
    x_ = T.ToTensor()(x_)[:3, :, :]
    return x_


def probability_to_normalized_entropy(probabilities):
    entropy = torch.zeros_like(probabilities[:, :, 0])
    for i in range(probabilities.shape[2]):
        entropy = entropy - probabilities[:, :, i] * torch.log2(
            probabilities[:, :, i] + 1e-8)
    entropy = entropy / math.log2(probabilities.shape[2])
    return entropy


def load_and_save_with_entropy_and_confidence(out_filename, entropy,
                                              confidences):
    from torchvision.io import read_image
    from torchvision.utils import save_image
    org_img = read_image(out_filename).float() / 255.0
    e_img = visualize_tensor(1 - entropy)
    c_img = visualize_tensor(confidences)
    save_image(torch.cat(
        [org_img.unsqueeze(0),
         e_img.unsqueeze(0),
         c_img.unsqueeze(0)], dim=0),
               out_filename,
               value_range=(0, 1),
               normalize=True)


def save_panoptic(predictions, predictions_notta, _demo, out_filename):
    mask, segments, probabilities, confidences = predictions["panoptic_seg"]
    if len(predictions_notta["panoptic_seg"]) != 4:
        mask_notta, segments_notta = predictions_notta["panoptic_seg"]
        confidences_notta = torch.zeros_like(mask_notta)
    else:
        mask_notta, segments_notta, _, confidences_notta = predictions_notta[
            "panoptic_seg"]
    # since we use cat_ids from scannet, no need for mapping
    # for segment in segments:
    #     cat_id = segment["category_id"]
    #     segment["category_name"] = demo.metadata.stuff_classes[cat_id]
    with gzip.open(out_filename, "wb") as fid:
        torch.save(
            {
                "mask": mask,
                "segments": segments,
                "mask_notta": mask_notta,
                "segments_notta": segments_notta,
                "confidences_notta": confidences_notta,
                "probabilities": probabilities,
                "confidences": confidences,
                # "feats": predictions["res3_feats"]
            },
            fid)


def get_parser():
    parser = argparse.ArgumentParser(
        description="maskformer2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default=
        "configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--predictions",
        help="Save raw predictions together with visualizations.")
    parser.add_argument("--webcam",
                        action="store_true",
                        help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--n",
        type=int,
        default=1,
        help="Max procs",
    )
    parser.add_argument(
        "--p",
        type=int,
        default=0,
        help="Current proc",
    )
    parser.add_argument(
        "--zyq_code",
        help="",
        default=False,
    )
    parser.add_argument(
        "--zyq_mapping",
        help="",
        default=False,
    )
    parser.add_argument(
        "--resize",
        help="",
        default=False,
    )
    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg, True)

    input_path = Path(args.input[0])

    if args.input:
        used_files = [
            str(file.relative_to(input_path)) for file in input_path.rglob('*')
            if file.suffix.lower() in extensions
        ]
        used_files.sort()
        print(f"pending files: {len(used_files)}")

        for relative_path in tqdm(used_files[408:410], disable=not args.output):
            path = input_path / relative_path

            with PathManager.open(path, "rb") as f:
                image = Image.open(f)
                # resize
                image_width, image_height = image.size
                if args.resize:
                    image = image.resize((int(image_width / resize_scale),
                                          int(image_height / resize_scale)))
                else:
                    image = image.resize((int(image_width), int(image_height)))

            ##### work around this bug: https://github.com/python-pillow/Pillow/issues/3973
            image = _apply_exif_orientation(image)
            img = convert_PIL_to_numpy(image, "BGR")

            start_time = time.time()
            augmentations = [
                A.HorizontalFlip(always_apply=True),
                A.RGBShift(always_apply=True),
                A.CLAHE(always_apply=True),
                A.RandomGamma(always_apply=True, gamma_limit=(80, 120)),
                A.RandomBrightnessContrast(always_apply=True),
                A.MedianBlur(blur_limit=7, always_apply=True),
                A.Sharpen(alpha=(0.2, 0.4),
                          lightness=(0.5, 1.0),
                          always_apply=True)
            ]
            augmentations.extend([
                A.Compose([augmentations[1], augmentations[2]]),
                A.Compose([augmentations[2], augmentations[3]]),
                A.Compose([augmentations[1], augmentations[3]]),
                A.Compose([augmentations[2], augmentations[4]]),
                A.Compose([augmentations[5], augmentations[6]])
            ])

            # NOTE 以下得到的semantic， instance， probability, confidences等全部是针对ADE20K的
            predictions_0, _ = demo.run_on_image(img, visualize=False)

            list_aug_probs, list_aug_confs = [
                x.cpu() for x in predictions_0["panoptic_seg"][0]
            ], [x.cpu() for x in predictions_0["panoptic_seg"][1]]
            for aud_idx, augmentation in enumerate(augmentations):
                transformed_image = augmentation(image=img)["image"]
                aug_pred, _ = demo.run_on_image(transformed_image,
                                                visualize=False)
                if not aud_idx == 0:
                    aug_probs, aug_conf = aug_pred["panoptic_seg"][
                        0], aug_pred["panoptic_seg"][1]
                    # aug_feat = aug_pred['res3_feats']
                else:
                    aug_probs, aug_conf = aug_pred["panoptic_seg"][
                        0], torch.fliplr(aug_pred["panoptic_seg"][1].permute(
                            (1, 2, 0))).permute((2, 0, 1))

                aug_probs = aug_probs.cpu()
                aug_conf = aug_conf.cpu()
                list_aug_probs.extend([x for x in aug_probs])
                list_aug_confs.extend([x for x in aug_conf])

            tta_handler_start_time = time.time()
            tta_handler = TTAHandler(list_aug_probs, list_aug_confs)
            probabilities, confidences = tta_handler.find_tta_probabilities_and_masks(
            )
            print(
                f'TTA Handler time: {time.time() - tta_handler_start_time:.2f}s'
            )
            del tta_handler
            # todo: deleted visualizations for now, turn on if needed
            predictions, visualized_output = demo.run_post_augmentation(
                img, probabilities, confidences, visualize=True)
            # predictions, visualized_output = demo.run_post_augmentation(img, probabilities, confidences, visualize=False)

            predictions_no_tta, _ = demo.run_post_augmentation(
                img,
                predictions_0["panoptic_seg"][0],
                predictions_0["panoptic_seg"][1],
                visualize=False)
            # predictions['res3_feats'] = averaged_feats
            logger.info("{}: {} in {:.2f}s".format(
                path,
                "detected {} instances".format(len(predictions["instances"]))
                if "instances" in predictions else "finished",
                time.time() - start_time,
            ))

            if args.output:
                output_path = Path(args.output)
                output_path.mkdir(exist_ok=True, parents=True)
                Path(output_path / "labels_m2f_ori").mkdir(exist_ok=True,
                                                           parents=True)
                Path(output_path /
                     "visualized_labels_m2f_ori").mkdir(exist_ok=True)
                Path(output_path / "labels_m2f").mkdir(exist_ok=True)
                Path(output_path / 'alpha').mkdir(exist_ok=True)
                Path(output_path / 'visualized_ptz').mkdir(exist_ok=True)
                Path(output_path / 'panoptic').mkdir(exist_ok=True)

                #save semantic
                mask, segments, _, _ = predictions["panoptic_seg"]
                semantic = convert_from_mask_to_semantics_and_instances_no_remap(
                    mask, segments)

                output_file = Path(output_path / 'labels_m2f_ori' /
                                   relative_path).with_suffix(".png")
                out_file_path = output_file.parent
                out_file_path.mkdir(exist_ok=True, parents=True)
                label_img = Image.fromarray(semantic.numpy().astype(np.uint8))
                label_img.save(output_file.with_suffix(".png"))

                # add
                rgbs = label2rgb(np.array(label_img))
                rgb_save_path = Path(output_path /
                                     f"visualized_labels_m2f_ori" /
                                     relative_path).with_suffix(".png")
                rgb_save_path.parent.mkdir(exist_ok=True, parents=True)
                Image.fromarray(rgbs).save(rgb_save_path)

                label_img = label_img.resize((ori_w, ori_h),
                                             Image.Resampling.NEAREST)
                resize_save_path = Path(output_path / "labels_m2f" /
                                        relative_path).with_suffix(".png")
                resize_save_path.parent.mkdir(exist_ok=True, parents=True)
                label_img.save(resize_save_path)

                merge = 0.5 * rgbs + 0.5 * img
                merge_save_path = Path(output_path / "alpha" /
                                       relative_path).with_suffix(".png")
                merge_save_path.parent.mkdir(exist_ok=True, parents=True)
                Image.fromarray(merge.astype(np.uint8)).save(merge_save_path)

                if visualized_output is not None:
                    visualized_ptz_save_path = Path(
                        output_path / f"visualized_ptz" /
                        relative_path).with_suffix(".jpg")
                    visualized_ptz_save_path.parent.mkdir(exist_ok=True,
                                                          parents=True)
                    visualized_output.save(visualized_ptz_save_path)

                    probabilities, confidences = predictions["panoptic_seg"][
                        2], predictions["panoptic_seg"][3]
                    entropy = probability_to_normalized_entropy(probabilities)
                    load_and_save_with_entropy_and_confidence(
                        str(visualized_ptz_save_path), entropy, confidences)

                    panoptic_save_path = Path(
                        output_path / f"panoptic" /
                        relative_path).with_suffix(".ptz")
                    panoptic_save_path.parent.mkdir(exist_ok=True,
                                                    parents=True)
                    save_panoptic(predictions, predictions_no_tta, demo,
                                  panoptic_save_path)
            else:
                print("something wrong in the code or input")
