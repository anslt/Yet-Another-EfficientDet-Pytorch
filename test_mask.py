# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from mask import EfficientMask
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.utils.comm import synchronize
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import os
import argparse

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from efficientdet.dataset import CocoDataset, Resizer, Normalizer, Augmenter, collater

from mask import EfficientMask
from maskrcnn_benchmark.config import cfg


def main():
    ap = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    ap.add_argument('--data_path', type=str, default='datasets/', help='the root folder of dataset')
    ap.add_argument('-n', '--num_workers', type=int, default=12, help='num_workers of dataloader')
    ap.add_argument('--batch_size', type=int, default=12, help='The number of images per batch among all devices')
    ap.add_argument('-p', '--project', type=str, default='coco', help='project file that contains parameters')
    ap.add_argument("-cfg", "--config-file", default="", metavar="FILE", help="path to config file", type=str, )
    ap.add_argument('-c', '--compound_coef', type=int, default=0, help='coefficients of efficientdet')
    ap.add_argument('-w', '--weights', type=str, default=None, help='/path/to/weights')
    ap.add_argument('--nms_threshold', type=float, default=0.5,
                    help='nms threshold, don\'t change it if not for testing purposes')
    ap.add_argument('--cuda', type=bool, default=False)
    ap.add_argument('--device', type=int, default=0)
    ap.add_argument('--float16', type=bool, default=False)
    ap.add_argument('--override', type=bool, default=True, help='override previous bbox results file if exists')
    ap.add_argument("--local_rank", type=int, default=0)
    args = ap.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    # distributed = num_gpus > 1
    #
    # if distributed:
    #     torch.cuda.set_device(args.local_rank)
    #     torch.distributed.deprecated.init_process_group(
    #         backend="nccl", init_method="env://"
    #     )

    compound_coef = args.compound_coef
    nms_threshold = args.nms_threshold
    use_cuda = args.cuda
    gpu = args.device
    use_float16 = args.float16
    override_prev_results = args.override
    project_name = args.project
    weights_path = f'weights/efficientdet-d{compound_coef}.pth' if args.weights is None else args.weights

    print(f'running coco-style evaluation on project {project_name}, weights {weights_path}...')

    params = yaml.safe_load(open(f'projects/{project_name}.yml'))
    obj_list = params["obj_list"]
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]

    cfg.merge_from_file(args.config_file)
    cfg.MODEL.MASK_ON = True
    cfg.RETINANET.NUM_CLASSES = len(obj_list)
    cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES = len(obj_list)
    cfg.freeze()

    save_dir = ""
    # logger = setup_logger("maskrcnn_benchmark", save_dir, get_rank())
    # logger.info("Using {} GPUs".format(num_gpus))
    # logger.info(cfg)
    #
    # logger.info("Collecting env info (might take some time)")
    # logger.info("\n" + collect_env_info())

    # TODO: replace model
    # -----------------------
    model = EfficientMask(cfg, debug=False, compound_coef=compound_coef, num_classes=len(obj_list))
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    # -----------------------

    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    if cfg.OUTPUT_DIR:
        dataset_names = cfg.DATASETS.TEST
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    # TODO: replace dataloader
    # -----------------------
    val_params = {'batch_size': args.batch_size,
                  'shuffle': False,
                  'drop_last': True,
                  'collate_fn': collater,
                  'num_workers': args.num_workers}
    val_set = CocoDataset(root_dir=os.path.join(args.data_path, project_name), set=params["val_set"],
                          transform=transforms.Compose([Normalizer(mean=params["mean"], std=params["std"]),
                                                        Resizer(input_sizes[compound_coef])]))
    val_generator = DataLoader(val_set, **val_params)
    data_loaders_val = [val_generator]
    # -----------------------
    for output_folder, data_loader_val in zip(output_folders, data_loaders_val):
        inference(
            model,
            data_loader_val,
            obj_list,
            iou_types=iou_types,
            #box_only=cfg.MODEL.RPN_ONLY,
            box_only=False if cfg.RETINANET.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device="cpu",
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
        )
        synchronize()


if __name__ == "__main__":
    main()
