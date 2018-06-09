from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import time

from caffe2.python import workspace

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
from detectron.utils.timer import Timer
import detectron.core.test_engine as infer_engine
import detectron.utils.c2 as c2_utils
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.logging
import detectron.utils.vis as vis_utils

from moviepy.editor import VideoFileClip

import pycocotools.mask as mask_util
import numpy as np

from moviepy.editor import VideoFileClip

from smartcrop.helpers import resize_background_image

c2_utils.import_detectron_ops()
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)




def extract_person(image, background_image, model, dataset):

    timers = defaultdict(Timer)
    t = time.time()
    with c2_utils.NamedCudaScope(0):
        cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
            model, image, None, timers=timers
        )
    boxes, segms, keypoints, classes = vis_utils.convert_from_cls_format(
                                        cls_boxes,
                                        cls_segms,
                                        cls_keyps,)


    masks = mask_util.decode(segms)
    masks = np.moveaxis(masks, 2, 0)

    output_image = np.copy(background_image)

    for box, mask , c, in zip(boxes, masks, classes):
        score = box[-1]
        if score < 0.9:
            continue
        if  dataset.classes[c] != 'person':
            continue
        idx = np.where(mask!=0)
        output_image[idx[0], idx[1], :] = image[idx[0], idx[1], :]

    return output_image


def run_task(clip, background_image):

    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    merge_cfg_from_file('/smartcrop/configs/e2e_mask_rcnn_R-101-FPN_2x.yaml')
    assert_and_infer_cfg()
    model = infer_engine.initialize_model_from_cfg('/smartcrop/weights/e2e_mask_rcnn_R-101-FPN_2x/model_final.pkl')
    dataset = dummy_datasets.get_coco_dataset()

    clip = clip.fl_image(lambda image: extract_person(image, background_image, model, dataset))

    return clip
