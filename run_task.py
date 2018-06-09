import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import os

import detectron.core.test_engine as infer_engine # NOQA (Must import before import smartcrop.task)

from moviepy.editor import VideoFileClip

from smartcrop.helpers import resize_background_image
from smartcrop.task import run_task


if __name__ == "__main__":

    path = "/smartcrop/data/"

    clip = (
        VideoFileClip(os.path.join(path, 'input', 'clip.mp4'))
        .subclip(0, 1)
    )

    background_image = (
        cv2.imread(os.path.join(path, 'input', 'bkg.jpg'))
    )
    background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)
    background_image = resize_background_image(background_image, clip)

    clip = run_task(clip, background_image)

    clip.write_videofile(os.path.join(path, 'output', 'clip.mp4'))
