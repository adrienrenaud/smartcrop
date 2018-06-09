import cv2

def resize_background_image(background_image, clip):
    clip_image = clip.get_frame(1)
    clip_shape = (clip_image.shape[1], clip_image.shape[0])
    background_image = cv2.resize(background_image, dsize=clip_shape)
    return background_image
