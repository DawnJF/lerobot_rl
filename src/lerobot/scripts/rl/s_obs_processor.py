from PIL import Image
import numpy as np


def process_image(img: Image, left, top, right, bottom, new_size=None):
    """
    裁剪单张图片并保存为新文件

    参数:
    - left: int, 裁剪区域的左边界
    - top: int, 裁剪区域的上边界
    - right: int, 裁剪区域的右边界
    - bottom: int, 裁剪区域的下边界
    """

    # 裁剪图片
    size = img.size
    img = img.crop((left, top, right, bottom))

    # 设置新的尺寸（例如，宽度=800，高度=600）
    if new_size:
        img = img.resize(new_size, Image.BICUBIC)
    return img


def process_image_npy_i(npy, o_size, left, top, right, bottom, re_size):
    try:
        img = Image.fromarray(npy)
        assert o_size == img.size
        img = process_image(img, left, top, right, bottom, re_size)
        return np.array(img)
    except Exception as e:
        print(f"发生错误: {e}")


def process_image_npy(npy, type, image_params):
    params = image_params

    left, top, right, bottom = params[type]["crop"]

    return process_image_npy_i(
        npy, params[type]["size"], left, top, right, bottom, params[type]["resize"]
    )


image_params = {
    "rgb": {
        "size": (960, 540),
        "crop": (230, 0, 770, 540),
        "resize": (128, 128),
    },
    "wrist": {
        "size": (640, 480),
        "crop": (80, 0, 560, 480),
        "resize": (128, 128),
    },
    "scene": {
        "size": (640, 480),
        "crop": (100, 160, 540, 480),
        "resize": (128, 128),
    },
}


def image_processing_temp(image_np, key):
    """
    临时处理函数，处理图像数据
    """
    if key not in image_params:
        raise ValueError(f"Unsupported image type: {key}")

    return process_image_npy(image_np, key, image_params)
