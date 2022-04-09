import numpy as np
from PIL import Image
from io import BytesIO
from PIL import Image
import torch
import torchvision.transforms as transforms


def flip_if_vertical(image):
    height = image.shape[0]
    width = image.shape[1]
    if height >= (4*width/3.0):
        image = torch.nn.functional.pad(torch.rot90(image), (0,0,4,4,0,0), mode="constant", value=0.5)
    return image


def resize_and_pad(image, desired_output_size,
                   random_scale_min=0.1, random_scale_max=2.0, do_random_scale=False,
                   shrink_both_sides=True,
                   do_flip_if_vertical=True,
                   resize_method="random"):
    if do_flip_if_vertical:  # padding
        image = flip_if_vertical(image)
    desired_height = torch.tensor(desired_output_size[0], dtype=float)
    desired_width = torch.tensor(desired_output_size[1], dtype=float)

    height = torch.tensor(image.shape[0], dtype=float)  # 288 -> 192
    width = torch.tensor(image.shape[1], dtype=float)  # 512 -> 320

    if do_random_scale:
        random_scale_factor = torch.rand([]) * (random_scale_max - random_scale_min) + random_scale_min
        if not shrink_both_sides:
            rsf_max = torch.maximum(desired_width / width, desired_height / height)
            random_scale_factor = torch.minimum(rsf_max, random_scale_factor)

        scaled_y = (random_scale_factor * desired_height).int()  # 165
        scaled_x = (random_scale_factor * desired_width).int()  # 276

        image_scale_y = scaled_y.float() / height
        image_scale_x = scaled_x.float() / width
        image_scale = torch.minimum(image_scale_x, image_scale_y)  # 0.5391
        image_scale = torch.maximum(image_scale, 64.0 / torch.minimum(height, width))

        scaled_height = (height * image_scale)  # 64
        scaled_width = (width * image_scale)  # 113

        offset_y = (scaled_height - desired_height).int()
        offset_x = (scaled_width - desired_width).int()
        offset_y = (torch.maximum(torch.zeros(1), offset_y.clone().detach()) * torch.rand([])).int().item()
        offset_x = (torch.maximum(torch.zeros(1), offset_x.clone().detach()) * torch.rand([])).int().item()

    else:
        image_scale_y = desired_height / height
        image_scale_x = desired_width / width
        image_scale = torch.minimum(image_scale_x, image_scale_y)
        scaled_height = (height * image_scale)
        scaled_width = (width * image_scale)
        offset_y = torch.zeros(1)
        offset_x = torch.zeros(1)

    # resize and crop
    if resize_method == "random" and do_random_scale:  # tensorflow에서 지원하는 resize 방법 일부가 torch에서 지원 안됨
        image = image.view(image.shape[2], image.shape[0], image.shape[1]).float()
        image = transforms.Resize((scaled_height, scaled_width), antialias=True)(image)  # 0~255 사이값으로 텐서값 변경,
        image /= torch.max(image)
        image = image.view(image.shape[1], image.shape[2], image.shape[0])

    image = torch.clamp(image, 0.0, 1.0)
    image = image[offset_y:offset_y + desired_output_size[0],
            offset_x:offset_x + desired_output_size[1], :]
    if image.shape[0] != desired_output_size[0]:
        n_pad = desired_output_size[0] - image.shape[0]
        image = torch.cat((image, torch.zeros(n_pad, image.shape[1], image.shape[2])), dim=0)

    if image.shape[1] != desired_output_size[1]:
        n_pad = desired_output_size[1] - image.shape[1]
        image = torch.cat((image, torch.zeros(image.shape[0], n_pad, image.shape[2])), dim=1)

    effective_height = torch.minimum(scaled_height, desired_height)
    effective_width = torch.minimum(scaled_width, desired_width)
    image_info = torch.stack([effective_height.float() / desired_height,
                              effective_width.float() / desired_width,
                              1.0 / image_scale,
                              height,
                              width,
                              offset_y / height,
                              offset_x / width])
    return image, image_info

def load_and_resize_img(encoded_jpg, config):
    
    P = config["vit_patch_size"]
    h1, w1 = config["output_grid"]
    
    img = torch.tensor(np.array(Image.open(BytesIO(encoded_jpg))), dtype=torch.float32)
    
    img, this_image_info = resize_and_pad(img, (h1*P, w1*P),
                                         do_random_scale=config.get('do_random_scale', True),
                                         random_scale_max=config.get('random_scale_max', 1.1),
                                         random_scale_min=config.get('random_scale_min', 1.05),
                                         shrink_both_sides=config.get('shrink_both_sides', True),
                                         do_flip_if_vertical=config.get('do_flip_if_vertical', True),
                                         resize_method="random")

    img = torch.nn.functional.pixel_shuffle(img[None], int(np.sqrt(P)))
    img = img.reshape(h1*w1, P*P*3)
    return img

def get_size_for_resize(image_size, shorter_size_trg=384, longer_size_max=512):
    """
    Gets a new size for the image. We will try to make it such that the bigger size is less than
    longer_size_max. However, we won't resize it if its shortest side is <= shorter_size_trg.
    :param image_size:
    :param shorter_size_trg:
    :param longer_size_max:
    :return:
    """

    w, h = image_size
    size = shorter_size_trg  # Try [size, size]

    if min(w, h) <= size:
        return w, h

    min_original_size = float(min((w, h)))
    max_original_size = float(max((w, h)))
    if max_original_size / min_original_size * size > longer_size_max:
        size = int(round(longer_size_max * min_original_size / max_original_size))

    if (w <= h and w == size) or (h <= w and h == size):
        return w, h
    if w < h:
        ow = size
        oh = int(size * h / w)
    else:
        oh = size
        ow = int(size * w / h)
    return ow, oh

def resize_image(image, shorter_size_trg=384, longer_size_max=512):
    """
    Resize image such that the longer size is <= longer_size_max.
    Gets a new size for the image. We will try to make it such that the bigger size is less than
    longer_size_max. However, we won't resize it if its shortest side is <= shorter_size_trg.
    :param image:
    :param shorter_size_trg:
    :param longer_size_max:
    """
    trg_size = get_size_for_resize(image.size, shorter_size_trg=shorter_size_trg,
                                       longer_size_max=longer_size_max)
    if trg_size != image.size:
        return image.resize(trg_size, resample=Image.BICUBIC)
    return image

def pil_image_to_jpgstring(image: Image, quality=95):
    """
    :param image: PIL image
    :return: it, as a jpg string
    """
    with BytesIO() as output:
        image.save(output, format='JPEG', quality=quality, optimize=True)
        return output.getvalue()
