import torchvision.transforms.functional as TF
from PIL import Image


# define a helper function to resize images
def resize_img_label(image, label=(0., 0.), target_size=(256,256)):
    w_orig, h_orig = image.size
    w_target, h_target = target_size
    cx, cy = label
    image_new = TF.resize(image, target_size)
    label_new = cx/w_orig*w_target, cy/h_orig*h_target
    return image_new, label_new

image = Image.open('./cat.jpg')

resize_image, resize_label =resize_img_label(image,[10.0,10.0],[112,200])

resize_image.save('./cat_resize_ver2.jpg')






