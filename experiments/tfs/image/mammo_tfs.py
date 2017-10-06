import numpy as np
import poisson_blending

from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from scipy import ndimage

from skimage.util import random_noise, pad, crop
from skimage.filters import gaussian
from skimage.transform import rotate, rescale, swirl, warp, AffineTransform
from skimage.exposure import adjust_gamma
from skimage import img_as_ubyte, img_as_float
from skimage.color import rgb2hsv, hsv2rgb
from skimage.morphology import dilation, erosion

from PIL import Image, ImageEnhance


def TF_mammo_brightness(x, N=500, target=None): 
    #compute mean of top N pixels
    assert len(x.shape) == 3
    x_new = np.uint8(img_as_ubyte(x))
    topN = sorted(x_new.flatten())[-N:]
    meanN = np.mean(topN)

    #compute upper bound and brightenss enhancement
    upper_bound = 255 / meanN 
    p = np.random.choice(np.linspace(1.0, upper_bound, num=10))
    enhancer = ImageEnhance.Brightness(np_to_pil(x))
    return pil_to_np(enhancer.enhance(p))


def compute_bb(args, num_pixels, upper_bound=100): 
    h0, h1 = min(args[0]), max(args[0])
    w0, w1 = min(args[1]), max(args[1])
    h, w = h1-h0, w1-w0 
    
    h_lo = max(2,h0-num_pixels)
    h_hi = min(upper_bound-2,h1+num_pixels) 
    w_lo = max(2,w0-num_pixels)
    w_hi = min(upper_bound-2,w1+num_pixels) 
    h = h_hi - h_lo
    w = w_hi - w_lo

    center = (h_lo + h / 2, w_lo + w / 2)
    
    return [h_lo, h_hi, w_lo, w_hi], \
        [h0, h1, w0, w1], center


def rotate_image(img, angle, pivot):
    padX = [img.shape[1] - pivot[0], pivot[0]]
    padY = [img.shape[0] - pivot[1], pivot[1]]
    imgP = np.pad(img, [padY, padX], 'constant')
    imgR = ndimage.rotate(imgP, angle, reshape=False)
    return imgR[padY[0] : -padY[1], padX[0] : -padX[1]]


def TF_translate_structure_with_tissue(imsg, translation=None, num_pixels=10, \
    target=None, dim=100):
    #convert format
    #imsg = img_as_float(imsg)
    #imsg = np.uint8(img_as_ubyte(imsg))
    imsg[:,:,0] = img_as_ubyte(imsg[:,:,0])
    
    #translate/rotate/dilate segmentation
    args = np.where(imsg[:,:,1] != 0)
    bb, seg, _ = compute_bb(args, num_pixels, upper_bound=dim)
    h_lo, h_hi, w_lo, w_hi = bb
    h0, h1, w0, w1 = seg

    if not translation: 
        y_translate = np.random.randint(-h0, dim-h1)
        x_translate = np.random.randint(-w0, dim-w1)
    else:
        y_translate, x_translate = translation

        maxa0, mina0 = max(args[0]), min(args[0])
        maxa1, mina1 = max(args[1]), min(args[1])
        if y_translate + maxa0 >= dim: 
            y_translate = dim - maxa0 - 1
        if x_translate + maxa1 >= dim: 
            x_translate = dim - maxa1 - 1 
        if mina0 + y_translate < 0: 
            y_translate = -mina0
        if mina1 + x_translate < 0: 
            x_translate = -mina1
    
    new_seg_args = [args[0]+y_translate, args[1]+x_translate]
    mass_seg = imsg[h_lo:h_hi, w_lo:w_hi, 0]
    new_mask = np.zeros((dim, dim))
    new_mask[h_lo:h_hi,w_lo:w_hi] = 255
    
    #-39,15
    new_im = poisson_blending.blend(target, imsg[:,:,0], \
            new_mask, offset=(y_translate, x_translate))    
    new_seg = np.zeros((dim, dim))
    new_seg[new_seg_args[0],new_seg_args[1]] = 255
    
    im = np.zeros((dim, dim, 2))
    im[:,:,0] = img_as_float(new_im.astype(np.uint8))
    im[:,:,1] = img_as_float(new_seg.astype(np.uint8))

    return img_as_float(im)


def TF_rotate_structure_with_tissue(imsg, p=None, num_pixels=10, \
    target=None, dim=100):
    #convert format
    imsg = np.uint8(img_as_ubyte(imsg))

    #translate/rotate/dilate segmentation
    args = np.where(imsg[:,:,1] != 0)
    bb, _, center = compute_bb(args, num_pixels, upper_bound=dim)
    h_lo, h_hi, w_lo, w_hi = bb
    
    if not p: 
        p = np.random.randint(360)
        
    mass_seg = imsg[h_lo:h_hi, w_lo:w_hi, 0]
    new_mask = np.zeros((dim, dim))
    new_mask[h_lo:h_hi,w_lo:w_hi] = 255
        
    #rotate 
    center_rev = (center[1], center[0])
    rot_im = rotate_image(imsg[:,:,0], p, center_rev)
    rot_mask = rotate_image(imsg[:,:,1], p, center_rev)    
    
    rot_args = np.where(rot_mask)
    rot_bb, _, _ = compute_bb(rot_args, num_pixels, upper_bound=dim)
    rh_lo, rh_hi, rw_lo, rw_hi = rot_bb
    
    new_mask = np.zeros((dim, dim))
    new_mask[rh_lo:rh_hi, rw_lo:rw_hi] = 255
    ##first: normal, second: full original \
    ##that's been transformed, third: mask \
    ##of the transformed
    new_im = poisson_blending.blend(target, rot_im, new_mask)    
    
    im = np.zeros((dim, dim, 2))
    
    im[:,:,0] = img_as_float(new_im.astype(np.uint8))
    im[:,:,1] = img_as_float(rot_mask.astype(np.uint8))

    return im



