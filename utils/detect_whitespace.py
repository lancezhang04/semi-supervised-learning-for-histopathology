import numpy as np
import cv2
from skimage.filters import threshold_otsu, gaussian
from scipy import ndimage
from PIL import Image


Image.MAX_IMAGE_PIXELS = None


def detect_whitespace(rgb, region_mask, whitespace_code):
    """
    Refine manual mask by detecting whitespace using color thresholds.
    Adapted from histomicstk / saliency / cellularity_detection_thresholding.
    """
    lab, _ = threshold_multichannel(
        im=rgb_to_hsi(rgb),
        channels=['hue', 'saturation', 'intensity'],
        thresholds={
            'hue': {'min': 0, 'max': 1.0},
            'saturation': {'min': 0, 'max': 0.2},
            'intensity': {'min': 220, 'max': 255},
        },
        just_threshold=False,
    )

    region_mask[lab > 0] = whitespace_code

    return region_mask


def threshold_multichannel(
        im, thresholds, channels=None,
        just_threshold=False, get_tissue_mask_kwargs=None):
    """Threshold a multi-channel image (eg. HSI image) to get tissue.

    The relies on the fact that oftentimes some slide elements (eg blood
    or whitespace) have a characteristic hue/saturation/intensity. This
    thresholds along each HSI channel, then optionally uses the
    get_tissue_mask() method (gaussian smoothing, otsu thresholding,
    connected components) to get each contiguous tissue piece.

    Parameters
    -----------
    im : np array
        (m, n, 3) array of Hue, Saturation, Intensity (in this order)
    thresholds : dict
        Each entry is a dict containing the keys min and max
    channels : list
        names of channels, in order (eg. hue, saturation, intensity)
    just_threshold : bool
        if Fase, get_tissue_mask() is used to smooth result and get regions.
    get_tissue_mask_kwargs : dict
        key-value pairs of parameters to pass to get_tissue_mask()

    Returns
    --------
    np int32 array
        if not just_threshold, unique values represent unique tissue regions
    np bool array
        if not just_threshold, largest contiguous tissue region.

    """
    channels = ['hue', 'saturation', 'intensity'] if channels is None else channels

    if get_tissue_mask_kwargs is None:
        get_tissue_mask_kwargs = {
            'n_thresholding_steps': 1,
            'sigma': 5.0,
            'min_size': 10,
        }

    # threshold each channel
    mask = np.ones(im.shape[:2])
    for ax, ch in enumerate(channels):

        channel = im[..., ax].copy()

        mask[channel < thresholds[ch]['min']] = 0
        mask[channel >= thresholds[ch]['max']] = 0

    # smoothing, otsu thresholding then connected components
    if just_threshold or (np.unique(mask).shape[0] < 1):
        labeled = mask
    else:
        get_tissue_mask_kwargs['deconvolve_first'] = False
        labeled, mask = get_tissue_mask(mask, **get_tissue_mask_kwargs)

    return labeled, mask


def rgb_to_hsi(im):
    """Convert to HSI the RGB pixels in im.

    Adapted from
    https://en.wikipedia.org/wiki/HSL_and_HSV#Hue_and_chroma.

    """
    im = np.moveaxis(im, -1, 0)
    if len(im) not in (3, 4):
        raise ValueError("Expected 3-channel RGB or 4-channel RGBA image;"
                         " received a {}-channel image".format(len(im)))
    im = im[:3]
    hues = (np.arctan2(3**0.5 * (im[1] - im[2]),
                       2 * im[0] - im[1] - im[2]) / (2 * np.pi)) % 1
    intensities = im.mean(0)
    saturations = np.where(
        intensities, 1 - im.min(0) / np.maximum(intensities, 1e-10), 0)
    return np.stack([hues, saturations, intensities], -1)


def get_tissue_mask(
        thumbnail_im,
        deconvolve_first=False, stain_unmixing_routine_kwargs=None,
        n_thresholding_steps=1, sigma=0., min_size=500):
    """Get binary tissue mask from slide thumbnail.

    Parameters
    -----------
    thumbnail_im : np array
        (m, n, 3) nd array of thumbnail RGB image
        or (m, n) nd array of thumbnail grayscale image
    deconvolve_first : bool
        use hematoxylin channel to find cellular areas?
        This will make things ever-so-slightly slower but is better in
        getting rid of sharpie marker (if it's green, for example).
        Sometimes things work better without it, though.
    stain_matrix_method : str
        see deconv_color method in seed_utils
    n_thresholding_steps : int
        number of gaussian smoothign steps
    sigma : float
        sigma of gaussian filter
    min_size : int
        minimum size (in pixels) of contiguous tissue regions to keep

    Returns
    --------
    np int32 array
        each unique value represents a unique tissue region
    np bool array
        largest contiguous tissue region.

    """
    stain_unmixing_routine_kwargs = (
        {} if stain_unmixing_routine_kwargs is None else stain_unmixing_routine_kwargs)

    if deconvolve_first and (len(thumbnail_im.shape) == 3):
        raise NotImplementedError()

    elif len(thumbnail_im.shape) == 3:
        # grayscale thumbnail (inverted)
        thumbnail = 255 - cv2.cvtColor(thumbnail_im, cv2.COLOR_BGR2GRAY)

    else:
        thumbnail = thumbnail_im

    for _ in range(n_thresholding_steps):

        # gaussian smoothing of grayscale thumbnail
        if sigma > 0.0:
            thumbnail = gaussian(
                thumbnail, sigma=sigma,
                output=None, mode='nearest', preserve_range=True)

        # get threshold to keep analysis region
        try:
            thresh = threshold_otsu(thumbnail[thumbnail > 0])
        except ValueError:  # all values are zero
            thresh = 0

        # replace pixels outside analysis region with upper quantile pixels
        thumbnail[thumbnail < thresh] = 0

    # convert to binary
    mask = 0 + (thumbnail > 0)

    # find connected components
    labeled, _ = ndimage.label(mask)

    # only keep
    unique, counts = np.unique(labeled[labeled > 0], return_counts=True)
    discard = np.in1d(labeled, unique[counts < min_size])
    discard = discard.reshape(labeled.shape)
    labeled[discard] = 0

    # largest tissue region
    mask = labeled == unique[np.argmax(counts)]

    return labeled, mask
