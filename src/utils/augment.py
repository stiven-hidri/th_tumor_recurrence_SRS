import torch
import numpy as np
import SimpleITK as sitk
import random
import torch.nn.functional as F
from torchvision.transforms import RandomAffine
from scipy.ndimage import affine_transform


def gaussian_noise(mr, rtd):
    """
    Add gaussian noise to the image
    """
    mask = mr == 0
    noise = torch.randn(mr.size()) * .02 + 0.0
    
    mr, rtd = mr + noise, rtd + noise
    mr[mask] = 0
    rtd[mask] = 0
    return mr, rtd

def brightness(mr, rtd):
    """
    Changing the brighness of a image using power-law gamma transformation.
    Gain and gamma are chosen randomly for each image channel.
    
    Gain chosen between [0.8 - 1.2]
    Gamma chosen between [0.8 - 1.2]
    
    new_im = gain * im^gamma
    """
    
    mr, rtd = mr.numpy(), rtd.numpy()
    
    mr_new, rtd_new = np.zeros(mr.shape), np.zeros(rtd.shape)
    
    gain, gamma = (1.5-.5) * np.random.random_sample(2,) + .5
    
    mr_new = np.sign(mr) * gain * (np.abs(mr)**gamma)
    rtd_new = np.sign(rtd) * gain * (np.abs(rtd)**gamma)
    
    mr_new = torch.Tensor(mr_new).to(torch.float32)
    rtd_new = torch.Tensor(rtd_new).to(torch.float32)

    return mr_new, rtd_new

def shear(mr, rtd):
    # Step 1: Convert tensor to SimpleITK Image
    mr, rtd = sitk.GetImageFromArray(mr), sitk.GetImageFromArray(rtd)

    # Step 2: Generate random shearing factors for a random axis
    shear_factors = [random.uniform(-1., 1.) for _ in range(2)]  # Random shear values
    axis = random.choice(['xy', 'xz', 'yz'])

    # Step 3: Create the shear transform
    shear_transform = sitk.AffineTransform(3)  # 3D affine transform

    if axis == 'xy':
        shear_transform.SetMatrix([1, shear_factors[0], 0, shear_factors[1], 1, 0, 0, 0, 1])
    elif axis == 'xz':
        shear_transform.SetMatrix([1, 0, shear_factors[0], 0, 1, 0, shear_factors[1], 0, 1])
    elif axis == 'yz':
        shear_transform.SetMatrix([1, 0, 0, 0, 1, shear_factors[0], 0, shear_factors[1], 1])

    # Step 4: Apply the shear transformation
    center = mr.TransformContinuousIndexToPhysicalPoint([(size-1)/2.0 for size in mr.GetSize()])
    shear_transform.SetCenter(center)
    mr, rtd = sitk.Resample(mr, shear_transform, sitk.sitkLinear, 0, mr.GetPixelID()), sitk.Resample(rtd, shear_transform, sitk.sitkLinear, 0, rtd.GetPixelID())

    # Step 6: Convert back to tensor (numpy array)
    mr = torch.Tensor(sitk.GetArrayFromImage(mr)).to(torch.float32)
    rtd = torch.Tensor(sitk.GetArrayFromImage(rtd)).to(torch.float32)
    
    return mr, rtd

def flip(mr, rtd):
    axis = random.choice([0, 1, 2])
    
    mr = torch.flip(mr, dims=[axis])
    rtd = torch.flip(rtd, dims=[axis])
        
    return mr, rtd

def rotate(mr, rtd):
    axis = random.choice([(1,2), (0,2), (0,1)]) # x, y, z 
    k = np.random.choice([1, 2, 3])
    
    mr = torch.rot90(mr, k=k, dims=axis)
    rtd = torch.rot90(rtd, k=k, dims=axis)
        
    return mr, rtd

def random_translate(mr, rtd, translate_range=(-10, 10)):
    # Check if input tensors have batch dimension
    if len(mr.shape) == 4:  # Shape: (C, D, H, W)
        N = 1  # Treat as a single sample
        C, D, H, W = mr.shape
        mr = mr.unsqueeze(0)  # Add batch dimension
        rtd = rtd.unsqueeze(0)  # Add batch dimension
    elif len(mr.shape) == 5:  # Shape: (N, C, D, H, W)
        N, C, D, H, W = mr.shape
    elif len(mr.shape) == 3:  # Shape: (D, H, W)
        mr = mr.unsqueeze(0)  # Add batch dimension
        mr = mr.unsqueeze(0)  # Add channel dimension
        rtd = rtd.unsqueeze(0)  # Add batch dimension
        rtd = rtd.unsqueeze(0)  # Add channel dimension
        
        N, C, D, H, W = mr.shape
    else:
        raise ValueError("Input tensors must have shape (N, C, D, H, W) or (C, D, H, W).")
    
    tx = random.randint(translate_range[0], translate_range[1])
    ty = random.randint(translate_range[0], translate_range[1])
    tz = random.randint(translate_range[0], translate_range[1])
    
    mr = F.pad(mr, (abs(tx), abs(tx), abs(ty), abs(ty), abs(tz), abs(tz)), mode='constant', value=0)
    rtd = F.pad(rtd, (abs(tx), abs(tx), abs(ty), abs(ty), abs(tz), abs(tz)), mode='constant', value=0)
    
    mr = mr[:, :, abs(tz) + tz : abs(tz) + tz + D, abs(ty) + ty : abs(ty) + ty + H, abs(tx) + tx : abs(tx) + tx + W]
    rtd = rtd[:, :, abs(tz) + tz : abs(tz) + tz + D, abs(ty) + ty : abs(ty) + ty + H, abs(tx) + tx : abs(tx) + tx + W]
    
    mr = mr.squeeze()  
    rtd = rtd.squeeze()
    
    return mr, rtd

import torchio as tio

def random_affine(mr, rtd):
    
    mr = mr.unsqueeze(0)
    rtd = rtd.unsqueeze(0)
    
    subject1 = tio.Subject(image=tio.ScalarImage(tensor=mr))
    subject2 = tio.Subject(image=tio.ScalarImage(tensor=rtd))

    # Create a random rotation transform
    deg = random.randint(-30, 30)
    transl = random.randint(-5, 5)
    scale = random.uniform(0.85, 1.15)
    
    transform = tio.RandomAffine(degrees=(deg, deg), translation=(transl, transl), scales=(scale, scale), isotropic=True, image_interpolation='linear')
    #['nearest', 'linear', 'bspline', 'gaussian', 'label_gaussian', 'hamming', 'cosine', 'welch', 'lanczos', 'blackman']
    # Apply the transformation to both subjects
    transformed_subject1 = transform(subject1)
    transformed_subject2 = transform(subject2)

    # Retrieve the rotated tensors
    mr_rotated = transformed_subject1['image'].data
    rtd_rotated = transformed_subject2['image'].data
    
    mr_rotated = mr_rotated.squeeze()
    rtd_rotated = rtd_rotated.squeeze()

    return mr_rotated, rtd_rotated

def combine_aug(mr, rtd, p_augmentation=.3, augmentations_techinques=['shear', 'gaussian_noise', 'flip', 'rotate' 'brightness', 'random_translate', 'random_rotate_3d']):
    
    augmentations = {
        'shear': shear, 
        'flip': flip, 
        'gaussian_noise':gaussian_noise, 
        'brightness':brightness,
        'rotate':rotate,
        'random_translate':random_translate,
        'random_affine':random_affine
    }
        
    if random.random() <= p_augmentation:
        augmentations = [augmentations[a] for a in augmentations_techinques if a in list(augmentations.keys())]
        n_augmentations_to_be_performed = random.randint(1, len(augmentations))
        
        augmentations_to_be_performed = random.sample(augmentations, n_augmentations_to_be_performed)
        
        for aug in augmentations_to_be_performed:
            mr, rtd = aug(mr, rtd)

    return mr, rtd