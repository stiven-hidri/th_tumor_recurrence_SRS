import torch
import numpy as np
import SimpleITK as sitk
import random

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

def combine_aug(mr, rtd, p_augmentation=.3, augmentations_techinques=['shear', 'gaussian_noise', 'flip', 'rotate' 'brightness']):
    
    augmentations = {
        'shear': shear, 
        'flip': flip, 
        'gaussian_noise':gaussian_noise, 
        'brightness':brightness,
        'rotate':rotate,
    }
        
    if random.random() <= p_augmentation:
        augmentations = [augmentations[a] for a in augmentations_techinques if a in list(augmentations.keys())]
        n_augmentations_to_be_performed = random.randint(1, len(augmentations))
        
        augmentations_to_be_performed = random.sample(augmentations, n_augmentations_to_be_performed)
        
        for aug in augmentations_to_be_performed:
            mr, rtd = aug(mr, rtd)

    return mr, rtd