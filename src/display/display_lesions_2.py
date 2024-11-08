import argparse
import matplotlib.pyplot    as plt
import numpy                as np
import pickle
import os

import pywt
from utils import clear_directory_content

def plot(stuff, label, progressive):
    r = len(stuff)  # Number of rows (types of images)
    indexes = np.where(np.sum(stuff[1], axis=(1, 2)) > 0)[0]  # Find non-empty slices more efficiently
    c = 6 if len(indexes) >= 6 else len(indexes)  # Number of columns (slices)
    indexes = np.linspace(min(indexes), max(indexes), c+2, dtype=int)[1:-1]
    labels = ['MRI', 'RTD','WDT_FUSION']
    
    fig = plt.figure(figsize=(13, 8))  # Optional: Smaller figure size
    plt.tight_layout(pad=0, w_pad=0, h_pad=0)  # Adjust values as needed
    
    fig.suptitle(f'Label: {label}', fontsize=12)

    for i, s in enumerate(stuff):
        images = s[indexes]
        for j, img in enumerate(images):
            ax = fig.add_subplot(r, c, i * c + j + 1)  # Create subplot
            ax.axis('off')
            ax.imshow(img)
            if j == 0:
                ax.set_title(f'{labels[i]}', fontsize=10)
    if label == 1:
        plt.savefig(os.path.join(os.path.dirname(__file__), 'img_lesions', 'recurrence', f'figure_{progressive}.png'))
    else:
        plt.savefig(os.path.join(os.path.dirname(__file__), 'img_lesions', 'stable', f'figure_{progressive}.png'))
        
    plt.close()

def display(plt, axes, images, cur_row, c):
    for i in range(c):
        axes[cur_row, i].imshow(images[i])
        axes[cur_row, i].set_axis_off()

def wdt_fusion(mr, rtd):
    wavelet = 'db1'
    coeffs_mr = pywt.dwtn(mr, wavelet, axes=(0, 1, 2))
    coeffs_rtd = pywt.dwtn(rtd, wavelet, axes=(0, 1, 2))

    fused_details_e = {}
    for key in coeffs_mr.keys():
        if key == 'aaa':  # Skip approximation coefficients for energy fusion
            fused_details_e[key] = (coeffs_mr[key]*0.45 + coeffs_rtd[key]*0.55)
        else:
            energy1 = np.abs(coeffs_mr[key]) ** 2
            energy2 = np.abs(coeffs_rtd[key]) ** 2
            fused_details_e[key] = np.where(energy1 > energy2, coeffs_mr[key], coeffs_rtd[key])

    fused_image_e = pywt.idwtn(fused_details_e, wavelet, axes=(0, 1, 2), mode='smooth')
    fused_image_e = fused_image_e * ( mr > 0 )
    return fused_image_e

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', action='store_true')
    args = parser.parse_args()
    if args.raw:
        path_to_data_folder = os.path.join(os.path.dirname(__file__), '..','..','data', 'raw') 
    else:
        path_to_data_folder = os.path.join(os.path.dirname(__file__), '..','..','data', 'processed') 

    clear_directory_content(os.path.join(os.path.dirname(__file__), 'img_lesions'))
    
    os.makedirs(os.path.join(os.path.dirname(__file__), 'img_lesions'), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(__file__), 'img_lesions', 'stable'), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(__file__), 'img_lesions', 'recurrence'), exist_ok=True)
    
    file_names = [f for f in os.listdir(path_to_data_folder) if f.endswith('.pkl')]
    images, labels =  [], []
    
    stats = {
        "rtd": {"min": 0, "max": 0 },
        "mr": {"min": 0, "max": 0 },
    }
    
    for file_split in file_names:
        if file_split.endswith('.pkl'):    
            with open(os.path.join(path_to_data_folder, file_split), 'rb') as input_file:
                data = pickle.load(input_file)    
            for i in range(len(data['mr'])):
                images.append([np.array(data['mr'][i]), np.array(data['rtd'][i])])
                if np.max(data['rtd'][i])>stats['rtd']['max']:
                    stats['rtd']['max'] = np.max(data['rtd'][i])
                if np.min(data['rtd'][i])<stats['rtd']['min']:
                    stats['rtd']['min'] = np.min(data['rtd'][i])
                if np.max(data['mr'][i])>stats['mr']['max']:
                    stats['mr']['max'] = np.max(data['mr'][i])
                if np.min(data['mr'][i])<stats['mr']['min']:
                    stats['mr']['min'] = np.min(data['mr'][i])  
            labels.extend(data['label'])
    
    for i in range(len(images)):
        images[i][0] = (images[i][0] - stats['mr']['min']) / (stats['mr']['max'] - stats['mr']['min'])
        images[i][1] = (images[i][1] - stats['rtd']['min']) / (stats['rtd']['max'] - stats['rtd']['min'])
        images[i].append(wdt_fusion(images[i][0], images[i][1]))
    
    for i in range(10):
        print(f'\r{i+1}/{len(images)}', end='')
        plot(images[i], labels[i], i)
