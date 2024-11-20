import argparse
import pprint
import random
import shutil
import matplotlib.pyplot    as plt
import numpy                as np
import pickle
import os

import pywt

def clear_directory_content(PATH):
    if os.path.exists(PATH):
        for filename in os.listdir(PATH):
            file_path = os.path.join(PATH, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

def plot(stuff, keys, progressive, c_init=6):
    r = len(keys)  # Number of rows (types of images)
    indexes = np.where(np.sum(stuff['mr'][progressive], axis=(1, 2)) > 0)[0]  # Find non-empty slices more efficiently
    
    c = c_init if len(indexes) >= c_init else len(indexes)  # Number of columns (slices)
    indexes = np.linspace(min(indexes), max(indexes), c+2, dtype=int)[1:-1]
    
    fig = plt.figure(figsize=(13, 8))  # Optional: Smaller figure size
    plt.tight_layout(pad=0, w_pad=0, h_pad=0)  # Adjust values as needed
    
    # fig.suptitle(f'Label: {stuff["label"][progressive]}', fontsize=12)

    for i, key in enumerate(keys):
        image = stuff[key][progressive]
        for j in range(c):
            ax = fig.add_subplot(r, c, i * c + j + 1)  # Create subplot
            ax.axis('off')
            ax.imshow(image[indexes[j]])
            if j == 0:
                ax.set_title(f'{key.upper().replace("_", " ")}', fontsize=10)
    
    if stuff["label"][progressive] == 'recurrence':
        plt.savefig(os.path.join(os.path.dirname(__file__), 'img_lesions', 'recurrence', f'figure_{progressive}.png'))
    else:
        plt.savefig(os.path.join(os.path.dirname(__file__), 'img_lesions', 'stable', f'figure_{progressive}.png'))
        
    plt.close()

def wdt_fusion(mr, rtd):
    wavelet = 'db1'
    coeffs_mr = pywt.dwtn(mr, wavelet, axes=(0, 1, 2))
    coeffs_rtd = pywt.dwtn(rtd, wavelet, axes=(0, 1, 2))

    fused_details_e = {}
    for key in coeffs_mr.keys():
        if key == 'aaa':  # Skip approximation coefficients for energy fusion
            fused_details_e[key] = coeffs_rtd[key]*.45 + coeffs_mr[key]*.55
        else:
            energy1 = np.abs(coeffs_mr[key]) ** 2
            energy2 = np.abs(coeffs_rtd[key]) ** 2
            fused_details_e[key] = np.where(energy1 > energy2, coeffs_mr[key], coeffs_rtd[key])

    fused_image_e = pywt.idwtn(fused_details_e, wavelet, axes=(0, 1, 2), mode='smooth')
    fused_image_e = fused_image_e
    return fused_image_e

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', action='store_true')
    args = parser.parse_args()
    
    path_to_data_folder = os.path.join(os.path.dirname(__file__), '..','..','data', 'processed') 
    path_to_data = os.path.join(path_to_data_folder, 'global_data.pkl')

    clear_directory_content(os.path.join(os.path.dirname(__file__), 'img_lesions'))
    
    os.makedirs(os.path.join(os.path.dirname(__file__), 'img_lesions'), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(__file__), 'img_lesions', 'stable'), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(__file__), 'img_lesions', 'recurrence'), exist_ok=True)
    
    images, labels =  [], []
    
    stats = {
        "rtd": {"min": 0, "max": 0 },
        "mr": {"min": 0, "max": 0 },
        "wdt_fusion": {"min": 0, "max": 0 }
    }

    with open(path_to_data, 'rb') as input_file:
        data = pickle.load(input_file)    
        
    data['wdt_fusion'] = []
                
    for i in range(len(data['mr'])):
        if np.max(data['rtd'][i])>stats['rtd']['max']:
            stats['rtd']['max'] = np.max(data['rtd'][i])
        if np.min(data['rtd'][i])<stats['rtd']['min']:
            stats['rtd']['min'] = np.min(data['rtd'][i])
        if np.max(data['mr'][i])>stats['mr']['max']:
            stats['mr']['max'] = np.max(data['mr'][i])
        if np.min(data['mr'][i])<stats['mr']['min']:
            stats['mr']['min'] = np.min(data['mr'][i])
    
    for i in range(len(data['mr'])):
        scaled_mr = (data['mr'][i] - stats['mr']['min']) / (stats['mr']['max'] - stats['mr']['min'])
        scaled_rtd = (data['rtd'][i] - stats['rtd']['min']) / (stats['rtd']['max'] - stats['rtd']['min'])
        wdtf = wdt_fusion(scaled_mr, scaled_rtd)
        wdtf = (wdtf - wdtf.min()) / (wdtf.max() - wdtf.min())
        data['wdt_fusion'].append(wdtf)
        if np.max(data['wdt_fusion'][i])>stats['wdt_fusion']['max']:
            stats['wdt_fusion']['max'] = np.max(data['wdt_fusion'][i])
        if np.min(data['wdt_fusion'][i])<stats['wdt_fusion']['min']:
            stats['wdt_fusion']['min'] = np.min(data['wdt_fusion'][i])
            
    pprint.pprint(stats['wdt_fusion'])
    
    keys = ['mr', 'rtd', 'wdt_fusion']
    
    sample_indexes = list(random.sample(range(len(data['mr'])), k=10))
    
    for i in sample_indexes:
        print(f'\r{i+1}/{len(sample_indexes)}', end='')
        plot(data, keys, i)
