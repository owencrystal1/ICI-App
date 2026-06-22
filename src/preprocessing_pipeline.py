import warnings
warnings.filterwarnings("ignore")
import cv2
import glob
import numpy
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import os
import numpy as np
import pydicom.pixel_data_handlers.util as util
import pydicom
from tqdm import tqdm
import pandas as pd
import pickle


def dicom2roi(dcm_path, output_path, binary_mask):
    dicom_file = pydicom.dcmread(dcm_path)

    pt_ids = []
    dates = []
    acc_nums = []
    final_paths = []

    file_name = dcm_path.rsplit('/',1)[-1]
    file_name = file_name.split('.dcm')[0]

    try:
        pt_id = dicom_file.PatientID
    except Exception as e:
        pt_id = ''
        print(f"PatientID not found: {e}")

    try:
        acq_date = dicom_file.AcquisitionDateTime
    except Exception as e:
        acq_date = ''
        print(f"Acquisition DateTime not found: {e}")
    
    try:
        acc_num = dicom_file.AccessionNumber
    except Exception as e:
        acc_num = ''
        print(f"Accession Number not found: {e}")

    # accession number, acquisition datetime - try except for these
    

    image_array = dicom_file.pixel_array

    rgb_pixel_array = util.convert_color_space(image_array,'YBR_FULL_422','RGB')

    for i in range(rgb_pixel_array.shape[0]):

        final_folder_path = os.path.join(output_path, acc_num + '_' + file_name)

        if not os.path.exists(final_folder_path):
            os.makedirs(final_folder_path)
        elif os.path.exists(final_folder_path) and i == 0:
            print('File already exists')
            break


        pt_ids.append(pt_id)
        dates.append(acq_date)
        acc_nums.append(acc_num)

        plt.imshow(rgb_pixel_array[i], cmap='gray')
        frame = rgb_pixel_array[i]
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)


        #COLOR CORRECTION STARTS HERE

        if frame.shape[1] <= 640:

            kernel = numpy.ones((2,1),numpy.uint8)

        elif frame.shape[1] <= 1080:

            kernel = numpy.ones((3,2),numpy.uint8)

        else:

            kernel = numpy.ones((4,4),numpy.uint8)

        lower_yellow = numpy.array([15, 25, 100])
        upper_yellow = numpy.array([45, 255, 220])
        lower_blue = numpy.array([45, 35, 0])
        upper_blue = numpy.array([165, 255, 255])
        low_y = numpy.full((10,10,3), lower_yellow, dtype = numpy.uint8)/255.0
        high_y = numpy.full((10,10,3), upper_yellow, dtype = numpy.uint8)/255.0
        low_b = numpy.full((10,10,3), lower_blue, dtype = numpy.uint8)/255.0
        high_b = numpy.full((10,10,3), upper_blue, dtype = numpy.uint8)/255.0

        mask_y = cv2.inRange(frame_hsv, lower_yellow, upper_yellow)
        mask_y = cv2.morphologyEx(mask_y, cv2.MORPH_CLOSE, kernel)
        mask_y = cv2.bitwise_not(mask_y)

        mask_b = cv2.inRange(frame_hsv, lower_blue, upper_blue)
        mask_b = cv2.morphologyEx(mask_b, cv2.MORPH_CLOSE, kernel)
        mask_b = cv2.bitwise_not(mask_b)

        mask_composite = mask_y & mask_b
        mask_composite = cv2.bitwise_not(mask_composite)
        #mask_composite = cv2.morphologyEx(mask_composite, cv2.MORPH_OPEN, kernel)
        mask_composite = cv2.dilate(mask_composite,kernel,iterations = 1)

        frame_colour_corrected = numpy.dstack((frame, mask_composite))
        frame_colour_corrected = Image.fromarray(frame_colour_corrected)
        
        frame_inpainted = cv2.inpaint(frame, mask_composite, 3, cv2.INPAINT_TELEA)  ###################################### SLOW ###########################################
        
        frame = Image.fromarray(frame_inpainted).convert('L')

        png_path = (final_folder_path + '/{}_{}_{}.png'.format(acc_num, file_name, i))

        final_paths.append(png_path)

        target_shape = (456, 456)

        frame = frame.resize(target_shape, Image.LANCZOS)

        frame = numpy.array(frame)

        frame_output = frame*binary_mask

        frame_output = Image.fromarray(frame_output)

        frame_output.save(png_path) 
        


    print('Saved', i, 'total frames')

    return pt_ids, dates, acc_nums, final_paths

# clin_sheet_path = input('Enter path to clinical sheet (.xlsx file) containing resting gradients and patient IDs: ')
# clin_sheet = pd.read_excel(clin_sheet_path)
# clin_sheet['Accession number'] = clin_sheet['Accession number'].astype(str)


save_path = '/app/tmp'

echo_mrns = []
echo_dates = []
echo_acc_nums = []
echo_paths = []
views = []
labels = []

binary_array = np.load('/app/src/mask_crop.npy')

dicom_data = pd.read_csv(save_path + '/viewclf_dicoms.csv')


dcm_list = dicom_data['file']

for i in tqdm(range(len(dcm_list)), desc='Converting DICOM to PNG for A4C echo data'):


    # convert DICOM file into png files containing conical ROIs (saved in png path)
    pt_ids, dates, acc_num, paths = dicom2roi(dcm_list[i], save_path, binary_array)
    view = [dicom_data['view'][i]]*len(pt_ids)
    pt_id = acc_num[0]

    # if str(pt_id) in clin_sheet['Accession number'].values:
    #     index = clin_sheet[clin_sheet['Accession number'] == str(pt_id)].index[0] # make sure data type matches that in clinical sheet
    #     rest_grad = clin_sheet['Resting Gradient'][index]
    #     if rest_grad >= 20:
    #         label = 1
    #     else:
    #         label = 0
    # else:
    #     print(pt_id, 'not found in clinical sheet.')
    #     continue
    
    #label_list = [label]*len(pt_ids)
    
    echo_mrns  = echo_mrns + pt_ids
    echo_dates  = echo_dates + dates
    echo_acc_nums = echo_acc_nums + acc_num
    echo_paths = echo_paths + paths
    views = views + view
    #labels = labels + label_list


print('ROIs extracted from', i, 'DICOM files and saved as PNG files.')
    
# create df with paths and IDs to save as pickle file (if needed to be saved)
echo_data = pd.DataFrame({
    'path': echo_paths,
    'pt_id': echo_mrns,
    'AccessionNumber': echo_acc_nums,
    'AcquisitionDate': echo_dates,
    'view': views
    #'label': labels
    })

test_data = {
    'test': echo_data
}

with open(save_path + '/test_data.pkl', 'wb') as file:
    pickle.dump(test_data, file)
