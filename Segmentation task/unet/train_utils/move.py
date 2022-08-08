import os
import shutil
path = r'E:\Data\Mydataset\all\crop\alls\fold4'
name_list = ['SUB', '剪影']
for set in os.listdir(path):
    if set=='train':
        set_path = os.path.join(path,set)
        for type in os.listdir(set_path):
            type_path = os.path.join(set_path,type)
            for patient in os.listdir(type_path):
                patient_path = os.path.join(type_path,patient)
                for jpg in os.listdir(patient_path):
                    if any(name in jpg for name in name_list) or len(jpg.split("_")[4])==5:
                        jpg_path = os.path.join(patient_path,jpg)
                        shutil.copy(jpg_path,r'E:\Data\Mydataset\segment\training\sub_images')
    else:
        set_path = os.path.join(path, set)
        for type in os.listdir(set_path):
            type_path = os.path.join(set_path, type)
            for patient in os.listdir(type_path):
                patient_path = os.path.join(type_path, patient)
                for jpg in os.listdir(patient_path):
                    if any(name in jpg for name in name_list) or len(jpg.split("_")[4]) == 5:
                        jpg_path = os.path.join(patient_path, jpg)
                        shutil.copy(jpg_path, r'E:\Data\Mydataset\segment\test\sub_images')
print('done')