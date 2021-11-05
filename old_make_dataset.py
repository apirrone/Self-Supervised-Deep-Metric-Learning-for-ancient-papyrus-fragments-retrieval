import numpy as np
import cv2
import os
import random
import patch_utils
import argparse
from progress import Progress
import re
import pickle
import backtrace
backtrace.hook(
    reverse=False,
    align=True,
    strip_path=True,
    enable_on_envvar_only=False,
    on_tty=False,
    conservative=False,
    styles={})

argParser = argparse.ArgumentParser(description='Training')
argParser.add_argument('-d', '--datasetDir', type=str, required=False, default='./',help="where to save the dataset")
argParser.add_argument('-n', '--name', type=str, required=True, help="name of the dataset")
args = argParser.parse_args()


dataset_path = os.path.join(args.datasetDir, args.name)
train_val_dir = os.path.join(dataset_path, "train_val")
test_dir      = os.path.join(dataset_path, "test")

os.system("mkdir -p "+dataset_path)
os.system("mkdir -p "+train_val_dir)
os.system("mkdir -p "+test_dir)

def get_papyrus_id(fragment, michigan=True, fragmentAsClass=False):

  if fragmentAsClass:
    return fragment
  
  if michigan :
    papyrus_id = fragment
    papyrus_id = papyrus_id.split('_')[0]
      
    tmp = re.search('[A-z]', papyrus_id)
    
    if tmp is not None:
      indexFirstCharacter = re.search('[A-z]', papyrus_id).start()
      papyrus_id = papyrus_id[:indexFirstCharacter]
        
  else:

    papyrus_id = fragment.split('_')[1]      

  return papyrus_id



# Dataset parameters
# ===================================
nb_papyri                = 417
patch_size               = 64
dataset_name             = args.name
# extension                = ".png"
# extension                = ".jpg"
extension                = ".JPG"
# data_path                = "/data2/apirrone/michigan_base_2/"
data_path                = "/data2/apirrone/geshaem_segmented_resized/cl_only_cleaned/recto_with_recovered/"
# data_path                = "/data2/apirrone/hisFrag2/hisfrag20_test/"
test_proportion          = 0 # proportion of the dataset put aside for testing (independent from train/validation)
# test_proportion          = 0.9 # proportion of the dataset put aside for testing (independent from train/validation)
max_patches_per_fragment = 5
gray_scale               = False
michigan                 = False
oneBigPatch              = False
fragmentAsClass          = True
geshaem                  = True
resize                   = None
# =====================================

nbChannels = 1 if gray_scale else 3
imShape = (patch_size, patch_size, nbChannels)

nb_test_papyri      = int(test_proportion*nb_papyri)
nb_train_val_papyri = nb_papyri-nb_test_papyri

filesPaths = []

for file in os.listdir(data_path):
  if geshaem:    
    if(os.path.isfile(os.path.join(data_path, file)) and file.endswith(extension)) and "infered" not in file and "CL" in file and "mask" not in file and 'r' in file:
      filesPaths.append(os.path.join(data_path, file))
      print("LOADING\t:\t", os.path.join(data_path, file))      
  else:
    if(os.path.isfile(os.path.join(data_path, file)) and file.endswith(extension)) and "infered" not in file and ('r' in file or not michigan):
      filesPaths.append(os.path.join(data_path, file))
      print("LOADING\t:\t", os.path.join(data_path, file))
        
print("len(filesPaths) : ", len(filesPaths))
random.shuffle(filesPaths)
# exit()

tmp_papyri_train_val = {}
for f in filesPaths:
  extension = f.split('.')[-1]
  file_name = f.split('/')[-1][:-(len(extension)+1)]
  papyrus_id = get_papyrus_id(file_name, michigan=michigan, fragmentAsClass=fragmentAsClass)

  if papyrus_id not in tmp_papyri_train_val:
    tmp_papyri_train_val[papyrus_id] = []
  tmp_papyri_train_val[papyrus_id].append(f)

papyri_train_val = {}
for papyrus_id, fragments in tmp_papyri_train_val.items():
  papyri_train_val[papyrus_id] = fragments

tmp_papyri_test = {}
for f in filesPaths:
  extension = f.split('.')[-1]
  file_name = f.split('/')[-1][:-(len(extension)+1)]
  papyrus_id = get_papyrus_id(file_name, michigan=fragmentAsClass)

  if papyrus_id not in tmp_papyri_test:
    tmp_papyri_test[papyrus_id] = []
  tmp_papyri_test[papyrus_id].append(f)

nb_fragments = 0
  
papyri_test = {}
for papyrus_id, fragments in tmp_papyri_test.items():
  papyri_test[papyrus_id] = fragments
  nb_fragments += len(fragments)
  
pr = Progress()
count = -1
for papyrus_id, fragments in papyri_train_val.items():
  count += 1
  if count >= nb_train_val_papyri:
    break
  pr.tick(count, nb_train_val_papyri)

  for i, f in enumerate(fragments):
    extension = f.split('.')[-1]
    file_name = f.split('/')[-1][:-(len(extension)+1)]
    
    # papyrus_class_path = os.path.join((test_dir if count<nb_test_papyri else train_val_dir) , papyrus_id)
    papyrus_class_path = os.path.join(train_val_dir, papyrus_id)

    patches = patch_utils.getPatchesFromImage(f,
                                              patch_size,
                                              nbChannels,
                                              max_patches_per_fragment,
                                              michigan=michigan,
                                              textMaskAvailable=True,
                                              oneBigPatch=oneBigPatch,
                                              geshaem=True,
                                              resize=resize)
    
    ignorePatches = False
    for l in range(0, len(patches)-1):
        for g in range(l+1, len(patches)):
            if (patches[l].shape == patches[g].shape and (patches[l]==patches[g]).all()):
                print("ERROR, 2 SAME PATCHES")
                ignorePatches = True

    if ignorePatches:
      continue
                
    if not oneBigPatch:
      if len(patches) < 2:
        print("COUCOU train val")
        continue

    os.system("mkdir -p "+papyrus_class_path)
    for j, p in enumerate(patches):
        cv2.imwrite(os.path.join(papyrus_class_path, file_name+'_f'+str(i)+'_p'+str(j)+'.png'), p)


pr = Progress()
count = -1
for papyrus_id, fragments in papyri_test.items():
  count += 1
  if count >= nb_test_papyri:
    break
  pr.tick(count, nb_test_papyri)

  for i, f in enumerate(fragments):
    extension = f.split('.')[-1]
    file_name = f.split('/')[-1][:-(len(extension)+1)]
    
    # papyrus_class_path = os.path.join((test_dir if count<nb_test_papyri else train_val_dir) , papyrus_id)
    papyrus_class_path = os.path.join(test_dir, papyrus_id)

    patches = patch_utils.getPatchesFromImage(f,
                                              patch_size,
                                              nbChannels,
                                              max_patches_per_fragment,
                                              michigan=michigan,
                                              textMaskAvailable=True,
                                              oneBigPatch=oneBigPatch,
                                              geshaem=True)
    
    ignorePatches = False
    for l in range(0, len(patches)-1):
        for g in range(l+1, len(patches)):
            if (patches[l].shape == patches[g].shape and (patches[l]==patches[g]).all()):
                print("ERROR, 2 SAME PATCHES")
                ignorePatches = True

    if ignorePatches:
      continue
                
    if not oneBigPatch:
      if len(patches) < 2:
        print("COUCOU test")
        continue

    os.system("mkdir -p "+papyrus_class_path)
    for j, p in enumerate(patches):
        cv2.imwrite(os.path.join(papyrus_class_path, file_name+'_f'+str(i)+'_p'+str(j)+'.png'), p)


dataset_train = {}
dataset_val   = {}
dataset_test  = {}
dataset_train_fragments = {}
dataset_val_fragments   = {}
dataset_test_fragments  = {}
train_val_proportion = 0.8

for i, _class in enumerate(os.listdir(train_val_dir)):
    for file in os.listdir(train_val_dir+'/'+_class):
        if file.endswith(".png"):

          fragment = file.split('_')[-2]
          
          if gray_scale:
            im = cv2.imread(train_val_dir+'/'+_class+'/'+file, cv2.IMREAD_GRAYSCALE)
          else:
            im = cv2.imread(train_val_dir+'/'+_class+'/'+file)
            
          im = im.reshape(-1, imShape[0], imShape[1], imShape[2])
          
          if i < train_val_proportion*len(os.listdir(train_val_dir)):
                
            if _class not in dataset_train:
              dataset_train[_class]       = []
              dataset_train_fragments[_class] = []
              
            dataset_train[_class].append(im)
            dataset_train_fragments[_class].append(fragment)
            
          else:
            
            if _class not in dataset_val:  
              dataset_val[_class]       = []
              dataset_val_fragments[_class] = []
              
            dataset_val[_class].append(im)
            dataset_val_fragments[_class].append(fragment)

for i, _class in enumerate(os.listdir(test_dir)):
    for file in os.listdir(test_dir+'/'+_class):
        if file.endswith(".png"):
          
          fragment = file.split('_')[-2]
          
          if gray_scale:
            im = cv2.imread(test_dir+'/'+_class+'/'+file, cv2.IMREAD_GRAYSCALE)
          else:
            im = cv2.imread(test_dir+'/'+_class+'/'+file)
            
          im = im.reshape(-1, imShape[0], imShape[1], imShape[2])
          
          if _class not in dataset_test:
            dataset_test[_class]           = []
            dataset_test_fragments[_class] = []
            
            
          dataset_test[_class].append(im)
          dataset_test_fragments[_class].append(fragment)

pickleFilesDir = os.path.join(dataset_path, "pickleFiles")
os.system("mkdir -p "+pickleFilesDir)
            
with open(os.path.join(pickleFilesDir, "dataset_train.pickle"), "wb") as f:
    pickle.dump(dataset_train,f)
    
with open(os.path.join(pickleFilesDir, "dataset_val.pickle"), "wb") as f:
    pickle.dump(dataset_val,f)

with open(os.path.join(pickleFilesDir, "dataset_test.pickle"), "wb") as f:
    pickle.dump(dataset_test,f)                


with open(os.path.join(pickleFilesDir, "dataset_train_fragments.pickle"), "wb") as f:
    pickle.dump(dataset_train_fragments,f)
    
with open(os.path.join(pickleFilesDir, "dataset_val_fragments.pickle"), "wb") as f:
    pickle.dump(dataset_val_fragments,f)

with open(os.path.join(pickleFilesDir, "dataset_test_fragments.pickle"), "wb") as f:
    pickle.dump(dataset_test_fragments,f)                
    
exit()
