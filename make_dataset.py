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
train_dir    = os.path.join(dataset_path, "train")
val_dir      = os.path.join(dataset_path, "val")
test_dir     = os.path.join(dataset_path, "test")

os.system("mkdir -p "+dataset_path)
os.system("mkdir -p "+train_dir)
os.system("mkdir -p "+val_dir)
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
nb_papyri                = None
patch_size               = 64
dataset_name             = args.name
extension                = ".png"
# extension                = ".jpg"
data_path                = "/data2/apirrone/final_michigan_base_2_published/"
# data_path                = "/data2/apirrone/michigan_base_2/"
# data_path                = "/data2/apirrone/hisFrag/hisfrag20/"
test_proportion          = 0.1 # proportion of the dataset put aside for testing (independent from train/validation)
# test_proportion          = 0.9 # proportion of the dataset put aside for testing (independent from train/validation)
max_patches_per_fragment = 5
gray_scale               = False
michigan                 = True
onlyRecto                = True
fragmentAsClass          = True
geshaem                  = False # TODO, pour l'instant laisser à False, on verra plus tard
resize                   = None

## Writing parameters in dataset_path/parameters.info

with open(os.path.join(dataset_path, "parameters.info"), "w") as f:
    # Writing data to a file
    f.write("dataset_name : "+str(dataset_name)+'\n')
    f.write("nb_papyri : "+str(nb_papyri)+'\n')
    f.write("patch_size : "+str(patch_size)+'\n')
    f.write("data_path : "+str(data_path)+'\n')
    f.write("test_proportion : "+str(test_proportion)+'\n')
    f.write("max_patches_per_fragment : "+str(max_patches_per_fragment)+'\n')
    f.write("gray_scale : "+str(gray_scale)+'\n')
    f.write("michigan : "+str(michigan)+'\n')
    f.write("onlyRecto : "+str(onlyRecto)+'\n')
    f.write("fragmentAsClass : "+str(fragmentAsClass)+'\n')
    f.write("geshaem : "+str(geshaem)+'\n')
    f.write("resize : "+str(resize)+'\n')
# =====================================



nbChannels = 1 if gray_scale else 3
imShape = (patch_size, patch_size, nbChannels)

# nb_test_papyri      = int(test_proportion*nb_papyri)
# nb_train_val_papyri = nb_papyri-nb_test_papyri

filesPaths = []

for file in os.listdir(data_path):
    if(os.path.isfile(os.path.join(data_path, file)) and file.endswith(extension)) and "infered" not in file:# and 'r' in file:
        if onlyRecto and 'r' not in file:
            continue
        filesPaths.append(os.path.join(data_path, file))
        print("LOADING\t:\t", os.path.join(data_path, file))

print("len(filesPaths) : ", len(filesPaths))

papyri = {}

for f in filesPaths:
    extension = f.split('.')[-1]
    file_name = f.split('/')[-1][:-(len(extension)+1)]
    papyrus_id = get_papyrus_id(file_name, michigan=michigan)
    if papyrus_id not in papyri:
        papyri[papyrus_id] = []
    papyri[papyrus_id].append(f)

papyri_to_remove = [] # because they contain only one fragment

for k, v in papyri.items():
    if len(v) < 2:
        papyri_to_remove.append(k)


for papyrus_id in papyri_to_remove:
    del papyri[papyrus_id]

# From now on, papyri contains all papyri made of at least 2 fragments

papyrus_ids = list(papyri.keys())
random.shuffle(papyrus_ids)

if nb_papyri is not None:
    papyrus_ids = papyrus_ids[:nb_papyri]


train_val_proportion = 0.9

train_val_ids = papyrus_ids[int(len(papyrus_ids)*test_proportion):]
train_ids     = train_val_ids[:int(train_val_proportion*len(train_val_ids))]
val_ids       = train_val_ids[int(train_val_proportion*len(train_val_ids)):]
test_ids      = papyrus_ids[:int(len(papyrus_ids)*test_proportion)]

print(len(train_ids))
print(len(val_ids))
print(len(test_ids))


dataset_train = {}
dataset_val   = {}
dataset_test  = {}


dataset_train_fragments = {}
dataset_val_fragments   = {}
dataset_test_fragments  = {}

# Test 

print("Test : ")
print("")
pr = Progress()
for count, papyrus_id in enumerate(test_ids):

    pr.tick(count, len(test_ids))
    
    for i, fragment in enumerate(papyri[papyrus_id]):

        id = papyrus_id

        if id not in dataset_test:
            dataset_test[id] = []
            dataset_test_fragments[id] = []

        patches = patch_utils.getPatchesFromImage(fragment,
                                                patch_size,
                                                nbChannels,
                                                max_patches_per_fragment,
                                                michigan=michigan,
                                                textMaskAvailable=michigan,
                                                resize=resize)
        ignorePatches = False

        for l in range(0, len(patches)-1):
            for g in range(l+1, len(patches)):
                if (patches[l].shape == patches[g].shape and (patches[l]==patches[g]).all()):
                    print("ERROR, 2 SAME PATCHES")
                    ignorePatches = True

        if ignorePatches or len(patches) < 2:
            continue

        for p in patches:
            dataset_test[id].append(p.reshape(-1, imShape[0], imShape[1], imShape[2]))


        papyrus_class_path = os.path.join(test_dir, id)

        os.system("mkdir -p "+papyrus_class_path)

        for j, p in enumerate(patches):
            # print(os.path.join(papyrus_class_path, id+'_f'+str(i)+'_'+str(j)+'.png'))
            cv2.imwrite(os.path.join(papyrus_class_path, id+'_f'+str(i)+'_'+str(j)+'.png'), p)
            dataset_test_fragments[id].append("f"+str(i))


# Train 

print("Train : ")
print("")
pr = Progress()
for count, papyrus_id in enumerate(train_ids):

    pr.tick(count, len(train_ids))
    
    for i, fragment in enumerate(papyri[papyrus_id]):

        if fragmentAsClass:
            tmp = fragment[:-(len(extension)+1)]
            id = tmp.split('/')[-1]
        else:
            id = papyrus_id

        if id not in dataset_train:
            dataset_train[id] = []
            dataset_train_fragments[id] = []

        patches = patch_utils.getPatchesFromImage(fragment,
                                                patch_size,
                                                nbChannels,
                                                max_patches_per_fragment,
                                                michigan=michigan,
                                                textMaskAvailable=michigan,
                                                resize=resize)
        ignorePatches = False

        for l in range(0, len(patches)-1):
            for g in range(l+1, len(patches)):
                if (patches[l].shape == patches[g].shape and (patches[l]==patches[g]).all()):
                    print("ERROR, 2 SAME PATCHES")
                    ignorePatches = True

        if ignorePatches:
            continue

        for p in patches:
            dataset_train[id].append(p.reshape(-1, imShape[0], imShape[1], imShape[2]))


        papyrus_class_path = os.path.join(train_dir, id)

        os.system("mkdir -p "+papyrus_class_path)

        for j, p in enumerate(patches):
            cv2.imwrite(os.path.join(papyrus_class_path, id+'_f'+str(i)+'_'+str(j)+'.png'), p)
            dataset_train_fragments[id].append("f"+str(i))

# Val

print("Val : ")
print("")
pr = Progress()
for count, papyrus_id in enumerate(val_ids):

    pr.tick(count, len(val_ids))
    
    for i, fragment in enumerate(papyri[papyrus_id]):

        if fragmentAsClass:
            tmp = fragment[:-(len(extension)+1)]
            id = tmp.split('/')[-1]
        else:
            id = papyrus_id

        if id not in dataset_val:
            dataset_val[id] = []
            dataset_val_fragments[id] = []

        patches = patch_utils.getPatchesFromImage(fragment,
                                                patch_size,
                                                nbChannels,
                                                max_patches_per_fragment,
                                                michigan=michigan,
                                                textMaskAvailable=michigan,
                                                resize=resize)
        ignorePatches = False

        for l in range(0, len(patches)-1):
            for g in range(l+1, len(patches)):
                if (patches[l].shape == patches[g].shape and (patches[l]==patches[g]).all()):
                    print("ERROR, 2 SAME PATCHES")
                    ignorePatches = True

        if ignorePatches:
            continue

        for p in patches:
            dataset_val[id].append(p.reshape(-1, imShape[0], imShape[1], imShape[2]))


        papyrus_class_path = os.path.join(val_dir, id)

        os.system("mkdir -p "+papyrus_class_path)

        for j, p in enumerate(patches):
            cv2.imwrite(os.path.join(papyrus_class_path, id+'_f'+str(i)+'_'+str(j)+'.png'), p)
            dataset_val_fragments[id].append("f"+str(i))
        

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