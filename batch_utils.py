import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2

def get_batch_random(batch_size, dataset, input_shape):
    pairs   = [np.zeros((batch_size, input_shape[0], input_shape[1], input_shape[2])) for i in range(2)]
    targets = np.zeros((batch_size,))
    
    patches_list        = []
    patches_labels_list = []
    for papyrus_id, patches in dataset.items():
        for patch in patches :
            patches_list.append(patch)
            patches_labels_list.append(papyrus_id)

    for i in range(batch_size):
                    
        idx1 = np.random.randint(0, len(patches_list))
        idx2 = np.random.randint(0, len(patches_list))
        while idx1 == idx2:
            idx2 = np.random.randint(0, len(patches_list))

        im1 = patches_list[idx1].reshape(input_shape)
        im1_label = patches_labels_list[idx1]
        
        im2 = patches_list[idx2].reshape(input_shape)
        im2_label = patches_labels_list[idx2]
        targets[i] = int(im1_label == im2_label)

        
        pairs[0][i, :, :, :] = im1
        pairs[1][i, :, :, :] = im2

    return pairs, targets

def generate_random(batch_size, dataset, input_shape):
    while True:
        pairs, targets = get_batch_random(batch_size, dataset, input_shape)
        yield (pairs, targets)

        
# if we want to control the proportion of classes in each batch
def get_batch(batch_size, dataset, input_shape, currentEpoch, adaptative_distribution=False):

    if adaptative_distribution:
    
        # epoch at which we go back to original distribution of data
        originalDistribution      = 0.004 # 9661 similar pairs, 2302664 dissimilar pairs
        startingDistribution      = 0.9
        originalDistributionEpoch = 400
        
        a                         = (originalDistribution-startingDistribution)/originalDistributionEpoch
        currentDistribution       = max(0.004, currentEpoch*a + startingDistribution)
        
        nbSimilar    = int(round(currentDistribution*batch_size, 0))
        nbDissimilar = int(round((1-currentDistribution)*batch_size, 0))

    else:
        
        nbSimilar    = batch_size//2
        nbDissimilar = batch_size//2
        
    assert (nbSimilar+nbDissimilar) == batch_size
    
    pairs   = [np.zeros((batch_size, input_shape[0], input_shape[1], input_shape[2])) for i in range(2)]
    targets = np.zeros((batch_size,))
    labels = list(dataset.keys())
    
    for i in range(batch_size):
        current_label = np.random.choice(labels)

        if i < nbSimilar:
            other_label = current_label
        else:
            other_label = np.random.choice(labels)
            while current_label == other_label:
                other_label = np.random.choice(labels)
        
        targets[i] = int(current_label == other_label)
        current_im_idx = np.random.randint(0, len(dataset[current_label]))
        other_im_idx   = np.random.randint(0, len(dataset[other_label]))
        if current_label == other_label:
            while current_im_idx == other_im_idx:
                other_im_idx  = np.random.randint(0, len(dataset[other_label]))
                
        current_im = dataset[current_label][current_im_idx]
        other_im   = dataset[other_label][other_im_idx]
        pairs[0][i, :, :, :] = current_im.reshape(input_shape)
        pairs[1][i, :, :, :] = other_im.reshape(input_shape)

    return pairs, targets

def generate_class_controled(batch_size, dataset, input_shape, currentEpoch):
    while True:
        pairs, targets = get_batch(batch_size, dataset, input_shape, currentEpoch)
        yield (pairs, targets)

# as much similar as dissimilar
def make_friendly_batches(batch_size, dataset, dataset_fragments, input_shape, nb_batches=None):
    # if nb_batches is not None:
    #     assert nb_batches <= len(dataset)

    batches         = []
    targets         = []
    pairs_labels    = []
    pairs_fragments = []
    
    nb_images = 0
    for k, v in dataset.items():
        nb_images += len(v)
        
    patches_list           = []
    patches_labels_list    = []
    patches_fragments_list = []

    for papyrus_id, patches in dataset.items():
        for i, patch in enumerate(patches) :
            patches_list.append(patch)
            patches_labels_list.append(papyrus_id)
            patches_fragments_list.append(dataset_fragments[papyrus_id][i])
            
    patches_list           = np.array(patches_list)
    patches_labels_list    = np.array(patches_labels_list)
    patches_fragments_list = np.array(patches_fragments_list)
        
    nb_pairs = nb_images*2

    if nb_batches is None:
        nb_batches = (nb_pairs // batch_size)+1
    
    # pairs = [np.zeros((batch_size, input_shape[0], input_shape[1], input_shape[2])) for i in range(2)]

    
    for i in range(nb_batches):
        batches.append([np.zeros((batch_size, input_shape[0], input_shape[1], input_shape[2])) for i in range(2)])
        targets.append(np.zeros((batch_size,)))
        pairs_labels.append([])
        pairs_fragments.append([])

    currentBatch = 0
    k = 0
    for i in range(len(patches_list)):
        im1          = patches_list[i].reshape(input_shape)
        im1_label    = patches_labels_list[i]
        im1_fragment = patches_fragments_list[i]
        
        im2_label = im1_label
        im3_label = np.random.choice(list(dataset.keys()))
        while im3_label == im1_label:
            im3_label = np.random.choice(list(dataset.keys()))
            
        tmp          = patches_list[patches_labels_list==im2_label]
        tmp2         = patches_fragments_list[patches_labels_list==im2_label]
        im2_idx      = np.random.randint(0, len(tmp))
        im2          = tmp[im2_idx].reshape(input_shape)
        im2_fragment = tmp2[im2_idx]

        tmp          = patches_list[patches_labels_list==im3_label]
        tmp2         = patches_fragments_list[patches_labels_list==im3_label]
        im3_idx      = np.random.randint(0, len(tmp))
        im3          = tmp[im3_idx].reshape(input_shape)
        im3_fragment = tmp2[im3_idx]

        batches[currentBatch][0][k, :, :, :] = im1
        batches[currentBatch][1][k, :, :, :] = im2
        targets[currentBatch][k] = int(im1_label == im2_label)
        pairs_labels[currentBatch].append([im1_label, im2_label])
        pairs_fragments[currentBatch].append([im1_fragment, im2_fragment])
        k += 1
        batches[currentBatch][0][k, :, :, :] = im1
        batches[currentBatch][1][k, :, :, :] = im3
        targets[currentBatch][k] = int(im1_label == im3_label)
        pairs_labels[currentBatch].append([im1_label, im3_label])
        pairs_fragments[currentBatch].append([im1_fragment, im3_fragment])
        k += 1
        if k >= batch_size:
            k = 0
            currentBatch += 1
            if nb_batches is not None and currentBatch >= nb_batches:
                break
            
            
    return batches, targets, np.array(pairs_labels), np.array(pairs_fragments)



def prepare_batch_dynamic(batch_size, dataset, dataset_fragments, input_shape, nb_paps=None):
    if nb_paps is not None:
        assert nb_paps <= len(dataset)
    
    nb_images = 0
    for k, v in dataset.items():
        nb_images += len(v)
        
    patches_list           = []
    patches_labels_list    = []
    patches_fragments_list = []
    
    count = -1

    nb_patches_per_paps = {}
    nb_fragments_per_paps = {}
    for papyrus_id, patches in dataset.items():
        nb_fragments_per_paps[papyrus_id] = []
        count += 1
        if nb_paps is not None and count >= nb_paps:
            break
        nb_patches_per_paps[papyrus_id] = len(patches)
        for i, patch in enumerate(patches) :
            patches_list.append(patch)
            patches_labels_list.append(papyrus_id)
            patches_fragments_list.append(dataset_fragments[papyrus_id][i])
            fragment_id  = papyrus_id+'_'+dataset_fragments[papyrus_id][i]
            if fragment_id not in nb_fragments_per_paps[papyrus_id]:
                nb_fragments_per_paps[papyrus_id].append(fragment_id)

    nb_fragments = 0
    for papyrus_id in nb_patches_per_paps.keys():
        print(papyrus_id, len(nb_fragments_per_paps[papyrus_id]), "fragments, ", nb_patches_per_paps[papyrus_id], "patches")
        nb_fragments += len(nb_fragments_per_paps[papyrus_id])

    nb_patches = len(patches_list)
    
    print("nb_fragments : ", nb_fragments)
    print("nb_patches : ", nb_patches)
            
    nb_pairs = 0
    for i in range(len(patches_list)):
        for j in range(0, len(patches_list)):
            nb_pairs += 1
            
    print("NB PAIRS : ", nb_pairs)
    
    nb_batches = (nb_pairs // batch_size)+1

    return patches_list, patches_labels_list, patches_fragments_list, nb_fragments, nb_pairs, nb_batches, nb_patches

def get_batch_dynamic(batch_size, input_shape, patches_list, patches_labels_list, patches_fragments_list, startIndex):

    batch           = [np.zeros((batch_size, input_shape[0], input_shape[1] , input_shape[2])) for i in range(2)]
    targets         = np.zeros((batch_size,))
    pairs_labels    = []
    pairs_fragments = []
    
    k = -1
    count = -1
    for i in range(0, len(patches_list)):
        im1          = patches_list[i].reshape(input_shape)
        im1_label    = patches_labels_list[i]
        im1_fragment = patches_fragments_list[i]
        
        for j in range(0, len(patches_list)):
            count += 1
            if count <= startIndex:
                continue
            k += 1
            im2          = patches_list[j].reshape(input_shape)
            im2_label    = patches_labels_list[j]
            im2_fragment = patches_fragments_list[j]

            batch[0][k, :, :, :] = im1
            batch[1][k, :, :, :] = im2
            targets[k] = (im1_label == im2_label)
            pairs_labels.append([im1_label, im2_label])
            pairs_fragments.append([im1_fragment, im2_fragment])

            if k >= batch_size-1:
                # return batch, targets, np.array(pairs_labels), np.array(pairs_fragments), count
                return batch, targets, pairs_labels, pairs_fragments, count


def make_batches3(batch_size, dataset, dataset_fragments, input_shape, nb_paps=None):
    if nb_paps is not None:
        print(len(dataset))
        assert nb_paps <= len(dataset)
    
    batches         = []
    targets         = []
    pairs_labels    = []
    pairs_fragments = []
    
    nb_images = 0
    for k, v in dataset.items():
        nb_images += len(v)
        
    patches_list           = []
    patches_labels_list    = []
    patches_fragments_list = []
    
    count = -1

    nb_patches_per_paps = {}
    nb_fragments_per_paps = {}
    for papyrus_id, patches in dataset.items():
        nb_fragments_per_paps[papyrus_id] = []
        count += 1
        if nb_paps is not None and count >= nb_paps:
            break
        nb_patches_per_paps[papyrus_id] = len(patches)
        for i, patch in enumerate(patches) :
            patches_list.append(patch)
            patches_labels_list.append(papyrus_id)
            patches_fragments_list.append(dataset_fragments[papyrus_id][i])
            fragment_id  = papyrus_id+'_'+dataset_fragments[papyrus_id][i]
            if fragment_id not in nb_fragments_per_paps[papyrus_id]:
                nb_fragments_per_paps[papyrus_id].append(fragment_id)

    nb_fragments = 0
    for papyrus_id in nb_patches_per_paps.keys():
        print(papyrus_id, len(nb_fragments_per_paps[papyrus_id]), "fragments, ", nb_patches_per_paps[papyrus_id], "patches")
        nb_fragments += len(nb_fragments_per_paps[papyrus_id])

    nb_patches = len(patches_list)
    
    print("nb_fragments : ", nb_fragments)
    print("nb_patches : ", nb_patches)
            
    nb_pairs = 0
    for i in range(len(patches_list)):
        for j in range(0, len(patches_list)):
            nb_pairs += 1
            
    print("NB PAIRS : ", nb_pairs)
    
    nb_batches = (nb_pairs // batch_size)+1



    
    # =======



    
    all_patches_pairs   = []
    all_labels_pairs    = []
    all_fragments_pairs = []
    
    for i in range(len(patches_list)):
        im1          = patches_list[i].reshape(input_shape)
        im1_label    = patches_labels_list[i]
        im1_fragment = patches_fragments_list[i]
        
        for j in range(0, len(patches_list)):
        # for j in range(i+1, len(patches_list)):
            im2          = patches_list[j].reshape(input_shape)
            im2_label    = patches_labels_list[j]
            im2_fragment = patches_fragments_list[j]

            all_patches_pairs.append([im1, im2])
            all_labels_pairs.append([im1_label, im2_label])
            all_fragments_pairs.append([im1_fragment, im2_fragment])
            
    for i in range(nb_batches):
        batches.append([np.zeros((batch_size, 2)) for i in range(2)])
        targets.append(np.zeros((batch_size,)))
        pairs_labels.append([])
        pairs_fragments.append([])
            
    currentBatch = 0
    k = 0
    for i in range(len(all_patches_pairs)):
        # batches[currentBatch][0][k, :, :, :] = all_patches_pairs[i][0]
        # batches[currentBatch][1][k, :, :, :] = all_patches_pairs[i][1]
        batches[currentBatch][0][k, :] = [i, 0]
        batches[currentBatch][1][k, :] = [i, 1]

        targets[currentBatch][k] = int(all_labels_pairs[i][0] == all_labels_pairs[i][1])
        pairs_labels[currentBatch].append(all_labels_pairs[i])
        pairs_fragments[currentBatch].append(all_fragments_pairs[i])

        k += 1
        if k >= batch_size:
            k = 0
            currentBatch += 1
            
    return batches, targets, np.array(pairs_labels), np.array(pairs_fragments), all_patches_pairs, nb_fragments, nb_patches


def make_batches2(batch_size, dataset, dataset_fragments, input_shape, nb_paps=None):
    if nb_paps is not None:
        assert nb_paps <= len(dataset)
    
    batches         = []
    targets         = []
    pairs_labels    = []
    pairs_fragments = []
    
    nb_images = 0
    for k, v in dataset.items():
        nb_images += len(v)
        
    patches_list           = []
    patches_labels_list    = []
    patches_fragments_list = []
    
    count = -1

    nb_fragments_per_paps = {}
    for papyrus_id, patches in dataset.items():
        count += 1
        if nb_paps is not None and count >= nb_paps:
            break
        nb_fragments_per_paps[papyrus_id] = len(patches)
        for i, patch in enumerate(patches) :
            patches_list.append(patch)
            patches_labels_list.append(papyrus_id)
            patches_fragments_list.append(dataset_fragments[papyrus_id][i])

    for k, v in nb_fragments_per_paps.items():
        print(k, v)


    print("nb_patches : ", len(patches_list))

    # TODO remove, for testing purposes only
    patches_list           = patches_list#[:300]
    patches_labels_list    = patches_labels_list#[:300]
    patches_fragments_list = patches_fragments_list#[:300]
    # =====================
            
    nb_pairs = 0
    for i in range(len(patches_list)):
        for j in range(0, len(patches_list)):
            nb_pairs += 1
            
    print("NB PAIRS : ", nb_pairs)
    
    nb_batches = (nb_pairs // batch_size)+1


    # putain quelle merde
    # le contenu de ce truc était propagé dans tous les batchs
    # quel putain d'enfer
    # pairs = [np.zeros((batch_size, input_shape[0], input_shape[1] , input_shape[2])) for i in range(2)]

    all_patches_pairs   = []
    all_labels_pairs    = []
    all_fragments_pairs = []
    
    for i in range(len(patches_list)):
        im1          = patches_list[i].reshape(input_shape)
        im1_label    = patches_labels_list[i]
        im1_fragment = patches_fragments_list[i]
        
        for j in range(0, len(patches_list)):
        # for j in range(i+1, len(patches_list)):
            im2          = patches_list[j].reshape(input_shape)
            im2_label    = patches_labels_list[j]
            im2_fragment = patches_fragments_list[j]

            all_patches_pairs.append([im1, im2])
            all_labels_pairs.append([im1_label, im2_label])
            all_fragments_pairs.append([im1_fragment, im2_fragment])
            
    for i in range(nb_batches):
        batches.append([np.zeros((batch_size, input_shape[0], input_shape[1] , input_shape[2])) for i in range(2)])
        targets.append(np.zeros((batch_size,)))
        pairs_labels.append([])
        pairs_fragments.append([])
            
    currentBatch = 0
    k = 0
    for i in range(len(all_patches_pairs)):
        batches[currentBatch][0][k, :, :, :] = all_patches_pairs[i][0]
        batches[currentBatch][1][k, :, :, :] = all_patches_pairs[i][1]

        targets[currentBatch][k] = int(all_labels_pairs[i][0] == all_labels_pairs[i][1])
        pairs_labels[currentBatch].append(all_labels_pairs[i])
        pairs_fragments[currentBatch].append(all_fragments_pairs[i])

        k += 1
        if k >= batch_size:
            k = 0
            currentBatch += 1
            
    return batches, targets, np.array(pairs_labels), np.array(pairs_fragments)
        


def make_batches(batch_size, dataset, dataset_fragments, input_shape, nb_paps=None):
    if nb_paps is not None:
        assert nb_paps <= len(dataset)
    
    batches         = []
    targets         = []
    pairs_labels    = []
    pairs_fragments = []
    
    nb_images = 0
    for k, v in dataset.items():
        nb_images += len(v)
        
    patches_list           = []
    patches_labels_list    = []
    patches_fragments_list = []
    
    count = -1

    nb_fragments_per_paps = {}
    for papyrus_id, patches in dataset.items():
        count += 1
        if nb_paps is not None and count >= nb_paps:
            break
        nb_fragments_per_paps[papyrus_id] = len(patches)
        for i, patch in enumerate(patches) :
            patches_list.append(patch)
            patches_labels_list.append(papyrus_id)
            patches_fragments_list.append(dataset_fragments[papyrus_id][i])

    for k, v in nb_fragments_per_paps.items():
        print(k, v)


    print("nb_patches : ", len(patches_list))

    # TODO remove, for testing purposes only
    patches_list           = patches_list#[:300]
    patches_labels_list    = patches_labels_list#[:300]
    patches_fragments_list = patches_fragments_list#[:300]
    # =====================
            
    nb_pairs = 0
    for i in range(len(patches_list)):
        for j in range(0, len(patches_list)):
            nb_pairs += 1
            
    print("NB PAIRS : ", nb_pairs)
    
    nb_batches = (nb_pairs // batch_size)+1

    pairs = [np.zeros((batch_size, input_shape[0], input_shape[1] , input_shape[2])) for i in range(2)]

    
    for i in range(nb_batches):
        batches.append(pairs)
        targets.append(np.zeros((batch_size,)))
        pairs_labels.append([])
        pairs_fragments.append([])

    currentBatch = 0
    k = 0
    for i in range(len(patches_list)):
        im1          = patches_list[i].reshape(input_shape)
        im1_label    = patches_labels_list[i]
        im1_fragment = patches_fragments_list[i]
        
        for j in range(0, len(patches_list)):
        # for j in range(i+1, len(patches_list)):
            im2          = patches_list[j].reshape(input_shape)
            im2_label    = patches_labels_list[j]
            im2_fragment = patches_fragments_list[j]
            
            batches[currentBatch][0][k, :, :, :] = im1
            batches[currentBatch][1][k, :, :, :] = im2
            targets[currentBatch][k] = int(im1_label == im2_label)
            pairs_labels[currentBatch].append([im1_label, im2_label])
            pairs_fragments[currentBatch].append([im1_fragment, im2_fragment])
            k+=1
            if k >= batch_size:
                k = 0
                currentBatch += 1
            
    return batches, targets, np.array(pairs_labels), np.array(pairs_fragments)


def generate(batches, targets):
    for i in range(len(batches)):
        pairs = batches[i]
        t     = targets[i]
        yield (pairs, t)

# to be used with make_batches3
def generate_predict3(batches, batch_size, input_shape, all_patches_pairs):

    for i in range(len(batches)):
        batch = batches[i]
        pairs = [np.zeros((batch_size, input_shape[0], input_shape[1] , input_shape[2])) for i in range(2)]

        for j, b in enumerate(batch[0]):            
            pairs[0][j, :, :, :] = all_patches_pairs[int(b[0])][int(b[1])]
            
        for j, b in enumerate(batch[1]):            
            pairs[1][j, :, :, :] = all_patches_pairs[int(b[0])][int(b[1])]
        
        yield pairs

    
    
    
        
def generate_predict(batches):
    for i in range(len(batches)):
        yield batches[i]
        
def make_test_batches(batch_size, dataset, input_shape):

    batches      = []
    targets      = []
    pairs_labels = []
    
    pairs = [np.zeros((batch_size, input_shape[0], input_shape[1] , input_shape[2])) for i in range(2)]
    batches.append(pairs)
    targets.append(np.empty((batch_size,)))
    
    patches_list = []
    labels_list  = []
    labels = list(dataset.keys())
    
    for label, fragments in dataset.items():
        for fragment_id, patches in fragments.items():
            for p in patches:
                patches_list.append(p)
                labels_list.append({"label" : label, "fragment_id":fragment_id})

    currentBatch = 0
    count        = 0
    for i in range(len(patches_list)):

        im1                = patches_list[i]
        im1_label          = labels_list[i]["label"]
        im1_fragment_id    = labels_list[i]["fragment_id"]
        im1_combined_label = str(im1_label)+'_'+str(im1_fragment_id)
        
        for j in range(i+1, len(patches_list)):
            im2                = patches_list[j]
            im2_label          = labels_list[j]["label"]
            im2_fragment_id    = labels_list[j]["fragment_id"]
            im2_combined_label = str(im2_label)+'_'+str(im2_fragment_id)

            if im1_label == im2_label and im1_fragment_id == im2_fragment_id:
                continue

            # print(im1_label, im2_label)
            
            batches[currentBatch][0][count, :, :, :] = im1.reshape(input_shape)
            batches[currentBatch][1][count, :, :, :] = im2.reshape(input_shape)
            targets[currentBatch][count] = (im1_label == im2_label)
            pairs_labels.append([{"label" : im1_label, "fragment_id" : im1_fragment_id}, {"label" : im2_label, "fragment_id" : im2_fragment_id}])
            
            count += 1
            
            if count >= batch_size-1:
                count = 0
                currentBatch += 1
                batches.append(pairs)
                targets.append(np.empty((batch_size,)))
                # targets.append([])
            
            # print("(", im1_label, ", ", im1_fragment_id, "), (", im2_label, ", ", im2_fragment_id, ")")

    return batches, targets, pairs_labels
        


def get_random_test_batch(batch_size, dataset, input_shape, balanced = False):

    pairs   = [np.zeros((batch_size, input_shape[0], input_shape[1] , input_shape[2])) for i in range(2)]
    targets = np.zeros((batch_size,))
    pairs_labels = []
    
    patches_list = []
    labels_list  = []
    labels = list(dataset.keys())
    
    for label, fragments in dataset.items():
        for fragment_id, patches in fragments.items():
            for p in patches:
                patches_list.append(p)
                labels_list.append({"label" : label, "fragment_id":fragment_id})


    for i in range(batch_size):

        idx1 = np.random.randint(0, len(patches_list))
        idx2 = np.random.randint(0, len(patches_list))
        
        im1_label          = labels_list[idx1]["label"]        
        im2_label          = labels_list[idx2]["label"]
        
        if balanced:
            if i%2 == 0: # similar pair
                while im1_label == im2_label:
                    idx2      = np.random.randint(0, len(patches_list))
                    im2_label = labels_list[idx2]["label"]
            else: # dissimilar pair
                while im1_label != im2_label:
                    idx2      = np.random.randint(0, len(patches_list))
                    im2_label = labels_list[idx2]["label"]
        else:
            while idx1 == idx2:
                idx2 = np.random.randint(0, len(patches_list))

        im1 = patches_list[idx1].reshape(input_shape)
        im1_label          = labels_list[idx1]["label"]
        im1_fragment_id    = labels_list[idx1]["fragment_id"]
        im1_combined_label = str(im1_label)+'_'+str(im1_fragment_id)
        
        im2 = patches_list[idx2].reshape(input_shape)
        im2_label          = labels_list[idx2]["label"]
        im2_fragment_id    = labels_list[idx2]["fragment_id"]
        im2_combined_label = str(im2_label)+'_'+str(im2_fragment_id)
        
        targets[i] = int(im1_label == im2_label)
        # targets[i] = int(im1_combined_label == im2_combined_label)

        pairs_labels.append([{"label" : im1_label, "fragment_id" : im1_fragment_id}, {"label" : im2_label, "fragment_id" : im2_fragment_id}])
        pairs[0][i, :, :, :] = im1
        pairs[1][i, :, :, :] = im2
                

    return pairs, targets, pairs_labels


def get_random_test_batch_fragments(batch_size, dataset, input_shape, balanced = False):

    pairs   = [np.zeros((batch_size, input_shape[0], input_shape[1] , input_shape[2])) for i in range(2)]
    targets = np.zeros((batch_size,))
    pairs_labels = []
    
    patches_list = []
    labels_list  = []
    labels = list(dataset.keys())
    
    for label, fragments in dataset.items():
        for fragment_id, patches in fragments.items():
            for p in patches:
                patches_list.append(p)
                labels_list.append({"label" : label, "fragment_id":fragment_id})

    k = 0
    nb_patches_per_fragment_pair = 8
    for i in range(batch_size//nb_patches_per_fragment_pair):

        im1_label = np.random.choice(labels)
        im2_label = np.random.choice(labels)

        if balanced:
            if i%2 == 0: # similar pair
                im2_label = im1_label
            else:
                while im1_label == im2_label:
                    im2_label = np.random.choice(labels)
                    

        fragment1_id = np.random.choice(list(dataset[im1_label].keys()))
        fragment2_id = np.random.choice(list(dataset[im2_label].keys()))
        if im1_label == im2_label:
            while fragment1_id == fragment2_id:
                fragment2_id = np.random.choice(list(dataset[im2_label].keys()))
                
        for j in range(nb_patches_per_fragment_pair):
            idx1 = np.random.randint(0, len(dataset[im1_label][fragment1_id]))
            im1 = dataset[im1_label][fragment1_id][idx1]

            idx2 = np.random.randint(0, len(dataset[im2_label][fragment2_id]))
            im2 = dataset[im2_label][fragment2_id][idx2]

            targets[k] = int(im1_label == im2_label)
            # targets[i] = int(im1_combined_label == im2_combined_label)

            pairs_labels.append([{"label" : im1_label, "fragment_id" : fragment1_id}, {"label" : im2_label, "fragment_id" : fragment2_id}])
            pairs[0][k, :, :, :] = im1
            pairs[1][k, :, :, :] = im2
            
            k += 1
                
    return pairs, targets, pairs_labels


def get_n_way_task_test(n, dataset, input_shape):
    
    pairs   = [np.zeros((n, input_shape[0], input_shape[1] , input_shape[2])) for i in range(2)]
    targets = np.zeros((n,))

    patches_list = []
    labels_list  = []
    labels = list(dataset.keys())
    
    for label, fragments in dataset.items():
        for fragment_id, patches in fragments.items():
            for p in patches:
                patches_list.append(p)
                labels_list.append({"label" : label, "fragment_id":fragment_id})

    for i in range(n):
        idx1 = np.random.randint(0, len(patches_list))
        idx2 = np.random.randint(0, len(patches_list))

        im1_label          = labels_list[idx1]["label"]        
        im2_label          = labels_list[idx2]["label"]

        # only one similar example
        if i == 0:
            while im1_label != im2_label:
                idx2 = np.random.randint(0, len(patches_list))        
                im2_label          = labels_list[idx2]["label"]
        else:
            while im1_label == im2_label:
                idx2 = np.random.randint(0, len(patches_list))        
                im2_label          = labels_list[idx2]["label"]

        im1 = patches_list[idx1].reshape(input_shape)
        im2 = patches_list[idx2].reshape(input_shape)
        
        pairs[0][i, :, :, :] = im1
        pairs[1][i, :, :, :] = im2
        targets[i]           = (im1_label == im2_label)

    return pairs, targets

def get_map_task_val(n, dataset, input_shape):
    
    pairs   = [np.zeros((n, input_shape[0], input_shape[1] , input_shape[2])) for i in range(2)]
    targets = np.zeros((n,))

    patches_list = []
    labels_list  = []
    labels = list(dataset.keys())
    
    for label, patches in dataset.items():
        for p in patches:
            patches_list.append(p)
            labels_list.append({"label" : label})

    for i in range(n):
        idx1 = np.random.randint(0, len(patches_list))
        idx2 = np.random.randint(0, len(patches_list))

        im1_label          = labels_list[idx1]["label"]        
        im2_label          = labels_list[idx2]["label"]

        if i%2 == 0:
            while im1_label != im2_label:
                idx2 = np.random.randint(0, len(patches_list))        
                im2_label          = labels_list[idx2]["label"]
        else:
            while im1_label == im2_label:
                idx2 = np.random.randint(0, len(patches_list))        
                im2_label          = labels_list[idx2]["label"]

        im1 = patches_list[idx1].reshape(input_shape)
        im2 = patches_list[idx2].reshape(input_shape)
        
        pairs[0][i, :, :, :] = im1
        pairs[1][i, :, :, :] = im2
        targets[i]           = (im1_label == im2_label)

    return pairs, targets


def get_n_way_task_val(n, dataset, input_shape):
    
    pairs   = [np.zeros((n, input_shape[0], input_shape[1] , input_shape[2])) for i in range(2)]
    targets = np.zeros((n,))

    patches_list = []
    labels_list  = []
    labels = list(dataset.keys())
    
    for label, patches in dataset.items():
        for p in patches:
            patches_list.append(p)
            labels_list.append({"label" : label})

    for i in range(n):
        idx1 = np.random.randint(0, len(patches_list))
        idx2 = np.random.randint(0, len(patches_list))

        im1_label          = labels_list[idx1]["label"]        
        im2_label          = labels_list[idx2]["label"]

        # only one similar example
        if i == 0:
            while im1_label != im2_label:
                idx2 = np.random.randint(0, len(patches_list))        
                im2_label          = labels_list[idx2]["label"]
        else:
            while im1_label == im2_label:
                idx2 = np.random.randint(0, len(patches_list))        
                im2_label          = labels_list[idx2]["label"]

        im1 = patches_list[idx1].reshape(input_shape)
        im2 = patches_list[idx2].reshape(input_shape)
        
        pairs[0][i, :, :, :] = im1
        pairs[1][i, :, :, :] = im2
        targets[i]           = (im1_label == im2_label)

    return pairs, targets

# at fragment level
# multiple pairs of patches between pairs of fragments
def get_n_way_task_test_2(n, dataset, input_shape):

    nb_patches_per_fragment_pair = 8

    batches = []
    targets = []
    
    pairs   = [np.zeros((nb_patches_per_fragment_pair, input_shape[0], input_shape[1] , input_shape[2])) for i in range(2)]
    # targets = np.zeros((nb_patches_per_fragment_pair,))

    for i in range(n):
        batches.append(pairs)
        # targets.append(np.zeros((,)))

    patches_list = []
    labels_list  = []
    labels = list(dataset.keys())
    
    for label, fragments in dataset.items():
        for fragment_id, patches in fragments.items():
            for p in patches:
                patches_list.append(p)
                labels_list.append({"label" : label, "fragment_id":fragment_id})

    for i in range(n):
        idx1 = np.random.randint(0, len(patches_list))
        idx2 = np.random.randint(0, len(patches_list))

        im1_label          = labels_list[idx1]["label"]        
        im2_label          = labels_list[idx2]["label"]

        # only one similar example
        if i == 0:
            while im1_label != im2_label:
                idx2 = np.random.randint(0, len(patches_list))        
                im2_label          = labels_list[idx2]["label"]
        else:
            while im1_label == im2_label:
                idx2 = np.random.randint(0, len(patches_list))        
                im2_label          = labels_list[idx2]["label"]

        im1 = patches_list[idx1].reshape(input_shape)
        im2 = patches_list[idx2].reshape(input_shape)
        
        pairs[0][i, :, :, :] = im1
        pairs[1][i, :, :, :] = im2
        targets[i]           = (im1_label == im2_label)

    return pairs, targets
