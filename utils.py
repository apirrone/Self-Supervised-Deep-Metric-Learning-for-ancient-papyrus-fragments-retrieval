import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
import keras.backend as K
import csv
import os
def lr_schedule(epoch):
    """
    Returns a custom learning rate that decreases as epochs progress.
    """
    learning_rate = 1e-3
    if epoch > 10:
        learning_rate = 5e-4
    if epoch > 25:
        learning_rate = 1e-4
    if epoch > 40:
        learning_rate = 5e-5

    tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
    return learning_rate

def normalize_confusion_matrix(confusion_matrix):
    ncm = [[0, 0],
           [0, 0]]
    ncm[0][0] = str(confusion_matrix[0][0]/(confusion_matrix[0][0]+confusion_matrix[1][0])) + " (TN)"
    ncm[0][1] = str(confusion_matrix[0][1]/(confusion_matrix[0][1]+confusion_matrix[1][1])) + " (FP)"
    ncm[1][0] = str(confusion_matrix[1][0]/(confusion_matrix[1][0]+confusion_matrix[0][0])) + " (FN)"
    ncm[1][1] = str(confusion_matrix[1][1]/(confusion_matrix[1][1]+confusion_matrix[0][1])) + " (TP)"

    return ncm

def get_dataset_class_weights_and_initial_bias(dataset):
    patches_list        = []
    patches_labels_list = []
    for papyrus_id, patches in dataset.items():
        for patch in patches :
            patches_list.append(patch)
            patches_labels_list.append(papyrus_id)


    nb_similar    = 0
    nb_dissimilar = 0

    for i in range(len(patches_list)):
        label1 = patches_labels_list[i]
        for j in range(i+1, len(patches_list)):
            label2 = patches_labels_list[j]
            if label1 == label2:
                nb_similar += 1
            else:
                nb_dissimilar += 1


    print(nb_similar, nb_dissimilar)
                
    weight_for_0 = (1 / nb_dissimilar)*(nb_dissimilar+nb_similar)/2.0 
    weight_for_1 = (1 / nb_similar)*(nb_dissimilar+nb_similar)/2.0

    initial_bias = np.log([nb_similar/nb_dissimilar])

    class_weights = {0:weight_for_0, 1:weight_for_1}
    
    return class_weights, initial_bias

data_augmentation = keras.Sequential(
    [
      # layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RandomContrast(0.1)
        # layers.experimental.preprocessing.RandomRotation(0.1),
    ]
)

def my_average_precision_at_k(y_true, y_scores, k): # targets, predictions
    y_scores_sorted, y_true_sorted = (np.array(list(t)) for t in zip(*sorted(zip(y_scores, y_true), reverse=True)))

    y_scores_sorted = y_scores_sorted[:k]
    y_true_sorted   = y_true_sorted[:k]
    # print(y_true)
    # print(y_scores)
    nb_true = (y_true_sorted == 1).sum()
    if nb_true == 0:
        return 0
    nb_true_seen = 1
    sum = 0
    for i in range(0, len(y_scores_sorted)):
        sum += y_true_sorted[i]*(nb_true_seen/(i+1))
        if y_true_sorted[i] == 1:
            nb_true_seen+=1

    return (sum/nb_true)

def histogram_intersection(h1, h2):
    sm = tf.reduce_sum(tf.minimum(h1, h2))
    return sm


def Bhattacharyya_Distance(mu1, std1, mu2, std2):
    return (1/8)*((mu1-mu2)**2)*(1/((mu1**2 + mu2**2)/2.)) + 1/2*tf.math.log(tf.math.abs((mu1**2 + mu2**2)/2.)/(mu1*mu2))

def kprint(tensor):
    print(tf.keras.backend.eval(tensor))
    
def my_histogram_loss(y_true, y_pred):


    # y_true = tf.linalg.normalize(y_true)
    y_pred, _ = tf.linalg.normalize(y_pred)
    indices_pos = tf.where(tf.equal(y_true, 1))
    indices_neg = tf.where(tf.equal(y_true, 0))
    
    y_pred_positives = tf.gather_nd(y_pred, indices_pos)
    y_pred_negatives = tf.gather_nd(y_pred, indices_neg)

    y_pred_positives = K.reshape(y_pred_positives, shape=(-1,1))
    y_pred_negatives = K.reshape(y_pred_negatives, shape=(-1,1))

    mean_positives = tf.math.reduce_mean(y_pred_positives)
    mean_negatives = tf.math.reduce_mean(y_pred_negatives)
    std_positives = tf.math.reduce_std(y_pred_positives)
    std_negatives = tf.math.reduce_std(y_pred_negatives)
    print("")
    print("")
    print("")
    print("")
    # tf.print(mean_positives)
    tf.print(mean_positives, "")
    tf.print(mean_positives, "mean_positives")
    tf.print(mean_negatives, "mean_negatives")
    tf.print(std_positives, "std_positives")
    tf.print(std_negatives, "std_negatives")

    print("")
    print("")
    print("")
    
    return 1./(Bhattacharyya_Distance(mean_positives, std_positives, mean_negatives, std_negatives)+1e-2)

    nb_bins = 50
    rrange = (0., 1.)
    
    edges = tf.range(rrange[0], rrange[1], (rrange[1]-rrange[0])/nb_bins)
    hist_positives = tfp.stats.histogram(y_pred_positives, edges, axis=[0, 1], extend_lower_interval=True) 
    hist_negatives = tfp.stats.histogram(y_pred_negatives, edges, axis=[0, 1], extend_lower_interval=True)

    print(hist_positives)
    # kprint(hist_positives)
    # kprint(hist_negatives)
    return histogram_intersection(hist_positives, hist_negatives)

    

# fragments_scores : 2D dict containing predictions for each pairs of fragment 
def write_hisfrag_csv_and_gt(fragments_scores, path):

    ids_map = {}
    inverted_ids_map = {}
    for i, fragment_id in enumerate(list(fragments_scores.keys())):
        ids_map[fragment_id] = i
        inverted_ids_map[i] = fragment_id

        
    with open(os.path.join(path, "results.csv"), "w") as csv_file:
        for i in range(len(list(fragments_scores.keys()))):
            line = ''+str(i)+','
            for j in range(len(list(fragments_scores.keys()))):
                if i == j:
                    line += str(0)
                else:
                    line += str(1-np.mean(fragments_scores[inverted_ids_map[i]][inverted_ids_map[j]]))
                if j < len(list(fragments_scores.keys()))-1:
                    line += ','
            csv_file.write(line)
            csv_file.write('\n')

    with open(os.path.join(path, "gt.csv"), "w") as csv_file:
        for i in range(len(inverted_ids_map)):
            csv_file.write(str(i)+','+str(inverted_ids_map[i].split('_')[0]))
            csv_file.write('\n')            
        

    
