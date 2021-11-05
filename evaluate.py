import argparse
import os
import datetime
import numpy as np
now = datetime.datetime.now()
import sklearn.metrics
from progress import Progress

import seaborn as sn
import pandas as pd
import cv2

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (16.0, 10.0)

def typeDir(str):
    if(not os.path.isdir(str)):
        raise argparse.ArgumentTypeError("{0} is not a directory.".format(str))
    return str

argParser = argparse.ArgumentParser(description='Evaluate')
argParser.add_argument('-g', '--gpuNb', type=int, required=True, help="gpu for training")
argParser.add_argument('-sa', '--saveDir', type=typeDir, required=False, default='/', help="directory containing the experiment, contains logs/ . The files will be saved here")
argParser.add_argument('-m', '--modelDir', type=typeDir, required=True, help="directory containing the experiment, contains logs/ . The files will be saved here")
argParser.add_argument('-da', '--dataset', type=typeDir, required=True, help="path to dataset")
argParser.add_argument('-e', '--experimentName', type=str, required=False, default="experiment_"+now.strftime("%Y-%m-%d_%H:%M"), help="name of the experiment")
argParser.add_argument('-s', '--step', type=int, required=False, default=5, help="training step (epoch)")
argParser.add_argument('-r', '--reCompute', type=typeDir, required=False, default=None,  help="re create results figures from pickle files or csv files. provide path to the directory containing current results figures")
args = argParser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpuNb)

if args.reCompute is None:
    experimentDir = os.path.join(args.saveDir, args.experimentName)
    os.system("mkdir -p "+experimentDir)
else:
    experimentDir = args.reCompute

import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import tensorflow.keras.models
import utils
import batch_utils
import models


import backtrace
backtrace.hook(
    reverse=False,
    align=True,
    strip_path=True,
    enable_on_envvar_only=False,
    on_tty=False,
    conservative=False,
    styles={})

with open(os.path.join(args.dataset, "dataset_test.pickle"), "rb") as f:
    dataset_test = pickle.load(f)

with open(os.path.join(args.dataset, "dataset_test_fragments.pickle"), "rb") as f:
    dataset_test_fragments = pickle.load(f)

if args.reCompute is not None:
    with open(os.path.join(args.reCompute, "fpairs_fragments.pickle"), "rb") as f:
        fpairs_fragments = pickle.load(f)
        
    with open(os.path.join(args.reCompute, "fpairs_labels.pickle"), "rb") as f:
        fpairs_labels = pickle.load(f)
        
    with open(os.path.join(args.reCompute, "predictions.pickle"), "rb") as f:
        predictions = pickle.load(f)
        
    with open(os.path.join(args.reCompute, "targets.pickle"), "rb") as f:
        targets = pickle.load(f)    
        
sample_im_shape = dataset_test[list(dataset_test.keys())[0]][0].shape
input_shape = (sample_im_shape[1], sample_im_shape[2], sample_im_shape[3])

# === Evaluation Hyperparameters ===
nb_paps = 100 # number of papyrus from the test set to use
# ==================================

# if input_shape[2] == 1:
#     print("CONVERTING GRAYSCALE TO RGB (by duplicating channels) ...")
#     # convert grayscale to RGB by duplicating channels for resnet and vgg
#     for papyrus_id, fragments in dataset_test.items():
#         for i in range(len(fragments)):
#             fragment = fragments[i]
#             new_fragment = np.zeros((fragment.shape[0], fragment.shape[1], fragment.shape[2], 3))
#             new_fragment[:,:,:,0] = fragment[:,:,:,0]
#             new_fragment[:,:,:,1] = fragment[:,:,:,0]
#             new_fragment[:,:,:,2] = fragment[:,:,:,0]
#             dataset_test[papyrus_id][i] = new_fragment
            
#     input_shape = (64, 64, 3)

if args.reCompute is None:
    modelPath = os.path.join(args.modelDir, "logs", "save_at_"+str(args.step)+".h5")
    model = load_model(modelPath)#, custom_objects={"K":tensorflow.keras.backend, "models":tensorflow.keras.models})
    model.summary()



def make_map_tests():
    n_values = [x for x in range(2, 20, 1)]
    nb_trials_per_n = 10
    
    results = []
    for n in n_values:
        correct = 0
        map = []
        for t in range(nb_trials_per_n):
            n_way_pairs, n_way_targets = batch_utils.get_map_task_val(n, dataset_test, input_shape)
            
            predictions = model.predict(n_way_pairs, batch_size=n, steps=1)

            ap = utils.my_average_precision_at_k(n_way_targets, predictions, n)
            map.append(ap)
                
        results.append(np.mean(map))
            
    return results, n_values
                

def make_map():
    results, n_values = make_map_tests()
    
    plt.cla()
    plt.figure(figsize = (10,7))
    plt.plot(n_values, results)
    plt.xlabel('n')
    plt.ylabel('map')
    plt.savefig(os.path.join(experimentDir, "MAP.png"))
    plt.close()

def make_n_way_tests():
    n_values = [x for x in range(2, 20, 1)]
    nb_trials_per_n = 10
    
    results = []
    for n in n_values:
        correct = 0
        for t in range(nb_trials_per_n):
            n_way_pairs, n_way_targets = batch_utils.get_n_way_task_val(n, dataset_test, input_shape)
            
            predictions = model.predict(n_way_pairs, batch_size=n, steps=1)
            
            index_max = np.argmax(predictions)
            if int(n_way_targets[index_max]) == 1:
                correct += 1
        results.append((correct/(nb_trials_per_n*1.)))
                
    return results, n_values
                

def make_n_way():
    results, n_values = make_n_way_tests()
        
    plt.cla()
    plt.figure(figsize = (10,7))
    plt.plot(n_values, results)
    plt.xlabel('n')
    plt.ylabel('correct percentage')
    
    plt.savefig(os.path.join(experimentDir, "n_way.png"))
    plt.close()
    
def make_histogram(predictions, targets, name):
    predictions_true  = predictions[targets==1]
    predictions_false = predictions[targets==0]

    plt.cla()
    fig,ax = plt.subplots(1,1)
    # plt.yscale('log', nonposy='clip')
    ax.hist(predictions_false, label="predictions false")
    ax.hist(predictions_true, label="predictions true", fc=(1, 0, 0, 0.5))

    ax.set_title("histogram of result")
    ax.set_xlabel('score')
    ax.set_ylabel('nb scores')
    plt.legend()
    plt.savefig(os.path.join(experimentDir, name+".png"))
    plt.close()

def make_confusion_matrix(predictions, targets, name, normalized=False):
    threshold = 0.8
        
    if normalized :
        confusion_mat = sklearn.metrics.confusion_matrix(targets, [1 if x>threshold else 0 for x in predictions], normalize='pred')
    else:
        confusion_mat = sklearn.metrics.confusion_matrix(targets, [1 if x>threshold else 0 for x in predictions])
        
    df_cm = pd.DataFrame(confusion_mat,
                         index = ["TN, FP", "FN, TP"],
                         columns = [" ", " "])
    plt.cla()
    plt.figure(figsize = (10,7))
    if normalized:
        sn.heatmap(df_cm, annot=True, fmt="f")
    else:
        sn.heatmap(df_cm, annot=True, fmt="d")
        
    plt.savefig(os.path.join(experimentDir, name+".png"))
    plt.close()
    
    return confusion_mat
    
def make_precision_recall(predictions, targets, name):
        
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(targets, predictions)
    plt.cla()
    plt.figure(figsize = (10,7))
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('precision')
    plt.savefig(os.path.join(experimentDir, name+".png"))
    plt.close()
    
def make_ROC(predictions, targets, name):

    fpr, tpr, thresholds = sklearn.metrics.roc_curve(targets, predictions)        
    plt.cla()
    plt.figure(figsize = (10,7))
    plt.plot(fpr, tpr)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.savefig(os.path.join(experimentDir, name+".png"))
    plt.close()

if args.reCompute is None:
    print("Making batches")
    batch_size = 64
    batches, targets, pairs_labels, pairs_fragments, all_patches_pairs, nb_fragments, nb_patches = batch_utils.make_batches3(batch_size, dataset_test, dataset_test_fragments, input_shape, nb_paps=nb_paps)
    print("Making predictions")
    predictions = model.predict(batch_utils.generate_predict3(batches, batch_size, input_shape, all_patches_pairs), batch_size=batch_size, steps=len(batches))

    targets = np.array(targets).flatten()
    print(np.array(predictions).shape)
    print(predictions)
    print(np.mean(np.array(predictions).flatten()))
    print(np.std(np.array(predictions).flatten()))

    fpairs_labels = []
    for p in pairs_labels:
        for pp in p:
            fpairs_labels.append(pp)

    fpairs_fragments = []
    for p in pairs_fragments:
        for pp in p:
            fpairs_fragments.append(pp)

    # ========================
    print("SAVING ... ")
    
    with open(os.path.join(experimentDir, "predictions.pickle"), "wb") as f:
        pickle.dump(predictions, f)
        
    with open(os.path.join(experimentDir, "targets.pickle"), "wb") as f:
        pickle.dump(targets, f)

    
    with open(os.path.join(experimentDir, "fpairs_labels.pickle"), "wb") as f:
        pickle.dump(fpairs_labels, f)

    with open(os.path.join(experimentDir, "fpairs_fragments.pickle"), "wb") as f:
        pickle.dump(fpairs_fragments, f)

    print("Done Saving ")
    # =========================

print("COMPUTING METRICS ...")
fragments_scores = {}
papyrus_scores   = {}
print(len(fpairs_labels))
print(len(predictions))

for i in range(len(fpairs_labels)):
    prediction     = predictions[i]
    target         = targets[i]
    pair_labels    = fpairs_labels[i]
    pair_fragments = fpairs_fragments[i]

    fragment1_id = pair_labels[0]+'_'+pair_fragments[0]
    fragment2_id = pair_labels[1]+'_'+pair_fragments[1]

    papyrus1_id  = pair_labels[0]
    papyrus2_id  = pair_labels[1]
    
    if fragment1_id not in fragments_scores:
        fragments_scores[fragment1_id] = {}
    if fragment2_id not in fragments_scores[fragment1_id]:
        fragments_scores[fragment1_id][fragment2_id] = []
    fragments_scores[fragment1_id][fragment2_id].append(prediction)

    if papyrus1_id not in papyrus_scores:
        papyrus_scores[papyrus1_id] = {}
    if papyrus2_id not in papyrus_scores[papyrus1_id]:
        papyrus_scores[papyrus1_id][papyrus2_id] = []

    papyrus_scores[papyrus1_id][papyrus2_id].append(prediction)

utils.write_hisfrag_csv_and_gt(fragments_scores, experimentDir)

def pr_at_k(sorted_targets, k):

    total_nb_relevant = len(np.array(sorted_targets)[np.array(sorted_targets)==1])
    sorted_targets = sorted_targets[:k]
    nb_relevant = len(np.array(sorted_targets)[np.array(sorted_targets)==1])

    return nb_relevant/(min(total_nb_relevant, k))
        
        
        
        
        
print("Aggregating scores")
        
aps              = []
                 
all_scores       = []
all_targets      = []

top_1_accuracies = []
pr_at_10s        = []
pr_at_100s       = []

for fragment1_id in list(fragments_scores.keys()):
    scores  = []
    targets = []
    for fragment2_id in list(fragments_scores[fragment1_id].keys()):
        predictions = fragments_scores[fragment1_id][fragment2_id]
        # score = np.random.choice(predictions)
        score = np.mean(predictions)
        target = (fragment1_id.split('_')[0] == fragment2_id.split('_')[0])
        scores.append(score)
        targets.append(target)

        
    all_scores.extend(scores)
    all_targets.extend(targets)
    aps.append(sklearn.metrics.average_precision_score(targets, scores))
    
    sorted_scores, sorted_targets = (list(t)[::-1] for t in zip(*sorted(zip(scores, targets))))
    top_1_accuracies.append(sorted_targets[0] == 1)
    pr_at_10s.append(pr_at_k(sorted_targets, 10))
    pr_at_100s.append(pr_at_k(sorted_targets, 100))
    
predictions = np.array(all_scores)
targets     = np.array(all_targets)
print("Making figures")    
make_histogram(predictions, targets, "all_scores_hist")
conf_mat = make_confusion_matrix(predictions, targets, "confusion_matrix")
make_confusion_matrix(predictions, targets, "confusion_matrix_normalized", normalized=True)
make_precision_recall(predictions, targets, "precision_recall")
make_ROC(predictions, targets, "ROC")

fp = conf_mat[0][1]
fn = conf_mat[1][0]
tp = conf_mat[1][1]

f1_score = (tp*1.) / (tp+0.5*(fp+fn))

print("MAP : ", np.round(np.mean(aps), 2))
print("mean TOP 1 accuracy :", np.round(np.mean(top_1_accuracies),2))
print("mean pr@10 : ", np.round(np.mean(pr_at_10s), 2))
print("mean pr@100 : ", np.round(np.mean(pr_at_100s), 2))
print("f1 (threshold 0.8) : ", np.round(f1_score, 2))

f = open(os.path.join(experimentDir, "MAP"), "w")
f.write(str(np.mean(aps)))
f.close()

f = open(os.path.join(experimentDir, "TOP_1_accuracy"), "w")
f.write(str(np.mean(top_1_accuracies)))
f.close()


f = open(os.path.join(experimentDir, "pr@10"), "w")
f.write(str(np.mean(pr_at_10s)))
f.close()

f = open(os.path.join(experimentDir, "pr@100"), "w")
f.write(str(np.mean(pr_at_100s)))
f.close()


f = open(os.path.join(experimentDir, "f1"), "w")
f.write(str(f1_score))
f.close()

f = open(os.path.join(experimentDir, "nb_fragments"), "w")
f.write(str(nb_fragments))
f.close()

f = open(os.path.join(experimentDir, "nb_patches"), "w")
f.write(str(nb_patches))
f.close()
