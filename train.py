import argparse
import os
import datetime
import matplotlib.pyplot as plt
now = datetime.datetime.now()
import io
import seaborn as sn
import pandas as pd
import sklearn.metrics

def typeDir(str):
    if(not os.path.isdir(str)):
        raise argparse.ArgumentTypeError("{0} is not a directory.".format(str))
    return str

argParser = argparse.ArgumentParser(description='Training')
argParser.add_argument('-g', '--gpuNb', type=int, required=True, help="gpu id for training")
argParser.add_argument('-da', '--dataset', type=typeDir, required=True, help="path to dataset (directory containing .pickle files)")
argParser.add_argument('-d', '--saveDir', type=str, required=True, help="where to save the training")
argParser.add_argument('-f', '--fineTune', type=str, required=False, default=None,  help="if fine tuning, path to .h5 starting file")
args = argParser.parse_args()

saveDir = args.saveDir
logdir  = str(args.saveDir)+"/logs/scalars/" + now.strftime("%Y%m%d-%H%M%S")

os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpuNb)

import numpy as np
import cv2
import models
import pickle
import time
from tensorflow import keras
from tensorflow.keras.models import load_model
import tensorflow as tf
import utils
import batch_utils
import backtrace

backtrace.hook(
    reverse=False,
    align=True,
    strip_path=True,
    enable_on_envvar_only=False,
    on_tty=False,
    conservative=False,
    styles={})

with open(os.path.join(args.dataset, "dataset_train.pickle"), "rb") as f:
    dataset_train = pickle.load(f)

with open(os.path.join(args.dataset, "dataset_val.pickle"), "rb") as f:
    dataset_val = pickle.load(f)

    
sample_im_shape = dataset_train[list(dataset_train.keys())[0]][0].shape
input_shape = (sample_im_shape[1], sample_im_shape[2], sample_im_shape[3])
    
# if input_shape[2] == 1:
#     print("CONVERTING GRAYSCALE TO RGB (by duplicating channels) ...")
#     # convert grayscale to RGB by duplicating channels for resnet and vgg
#     for papyrus_id, fragments in dataset_train.items():
#         for i in range(len(fragments)):
#             fragment = fragments[i]
#             new_fragment = np.zeros((fragment.shape[0], fragment.shape[1], fragment.shape[2], 3))
#             new_fragment[:,:,:,0] = fragment[:,:,:,0]
#             new_fragment[:,:,:,1] = fragment[:,:,:,0]
#             new_fragment[:,:,:,2] = fragment[:,:,:,0]
#             dataset_train[papyrus_id][i] = new_fragment
            
#     for papyrus_id, fragments in dataset_val.items():
#         for i in range(len(fragments)):
#             fragment = fragments[i]
#             new_fragment = np.zeros((fragment.shape[0], fragment.shape[1], fragment.shape[2], 3))
#             new_fragment[:,:,:,0] = fragment[:,:,:,0]
#             new_fragment[:,:,:,1] = fragment[:,:,:,0]
#             new_fragment[:,:,:,2] = fragment[:,:,:,0]
#             dataset_val[papyrus_id][i] = new_fragment
            
#     input_shape = (64, 64, 3)
    
# === mnist ====================================
# input_shape = (28, 28, 1)
    
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path="mnist.npz")
# dataset_train = {}
# for i in range(len(x_train)):
#     label = y_train[i]
#     if label not in dataset_train:
#         dataset_train[label] = []
#     dataset_train[label].append(x_train[i])

# dataset_val = {}
# for i in range(len(x_test)):
#     label = y_test[i]
#     if label not in dataset_val:
#         dataset_val[label] = []
#     dataset_val[label].append(x_test[i])
# ==============================================

currentEpoch = 0

# === Training Hyperparameters ===
batch_size      = 64
nb_epochs       = 1000
learning_rate   = 1e-6 # overriden if the LearningRateScheduler is active
network_type    = "vgg"
# ===============================




if args.fineTune is not None:
    model = load_model(args.fineTune)
    # for layer in model.layers:
    #     print(layer.name)
    #     if "dense" not in layer.name:
    #         layer.trainable = False
    
    # model.load_weights(args.fineTune)

    # model_input_shape = tuple(model.layers[0].get_output_at(0).get_shape().as_list()[1:])
    
    # if model_input_shape[2] != input_shape[2]:
    #     # if finetune grayscale to color or color to grayscale
    #     model.layers.pop(0)
    #     left_input = keras.Input(shape=input_shape)
    #     right_input = keras.Input(shape=input_shape)
    #     inputs = [left_input,right_input]
    #     outputs = model(inputs)
    #     model = keras.Model(inputs, outputs)
else:
    model = models.make_siamese(input_shape, L2_normalize=False, network_type=network_type)#, weights="imagenet")#, dropout=0.5)

    


    

model.summary()

class model_predict_callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        batch_size = 10
        pairs, targets = batch_utils.get_batch(batch_size, dataset_val, input_shape, currentEpoch)
        print(model.predict((pairs, targets), batch_size=batch_size, steps=batch_size))
        print(targets)


class tensorboard_plot_callback(tf.keras.callbacks.Callback):

    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        assert self.mode in ["histogram", "confusion", "precision_recall", "ROC", "n_way", "map"]

    def make_map_tests(self):
        n_values = [x for x in range(2, 20, 1)]
        nb_trials_per_n = 10

        results = []
        for n in n_values:
            correct = 0
            map = []
            for t in range(nb_trials_per_n):
                n_way_pairs, n_way_targets = batch_utils.get_map_task_val(n, dataset_val, input_shape)

                predictions = model.predict(n_way_pairs, batch_size=n, steps=1)

                ap = utils.my_average_precision_at_k(n_way_targets, predictions, n)
                map.append(ap)
                
            results.append(np.mean(map))
            
        return results, n_values
                

    def get_map_curve_im(self):
        results, n_values = self.make_map_tests()
        
        plt.cla()
        plt.figure(figsize = (10,7))
        plt.plot(n_values, results)
        plt.xlabel('n')
        plt.ylabel('map')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        tf_img = tf.image.decode_png(buf.getvalue(), channels=4)
        tf_img = tf.expand_dims(tf_img, 0)
        plt.close()
        return tf_img
        
    def make_n_way_tests(self):
        n_values = [x for x in range(2, 20, 1)]
        nb_trials_per_n = 10

        results = []
        for n in n_values:
            correct = 0
            for t in range(nb_trials_per_n):
                n_way_pairs, n_way_targets = batch_utils.get_n_way_task_val(n, dataset_val, input_shape)

                predictions = model.predict(n_way_pairs, batch_size=n, steps=1)
                
                # n_way_predictions_left = model.predict(n_way_pairs[0], batch_size=n, steps=1)
                # n_way_predictions_right = model.predict(n_way_pairs[1], batch_size=n, steps=1)

                # dists = np.linalg.norm(n_way_predictions_left-n_way_predictions_right, axis=1)

                index_max = np.argmax(predictions)
                if int(n_way_targets[index_max]) == 1:
                    correct += 1
            results.append((correct/(nb_trials_per_n*1.)))
            
        return results, n_values
                

    def get_n_way_curve_im(self):
        results, n_values = self.make_n_way_tests()
        
        plt.cla()
        plt.figure(figsize = (10,7))
        plt.plot(n_values, results)
        plt.xlabel('n')
        plt.ylabel('correct percentage')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        tf_img = tf.image.decode_png(buf.getvalue(), channels=4)
        tf_img = tf.expand_dims(tf_img, 0)
        plt.close()
        return tf_img
        
    def get_histogram_im(self, predictions, targets):
        predictions_true  = predictions[targets==1]
        predictions_false = predictions[targets==0]

        plt.cla()
        fig,ax = plt.subplots(1,1)
        ax.hist(predictions_false, label="predictions false")
        ax.hist(predictions_true, label="predictions true", fc=(1, 0, 0, 0.5))
        
        ax.set_title("histogram of result")
        ax.set_xlabel('score')
        ax.set_ylabel('nb scores')
        plt.legend()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        tf_img = tf.image.decode_png(buf.getvalue(), channels=4)
        tf_img = tf.expand_dims(tf_img, 0)
        plt.close()
        return tf_img

    def get_confusion_matrix_im(self, predictions, targets, normalized=False):
        threshold = 0.5
        
        if normalized :
            confusion_mat = sklearn.metrics.confusion_matrix(targets, [1 if x>threshold else 0 for x in predictions], normalize='pred')
        else:
            confusion_mat   = sklearn.metrics.confusion_matrix(targets, [1 if x>threshold else 0 for x in predictions])

        df_cm = pd.DataFrame(confusion_mat,
                             index = ["TN, FP", "FN, TP"],
                             columns = [" ", " "])
        plt.cla()
        plt.figure(figsize = (10,7))
        if normalized:
            sn.heatmap(df_cm, annot=True, fmt="f")
        else:
            sn.heatmap(df_cm, annot=True, fmt="d")

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        tf_img = tf.image.decode_png(buf.getvalue(), channels=4)
        tf_img = tf.expand_dims(tf_img, 0)
        plt.close()
        return tf_img

    def get_precision_recall_im(self, predictions, targets):
        
        precision, recall, thresholds = sklearn.metrics.precision_recall_curve(targets, predictions)
        plt.cla()
        plt.figure(figsize = (10,7))
        plt.plot(recall, precision)
        plt.xlabel('Recall')
        plt.ylabel('precision')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        tf_img = tf.image.decode_png(buf.getvalue(), channels=4)
        tf_img = tf.expand_dims(tf_img, 0)
        plt.close()
        return tf_img

    def get_ROC_im(self, predictions, targets):

        fpr, tpr, thresholds = sklearn.metrics.roc_curve(targets, predictions)        
        plt.cla()
        plt.figure(figsize = (10,7))
        plt.plot(fpr, tpr)
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        tf_img = tf.image.decode_png(buf.getvalue(), channels=4)
        tf_img = tf.expand_dims(tf_img, 0)
        plt.close()
        return tf_img
    

        
    def on_epoch_end(self, epoch, logs={}):
        if epoch%5 != 0:
            return 
        
        pairs, targets    = batch_utils.get_batch(batch_size, dataset_val, input_shape, currentEpoch)
        predictions       = model.predict(pairs, batch_size=batch_size, steps=1).flatten()

        if self.mode == "histogram":
            tf_img = self.get_histogram_im(predictions, targets)
        elif self.mode == "confusion":
            tf_img   = self.get_confusion_matrix_im(predictions, targets)
            tf_img_n = self.get_confusion_matrix_im(predictions, targets, normalized=True)
        elif self.mode == "precision_recall":
            tf_img = self.get_precision_recall_im(predictions, targets)
        elif self.mode == "ROC":
            tf_img = self.get_ROC_im(predictions, targets)
        elif self.mode == "n_way":
            tf_img = self.get_n_way_curve_im()
        elif self.mode == "map":
            tf_img = self.get_map_curve_im()
            
            
        file_writer = tf.summary.create_file_writer(str(saveDir)+"/logs/")
        with file_writer.as_default():
            tf.summary.image(self.mode, tf_img, step=epoch)
            if self.mode == "confusion":
                tf.summary.image(self.mode+'_normalized', tf_img_n, step=epoch)

    
class CurrentEpochCallback(tf.keras.callbacks.Callback):
    global currentEpoch
    def on_epoch_end(self, epoch, logs={}):
        currentEpoch = epoch

tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)    


nb_images_train = 0
for k, v in dataset_train.items():
    nb_images_train += len(v)

nb_images_val = 0
for k, v in dataset_val.items():
    nb_images_val += len(v)


steps_per_epoch = nb_images_train//batch_size
validation_step = nb_images_val//batch_size


callbacks = [
    keras.callbacks.ModelCheckpoint(str(args.saveDir)+"/logs/save_at_{epoch}.h5", save_freq=(steps_per_epoch*5)),
    tensorboard_callback,
    # model_predict_callback(),
    CurrentEpochCallback(),
    tensorboard_plot_callback("histogram"),
    tensorboard_plot_callback("confusion"),
    tensorboard_plot_callback("precision_recall"),
    tensorboard_plot_callback("ROC"),
    tensorboard_plot_callback("n_way"),
    tensorboard_plot_callback("map"),
    keras.callbacks.LearningRateScheduler(utils.lr_schedule, verbose=0)
]

thresholds = []
for i in range(0, 101, 1):
    thresholds.append(i/100)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate),
    loss="binary_crossentropy",
    # loss = utils.my_histogram_loss,
    metrics=["accuracy",
             # tf.keras.metrics.Precision(thresholds=thresholds),
             # tf.keras.metrics.Recall(thresholds=thresholds),
             # tf.keras.metrics.TruePositives(thresholds=thresholds),
             # tf.keras.metrics.TrueNegatives(thresholds=thresholds),
             # tf.keras.metrics.FalsePositives(thresholds=thresholds),
             # tf.keras.metrics.FalseNegatives(thresholds=thresholds),
             # utils.keras_mean_average_precision(10)
    ]
)

training_history = model.fit(
    x                = batch_utils.generate_class_controled(batch_size, dataset_train, input_shape, currentEpoch),
    # x                = utils.generate_random(batch_size, dataset_train),
    steps_per_epoch  = steps_per_epoch,
    epochs           = nb_epochs,
    validation_data  = batch_utils.generate_class_controled(batch_size, dataset_val, input_shape, currentEpoch),
    # validation_data  = utils.generate_random(batch_size, dataset_val),
    validation_steps = validation_step,
    callbacks        = callbacks,
    # class_weight     = {0:0.5, 1:1}
)

# https://towardsdatascience.com/one-shot-learning-with-siamese-networks-using-keras-17f34e75bb3d

# https://github.com/hlamba28/One-Shot-Learning-with-Siamese-Networks/blob/master/Siamese%20on%20Omniglot%20Dataset.ipynb
