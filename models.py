from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
import tensorflow.keras.backend as K
# from keras.models import Model
import tensorflow.keras.models
import utils
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

def make_vgg16_branch(input_shape, L2_normalize=False, weights=None):
    print(input_shape)
    inputs = keras.Input(shape=input_shape)
    
    # en fait ça centre/reduit en fonction de imagenet, donc ça a pas de sens dans mon cas
    # x = tf.keras.applications.vgg16.preprocess_input(inputs)
    
    x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(inputs)
    
    v = VGG16(
        include_top=False,
        weights=weights,
        input_shape=input_shape)
    x = v(x)
    # v.summary()
    outputs = layers.Flatten()(x)
    # outputs = Dense(4096, activation='sigmoid',
    #                 kernel_regularizer=l2(1e-3),
    #                 kernel_initializer=initialize_weights,bias_initializer=initialize_bias)(outputs)
    
    
    if L2_normalize:
        outputs = keras.backend.l2_normalize(outputs, axis=0)
        
    return keras.Model(inputs, outputs)

# def initialize_bias(shape, name=None):
#     """
#         The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
#         suggests to initialize CNN layer bias with mean as 0.5 and standard deviation of 0.01
#     """
#     return np.random.normal(loc = 0.5, scale = 1e-2, size = shape)


def make_papy_s_net_branch(input_shape, dropout=None):
    inputs = keras.Input(shape=input_shape)
    # x = utils.data_augmentation(inputs)
    
    # x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(inputs)

    x = inputs
    x = layers.Conv2D(64, 3, padding="same")(x)
    if dropout is not None:
        x = layers.Dropout(dropout)(x)
    # x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    # x = layers.MaxPooling2D(2, padding="same")(x)
    
    x = layers.Conv2D(64, 3, padding="same")(x)
    if dropout is not None:
        x = layers.Dropout(dropout)(x)
    # x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2, padding="same")(x)
    
    x = layers.Conv2D(128, 3, padding="same")(x)
    if dropout is not None:
        x = layers.Dropout(dropout)(x)
    # x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    # x = layers.MaxPooling2D(2, padding="same")(x)
    
    x = layers.Conv2D(128, 3, padding="same")(x)
    if dropout is not None:
        x = layers.Dropout(dropout)(x)
    # x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    
    x = layers.MaxPooling2D(2, padding="same")(x)
    
    x = layers.Conv2D(256, 3, padding="same")(x)
    if dropout is not None:
        x = layers.Dropout(dropout)(x)
    # x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    # x = layers.MaxPooling2D(2, padding="same")(x)
    
    x = layers.Conv2D(256, 3, padding="same")(x)
    if dropout is not None:
        x = layers.Dropout(dropout)(x)
    # x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.MaxPooling2D(2, padding="same")(x)
    
    x = layers.Conv2D(512, 3, padding="same")(x)
    if dropout is not None:
        x = layers.Dropout(dropout)(x)
    # x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    # x = layers.MaxPooling2D(2, padding="same")(x)
    
    x = layers.Conv2D(512, 3, padding="same")(x)
    if dropout is not None:
        x = layers.Dropout(dropout)(x)
    # x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # x = layers.MaxPooling2D(2, padding="same")(x)

    outputs = layers.Flatten()(x)
    return keras.Model(inputs, outputs)

def make_resnet50_branch(input_shape, L2_normalize=False, weights=None):
    inputs = keras.Input(shape=input_shape)
    
    # en fait ça centre/reduit en fonction de imagenet, donc ça a pas de sens dans mon cas
    # x = tf.keras.applications.resnet.preprocess_input(inputs)
    
    x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(inputs)
    r = ResNet50(
        include_top=False,
        weights=weights,
        input_shape=input_shape)
    x = r(x)
    # r.summary()
    outputs = layers.Flatten()(x)
        
    if L2_normalize:
        outputs = keras.backend.l2_normalize(outputs, axis=0)
        
    return keras.Model(inputs, outputs)
    

def make_siamese(input_shape, L2_normalize=False, output_bias=None, network_type="vgg", dropout=None, weights=None):
    assert network_type in ["vgg", "papy", "resnet50"]
    
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    left_input = keras.Input(shape=input_shape)
    right_input = keras.Input(shape=input_shape)

    x_l = utils.data_augmentation(left_input)
    x_r = utils.data_augmentation(right_input)

    if network_type=="vgg":
        branch = make_vgg16_branch(input_shape, L2_normalize=L2_normalize, weights=weights)
    elif network_type=="papy":
        branch = make_papy_s_net_branch(input_shape, dropout=dropout)
    elif network_type=="resnet50":
        branch = make_resnet50_branch(input_shape, L2_normalize=L2_normalize, weights=weights)

    branch.summary()
        
    left_embeddings  = branch(x_l)
    right_embeddings = branch(x_r)

    L1_layer = layers.Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([left_embeddings, right_embeddings])


    # TODO à creuser, est-ce qu'il faut initialiser les biais et les weights manuellement ?
    # prediction = layers.Dense(1,
    #                           activation='sigmoid',
    #                           bias_initializer=initialize_bias)(L1_distance)

    x = layers.Dense(512, activation='relu')(L1_distance)
    # if dropout is not None:
    #     x = layers.Dropout(dropout)(x)
    x = layers.Dense(512, activation='relu')(x)
    prediction = layers.Dense(1, activation='sigmoid', bias_initializer=output_bias)(x)

    siamese_net = keras.Model(inputs=[left_input,right_input],
                              outputs=prediction)

    return siamese_net
    


