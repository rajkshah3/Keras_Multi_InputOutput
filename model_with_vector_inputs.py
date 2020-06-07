import keras
from keras.layers import Input,Dense,Embedding,Concatenate,Flatten
from keras.models import Model
from keras.constraints import UnitNorm

import numpy as np 
tensorboard = keras.callbacks.TensorBoard(log_dir='./model_with_vector_inputs/')

np.random.seed(12)

data_size = 100
input_classes = [6,4,3,4,6]
input_class_data = [
np.random.randint(0,5,data_size),
np.random.randint(0,3,data_size),
np.random.randint(0,2,data_size),
np.random.randint(0,3,data_size),
np.random.randint(0,5,data_size)
]

input_vector_sizes = [4,3,4,6,2]

input_vector_data = [
np.random.randint(0,5,(data_size,1,input_vector_sizes[0])),
np.random.randint(0,3,(data_size,1,input_vector_sizes[1])),
np.random.randint(0,2,(data_size,1,input_vector_sizes[2])),
np.random.randint(0,3,(data_size,1,input_vector_sizes[3])),
np.random.randint(0,5,(data_size,1,input_vector_sizes[4]))
]

output_data = np.random.randint(0,5,data_size)
output_classes = 6


def build_model(input_classes,input_vectors_sizes,output_classes):
    """Build model
    
    Create a model which takes a list of vectors along with a list
    of classes as input. There is a single classifier head.

    Arguments:
        input_classes {[type]} -- Number of one hot encoded inputs
        input_vectors_sizes {[type]} -- Number of vectorised inputs
        output_classes {[type]} -- Number of classes to classify 
    
    Returns:
        [keras.model] -- Compiled model
    """
    dimensions = 20
    inputs = []
    embedded_outputs = []
    for i in input_classes:
        input_layer = Input((1,))
        inputs.append(input_layer)
        embedder = Embedding(input_dim=i,output_dim=dimensions,input_length=1,embeddings_constraint=UnitNorm(axis=0))
        embedded_layer = embedder(input_layer)
        embedded_outputs.append(embedded_layer)
    
    for i in input_vector_sizes:
        input_layer = Input((1,i))
        inputs.append(input_layer)
        embedded_outputs.append(input_layer)


    embedded_concats = Concatenate()(embedded_outputs)
    flatten_layer = Flatten()

    dense_layer = Dense(output_classes,activation='softmax')

    flattened_output = flatten_layer(embedded_concats)
    dense_output = dense_layer(flattened_output)

    # dense_output = dense_layer(embedded_concats)

    model = Model(inputs,dense_output)
    print(model.summary())
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

    return model


def build_model_transform_input_vectors(input_classes,input_vectors_sizes,output_classes):
        """Build model
    
    Create a model which takes a list of vectors along with a list
    of classes as input. There is a single classifier head. this model also
    allows a transformation of the input vectors to be learned during training. 

    Arguments:
        input_classes {[type]} -- Number of one hot encoded inputs
        input_vectors_sizes {[type]} -- Number of vectorised inputs
        output_classes {[type]} -- Number of classes to classify 
    
    Returns:
        [keras.model] -- Compiled model
    """
    dimensions = 20
    inputs = []
    embedded_outputs = []
    for i in input_classes:
        input_layer = Input((1,))
        inputs.append(input_layer)
        embedder = Embedding(input_dim=i,output_dim=dimensions,input_length=1,embeddings_constraint=UnitNorm(axis=0))
        embedded_layer = embedder(input_layer)
        embedded_outputs.append(embedded_layer)
    
    for i in input_vector_sizes:
        input_layer = Input((1,i))
        inputs.append(input_layer)
        dense = Dense(dimensions,activation='linear')
        densed_layer = dense(input_layer)
        embedded_outputs.append(densed_layer)

    embedded_concats = Concatenate()(embedded_outputs)
    flatten_layer = Flatten()

    dense_layer = Dense(output_classes,activation='softmax')

    flattened_output = flatten_layer(embedded_concats)
    dense_output = dense_layer(flattened_output)

    # dense_output = dense_layer(embedded_concats)

    model = Model(inputs,dense_output)
    print(model.summary())
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

    return model

model = build_model_transform_input_vectors(input_classes,input_vector_sizes,output_classes)
# input_data_0 = np.expand_dims(input_data,axis=0)
# input_data_1 = np.expand_dims(input_data,axis=1)
# input_data_2 = np.expand_dims(input_data,axis=2)
import pdb; pdb.set_trace()  # breakpoint de3d9575 //

model.fit(x=input_class_data+input_vector_data,y=output_data,epochs=200,validation_split=0.8,callbacks=[tensorboard])
print('done')