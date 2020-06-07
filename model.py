import keras
from keras.layers import Input,Dense,Embedding,Concatenate,Flatten
from keras.models import Model
from keras.constraints import UnitNorm

import numpy as np
 
np.random.seed(12)
data_size = 100
input_classes = [6,4,3,4,6]
input_data = [
np.random.randint(0,5,data_size),
np.random.randint(0,3,data_size),
np.random.randint(0,2,data_size),
np.random.randint(0,3,data_size),
np.random.randint(0,5,data_size)
]

output_data = np.random.randint(0,5,data_size)
output_classes = 6


def build_model(input_classes,output_classes):
    """Build model
    
    Builds a simple model with classes as input and output
    
    Arguments:
        input_classes {[list]} -- List of input classes as features
        output_classes {[int]} -- Number of output classes to classify 
    
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

    embedded_concats = Concatenate()(embedded_outputs)
    flatten_layer = Flatten()

    dense_layer = Dense(output_classes)

    flattened_output = flatten_layer(embedded_concats)
    dense_output = dense_layer(flattened_output)

    # dense_output = dense_layer(embedded_concats)

    model = Model(inputs,dense_output)
    print(model.summary())
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

    return model


model = build_model(input_classes,output_classes)
# input_data_0 = np.expand_dims(input_data,axis=0)
# input_data_1 = np.expand_dims(input_data,axis=1)
# input_data_2 = np.expand_dims(input_data,axis=2)
import pdb; pdb.set_trace()  # breakpoint de3d9575 //

model.fit(x=input_data,y=output_data,epochs=20)
print('done')