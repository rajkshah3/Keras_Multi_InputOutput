import keras
from keras.layers import Input,Dense,Embedding,Concatenate,Flatten
from keras.models import Model
from keras.constraints import UnitNorm
import tensorflow as tf
import numpy as np 
import sys
def flatten(x):
    """Flatten a tensor.
    # Arguments
        x: A tensor or variable.
    # Returns
        A tensor, reshaped into 1-D
    """
    return tf.reshape(x, [-1])

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

output_classes_1 = 6
output_classes_2 = 8
out_data_1 = np.random.randint(0,output_classes_1,data_size)
out_data_2 = np.random.randint(0,output_classes_2,data_size)



output_data = [out_data_1,out_data_2]


def build_model(input_classes,input_vectors_sizes,output_classes_1,output_classes_2):
    """ Build the model
    
    Build a keras model that with multiple classification heads. It also 
    takes input in the form of classes, one hot encoded. First layer
    is an embedding layer for those classes. This model uses an adaptable
    loss to compile the model which masks specific target classes
    
    Arguments:
        input_classes {[type]} -- 
        input_vectors_sizes {[type]} -- [description]
        output_classes_1 {[number]} -- Classes in classifier 1
        output_classes_2 {[number]} -- Classes in classifier 2
    
    Returns:
        [keras.model] -- Multi classification model
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

    dense_output_1 = Dense(output_classes_1,activation='softmax')
    dense_output_2 = Dense(output_classes_2,activation='softmax')



    flattened_output = flatten_layer(embedded_concats)
    dense_output_1 = dense_output_1(flattened_output)
    dense_output_2 = dense_output_2(flattened_output)

    outputs = [dense_output_1,dense_output_2]
    # dense_output = dense_layer(embedded_concats)
    scale= 0.5

    def out_loss(y_true, y_pred):
        masks = y_true[:,1]
        targets = y_true[:,0]
        # mask = tf.math.equal(value_vec,targets)
        targets = tf.cast(tf.boolean_mask(targets,masks),dtype='int32')
        logits = tf.boolean_mask(y_pred,masks)
        # y_pred  = tf.boolean_mask(y_pred,mask)
        # tf.Print(targets,[targets],output_stream=sys.stdout)
        print(targets)
        # tf.Print(targets,[targets])
        res = scale*tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets,logits=logits)
        return res

    model = Model(inputs,outputs)
    print(model.summary())
    loss = keras.losses.sparse_categorical_crossentropy
    model.compile(loss=[loss,out_loss], optimizer='adam')

    return model


model = build_model(input_classes,input_vector_sizes,output_classes_1,output_classes_2)
# input_data_0 = np.expand_dims(input_data,axis=0)
# input_data_1 = np.expand_dims(input_data,axis=1)
# input_data_2 = np.expand_dims(input_data,axis=2)
import pdb; pdb.set_trace()  # breakpoint de3d9575 //

model.fit(x=input_class_data+input_vector_data,y=output_data,epochs=10,validation_split=0.1)

print('done')