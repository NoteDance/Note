import tensorflow as tf
import pickle
    
def restore(path):
    input_file=open(path,'rb')
    model=pickle.load(input_file)
    optimizer=tf.keras.optimizers.deserialize(pickle.load(input_file))
    input_file.close()
    return model,optimizer