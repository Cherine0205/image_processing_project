from keras.utils import plot_model
#from unet import *

def model_structure(model,output_name):
    plot_model(model, to_file=output_name+'.png',show_shapes=True)
