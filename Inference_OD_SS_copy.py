#!/usr/bin/env python
# coding: utf-8

# ### Load tensorRT graph

# In[1]:

import tensorflow as tf
from tensorflow.python.platform import gfile

from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast


import cv2
import numpy as np
#import matplotlib.pyplot as plt

##Uncomment the below import if you don't want the frames to be updated on jupyter notebook or if you are using opencv frames to visualize
#from IPython.display import clear_output

from tensorflow.python.keras.backend import set_session
graph = tf.get_default_graph()

#GRAPH_PB_PATH = './trained_models_local/saved_for_lab/tf_model_base_1502.pb'
#GRAPH_PB_PATH_OD = './converted_trt_graph_od/trt_graph_base_30.pb'
GRAPH_PB_PATH_OD='./frozen_model_od/tf_ssd7_model.pb'
#GRAPH_PB_PATH = './converted_trt_graph/trt_graph_st_prg_1601_80p.pb'
#GRAPH_PB_PATH = './converted_trt_graph/trt_graph_quantized.pb'
#GRAPH_PB_PATH_FROZEN_SS='./converted_trt_graph_ss/trt_graph_ss_model.pb'
GRAPH_PB_PATH_FROZEN_SS='./frozen_model_ss/frozen_model_ss_plf.pb'


#loading the graph for OD
with tf.Session() as sess1:
   print("load graph_OD")
   with gfile.FastGFile(GRAPH_PB_PATH_OD,'rb') as f:
       graph_def1 = tf.GraphDef()
   graph_def1.ParseFromString(f.read())
   sess1.graph.as_default()
   tf.import_graph_def(graph_def1, name='')
   graph_nodes1=[n for n in graph_def1.node]
   names = []
   for t in graph_nodes1:
      names.append(t.name)


#loading the graph for SS
with tf.Session() as sess2:
   print("load graph_SS")
   with gfile.FastGFile(GRAPH_PB_PATH_FROZEN_SS,'rb') as f:
       graph_def2 = tf.GraphDef()
   graph_def2.ParseFromString(f.read())
   sess2.graph.as_default()
   tf.import_graph_def(graph_def2, name='')
   graph_nodes2=[n for n in graph_def2.node]
   names = []
   for t in graph_nodes2:
      names.append(t.name)
    # print operations

   #                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            print(names)


# ### Loading the pb graph

# In[3]:


tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = False

tf_sess1 = tf.Session(config=tf_config)
tf.import_graph_def(graph_def1, name='')


tf_sess2 = tf.Session(config=tf_config)
tf.import_graph_def(graph_def2, name='')
# In[ ]:

'''
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print('done')
'''

# ### Imports for the y_decoded_pred[]

# In[2]:


#import os
#os.chdir('/home/agxdbot/github/ssd_keras/')
#goes a directory up and imports the below
#change the directory back to Jetson
#os.chdir('/home/agxdbot/github/Inference_OD_SS')


# In[4]:


tf_input1 = tf_sess1.graph.get_tensor_by_name('input_1:0')
print(tf_input1)
tf_predictions1 = tf_sess1.graph.get_tensor_by_name('predictions/concat:0')
print(tf_predictions1)


# In[ ]:


tf_input2 = tf_sess2.graph.get_tensor_by_name('input_1:0')
print('Tensor-2',tf_input2)

tf_predictions2 = tf_sess2.graph.get_tensor_by_name('sigmoid/Sigmoid:0')
print(tf_predictions2)


# ### Inference on live camera data with pb graph

# In[ ]:





## Drawing a bounding box around the predictions

classes = ['background', 'car', 'truck', 'pedestrian', 'bicyclist', 'light'] # Just so we can print class names onto the image instead of IDs
font = cv2.FONT_HERSHEY_SIMPLEX
  

# fontScale
fontScale = 0.5
   
# Blue color in BGR
color = (255, 255, 0)
  
# Line thickness of 2 px
thickness = 1


#Capture the video from the camera

def model_OS(img_os):
    try:
        image_resized2 = cv2.resize(img_os, (480, 300))
        with graph.as_default():
            set_session(sess1)
            inputs1, predictions1 = tf_sess1.run([tf_input1, tf_predictions1], feed_dict={
            tf_input1: image_resized2[None, ...]
        })

        y_pred_decoded = decode_detections(predictions1,
                                       confidence_thresh=0.5,
                                       iou_threshold=0.45,
                                       top_k=200,
                                       normalize_coords=True,
                                       img_height=300,
                                       img_width=480)
        np.set_printoptions(precision=2, suppress=True, linewidth=90)

        for box in y_pred_decoded[0]:
            xmin = box[-4]
            ymin = box[-3]
            xmax = box[-2]
            ymax = box[-1]
            label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
            # cv2.rectangle(im2, (xmin,ymin),(xmax,ymax), color=color, thickness=2 )
            cv2.rectangle(image_resized2, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color=(0, 255, 0), thickness=2)
            cv2.putText(image_resized2, label, (int(xmin), int(ymin)), font, fontScale, color, thickness)
        return image_resized2


    except:
        print("Error in model_OS")

def model_SS(img_ss):
    try:
        image_resized3 = cv2.resize(img_ss, (480, 320))
        with graph.as_default():
            set_session(sess2)
            inputs2, predictions2 = tf_sess2.run([tf_input2, tf_predictions2], feed_dict={
            tf_input2: image_resized3[None, ...]
        })

        print(predictions2)
        #cv2.imwrite('file5.jpeg', 255*predictions.squeeze())
        pred_image = 255*predictions2.squeeze()

        ##converts pred_image to CV_8UC1 format so that ColorMap can be applied on it
        u8 = pred_image.astype(np.uint8)

        #Color map autumn is applied to the CV_8UC1 pred_image
        im_color = cv2.applyColorMap(u8, cv2.COLORMAP_AUTUMN)
        #cv2.imshow('input image', image_resized2)
        cv2.imshow('prediction mask',im_color)

        cv2.waitKey(0)
        if cv2.waitKey(0):
            return False

    except:
        print("Error in model_SS")


