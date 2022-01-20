#!/usr/bin/env python
# coding: utf-8

# ### Load tensorRT graph

# In[1]:

import tensorflow as tf
from tensorflow.python.platform import gfile

import cv2
import numpy as np

from tensorflow.python.keras.backend import set_session
graph = tf.get_default_graph()

GRAPH_PB_PATH_FROZEN_SS='./frozen_model_ss/frozen_model_ss_plf.pb'


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

# ### Loading the pb graph

# In[3]:


tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = False

tf_sess2 = tf.Session(config=tf_config)
tf.import_graph_def(graph_def2, name='')
# In[ ]:


tf_input2 = tf_sess2.graph.get_tensor_by_name('input_1:0')
print('Tensor-2',tf_input2)

tf_predictions2 = tf_sess2.graph.get_tensor_by_name('sigmoid/Sigmoid:0')
print(tf_predictions2)




def model_SS(img_ss):
    try:
        with graph.as_default():
            set_session(sess2)
            inputs2, predictions2 = tf_sess2.run([tf_input2, tf_predictions2], feed_dict={
            tf_input2: img_ss[None, ...]
        })


        pred_image = 255*predictions2.squeeze()


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


