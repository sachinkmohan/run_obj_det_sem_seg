

#Uncomment the below if you are using TF1
#import Inference_Only_SS

#Uncomment the below if you are using TF2
import Inference_Only_SS_TF2

import cv2

image_os = cv2.imread("./1478899365487445082.jpg", cv2.IMREAD_UNCHANGED)
image_ss = cv2.imread("./0001TP_009060.png", cv2.IMREAD_UNCHANGED)


image_os = cv2.resize(image_os, (480, 300))
image_ss = cv2.resize(image_ss, (480, 320))


#Uncomment the below if you are using TF1
#b = Inference_Only_SS.model_SS(image_ss)


#Uncomment the below if you are using TF2
b = Inference_Only_SS_TF2.model_SS(image_ss)

cv2.destroyAllWindows()
