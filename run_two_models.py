
#Uncomment if you are using TF1
#import Inference_OD_SS

#Uncomment if you are using TF2
import Inference_OD_SS_TF2
import cv2

image_os = cv2.imread("./1478899365487445082.jpg", cv2.IMREAD_UNCHANGED)
image_ss = cv2.imread("./0001TP_009060.png", cv2.IMREAD_UNCHANGED)

image_os = cv2.resize(image_os, (480, 300))
image_ss = cv2.resize(image_ss, (480, 320))

#Uncomment if you are using TF1
#a = Inference_OD_SS.model_OS(image_os)
#b = Inference_OD_SS.model_SS(image_ss)

#Uncomment if you are using TF2
a = Inference_OD_SS_TF2.model_OS(image_os)
b = Inference_OD_SS_TF2.model_SS(image_ss)

print(b)

cv2.imshow('Input Images',a)
cv2.imshow('prediction mask',b)
cv2.waitKey(0)

cv2.destroyAllWindows()
