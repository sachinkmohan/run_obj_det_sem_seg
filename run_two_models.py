
import Inference_OD_SS_copy
import cv2

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        #a = Inference_OD_SS_copy.model_OS(frame)
        b = Inference_OD_SS_copy.model_SS(frame)
        
        '''
        if a == False:
            print('Object Detection complete')
            break
        elif b == False:
            print('Semantic Segmentation complete')
            break
        '''
    else:
        cap.release()
        break

cap.release()
cv2.destroyAllWindows()
