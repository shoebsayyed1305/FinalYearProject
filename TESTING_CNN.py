import cv2
import numpy as np
import time

from keras.models import load_model

model = load_model('NN.h5')
classes = ['Best','Good', 'Worse']

def get(file_path):

        img = cv2.imread(file_path)
        height, width = img.shape[:2]
        img = cv2.resize(img, (100,100))

        # predict!
        roi_X = np.expand_dims(img, axis=0)
        predictions = model.predict(roi_X)

        #print(np.argmax(predictions[0]))
        result_index = np.argmax(predictions[0])
        result  = classes[result_index]
        return result




#bt_result,bt_stage,bt_symptoms = get('./data/BT/T10.png')


