from keras.preprocessing.image import img_to_array
from keras.models import load_model
import cv2
import numpy as np
# load the trained convolutional neural network to read number plate
print("[INFO] loading Network...")


LABELS = ["0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H",
              "I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]


def detectObject(image):
    model = load_model("D:/Web/flask_np/app/all_models/np_read_model/NumberPlate.model")
    # pre-process the image for classification
    image = cv2.resize(image, (50, 100))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    
    model_val = model.predict(image)
    maxProb = np.argmax(model_val)
    print("Prob", LABELS[maxProb])

    return model_val


image = cv2.imread("D:/Number Plate Recognition/Car Images Number Plate Detection/tObjects/2/511img.jpg")
detectObject(image)