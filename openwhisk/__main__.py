import json
import numpy as np
from PIL import Image
from cv2 import dnn as nn
import time
import base64
import io

def response_message(statusCode, jsonContent):
    return {"statusCode": statusCode,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(jsonContent)}

neural_network = nn.readNetFromCaffe("./neuralNetwork/MobileNetSSD_deploy.prototxt.txt",
                                     "./neuralNetwork/MobileNetSSD_deploy.caffemodel")

classNames = ["background", "aeroplane", "bicycle", "bird", "boat",
              "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
              "dog", "horse", "motorbike", "person", "pottedplant",
              "sheep", "sofa", "train", "tvmonitor"]

print "Done preparing neural network!!"

def main(args):
    #payload = json.loads(args["body"])
    # image = Image.open(io.BytesIO(base64.b64decode(payload["image"])))
    image = Image.open("./testImages/person.jpg")
    open_cv_image = np.array(image)
    # fit neural network required input
    input_blob = nn.blobFromImage(open_cv_image,
                                       # scale factor
                                       0.007843,
                                       # input width and input height
                                       (300, 300),
                                       # mean values which are subtracted from channels
                                       (127.5, 127.5, 127.5),
                                       # swap red and blue
                                       False,
                                       # crop
                                       False)
    start_millis = int(round(time.time() * 1000))
    neural_network.setInput(input_blob)
    # network output is (1, 1, numDetectedObjects, 7), reshape in (numDetectedObjects, 7)
    detections = neural_network.forward()[0][0]
    millis = int(round(time.time() * 1000)) - start_millis
    print "OpenCV Neural network computation: {} milliseconds".format(millis)

    contents = []
    for elem in detections:
        # elem[1]: is the classId in classNames list
        name = classNames[int(elem[1])]
        # sorting elem[2:] descenting and take highest confidence
        confidence = 100 * sorted(elem[2:], reverse = True)[0]
        contents.append({"Name": name, "Confidence": confidence})

    return response_message(200, {'Labels': contents})
