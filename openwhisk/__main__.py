import numpy as np
import cv2
import base64

def main(args):
    neural_network = cv2.dnn.readNetFromCaffe("./imageRecognition/MobileNetSSD_deploy.prototxt.txt",
                                          "./imageRecognition/MobileNetSSD_deploy.caffemodel")

    classNames = ["background", "aeroplane", "bicycle", "bird", "boat",
              "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
              "dog", "horse", "motorbike", "person", "pottedplant",
              "sheep", "sofa", "train", "tvmonitor"]
    image = cv2.imdecode(np.fromstring(base64.b64decode(args["image"]), dtype=np.uint8),1)
    input_blob = cv2.dnn.blobFromImage(image, 0.007843,(300, 300), (127.5, 127.5, 127.5), False, False)
    neural_network.setInput(input_blob)
    detections = neural_network.forward()[0][0]
    contents = []
    for elem in detections:
        name = classNames[int(elem[1])]
        confidence = 100 * sorted(elem[2:], reverse = True)[0]
        if confidence > 100:
           confidence = 100
        contents.append({"Name": name, "Confidence": confidence})

    return {"statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": {'Labels': contents}}

# if __name__== "__main__":
#    main({})
