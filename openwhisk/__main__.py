import numpy as np
import cv2
import base64

neural_network = cv2.dnn.readNetFromCaffe("./imageRecognition/MobileNetSSD_deploy.prototxt.txt",
                                          "./imageRecognition/MobileNetSSD_deploy.caffemodel")

classNames = ["background", "aeroplane", "bicycle", "bird", "boat",
              "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
              "dog", "horse", "motorbike", "person", "pottedplant",
              "sheep", "sofa", "train", "tvmonitor"]

def main(args):
    image = cv2.imdecode(np.fromstring(base64.b64decode(args["image"]), dtype=np.uint8),1)
    # image = cv2.imread("./testImages/boat.jpg")
    # fit neural network required input
    input_blob = cv2.dnn.blobFromImage(image,
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
    # start_millis = int(round(time.time() * 1000))
    neural_network.setInput(input_blob)
    # network output is (1, 1, numDetectedObjects, 7), reshape in (numDetectedObjects, 7)
    detections = neural_network.forward()[0][0]
    # millis = int(round(time.time() * 1000)) - start_millis
    # print "OpenCV Neural network computation: {} milliseconds".format(millis)

    contents = []
    for elem in detections:
        # elem[1]: is the classId in classNames list
        name = classNames[int(elem[1])]
        # sorting elem[2:] descenting and take highest confidence
        confidence = 100 * sorted(elem[2:], reverse = True)[0]
        contents.append({"Name": name, "Confidence": confidence})

    return {"statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": {'Labels': contents}}

# if __name__== "__main__":
#    main({})
