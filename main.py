import numpy as np
import time
import cv2
import sys

from haar import detect_and_blur_plate

# dataset source
# https://graphics.cs.msu.ru/ru/node/1266


# yolo4-tiny
LABELS_FILE = 'yolo4/obj.names'
CONFIG_FILE = 'yolo4/yolov4-tiny-mycustom.cfg'
WEIGHTS_FILE = 'yolo4/yolov4-tiny-mycustom_best.weights'

# classes labels for png classes representation
classes_dict = {}
with open('yolo4/classes_map.txt') as f:
    for line in f:
        (key, val) = line.split()
        classes_dict[int(key)] = val

CONFIDENCE_THRESHOLD = 0.3

LABELS = open(LABELS_FILE).read().strip().split("\n")

# colors for bounding boxes
np.random.seed(4)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# get model
net = cv2.dnn.readNetFromDarknet(CONFIG_FILE, WEIGHTS_FILE)

cap = cv2.VideoCapture("yolo4/video.mp4")  # use cv2.VideoCapture(0) for getting video from default camera
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print('frames_count', length)

# Always a good idea to check if the video was acutally there
# If you get an error at thsi step, triple check your file path.
if cap.isOpened() == False:
    print("Error opening the video file. Please double check your "
          "file path for typos. Or move the movie file to the same location as this script/notebook")
    sys.exit()

# variables for creating result video
img_array = []
size = ()

# start detecting process
while cap.isOpened():
    # Read the video file.
    ret, image = cap.read()

    # If we got frames, show them.
    if ret == True:

        # blur car numbers on video
        image = detect_and_blur_plate(image)

        (H, W) = image.shape[:2]
        size = (W, H)

        # determine only the *output* layer names that we need from YOLO
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                     swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()
        print("[INFO] YOLO took {:.6f} seconds".format(end - start))

        # initialize our lists of detected bounding boxes, confidences, and
        # class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []

        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability) of
                # the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > CONFIDENCE_THRESHOLD:
                    # scale the bounding box coordinates back relative to the
                    # size of the image, keeping in mind that YOLO actually
                    # returns the center (x, y)-coordinates of the bounding
                    # box followed by the boxes' width and height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top and
                    # and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates, confidences,
                    # and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # apply non-maxima suppression to suppress weak, overlapping bounding
        # boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD,
                                CONFIDENCE_THRESHOLD)

        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

                img = cv2.imread(r'sign-labels-png/' + classes_dict.get(int(LABELS[classIDs[i]])) + '.png')
                if img is not None:
                    img = cv2.resize(img, (w, h))

                    # check if we can draw label image (border of frame)
                    if y > 0:
                        if x - w >= 0:
                            image[y:(y + h), (x - w):x, :] = img[:, :, :]
                        else:
                            image[y:(y + h), (x + w):(x + 2 * w), :] = img[:, :, :]

            # show confidence and label
            # text = "{}: {:.4f}".format(classes_dict.get(int(LABELS[classIDs[i]])), confidences[i])
            # cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
            #             0.5, color, 2)

        # append data for video
        img_array.append(image)

        #########################################
        # realtime detection
        #cv2.imshow('frame', image)

        # Press q to quit
        #if cv2.waitKey(25) & 0xFF == ord('q'):
        #    break
        #########################################
    else:
        break

print("saving video...")
out = cv2.VideoWriter('yolo4/result.avi', cv2.VideoWriter_fourcc(*'DIVX'), 18, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()

cap.release()
# Closes all the frames
cv2.destroyAllWindows()
