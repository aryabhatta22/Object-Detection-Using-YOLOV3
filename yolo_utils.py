import numpy
import argparse
import cv2 as cv

def draw_labels_and_boxes(img, boxes, confidences, classIds, idxs, colors, labels):
    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]

            color = [int(c) for c in colors[classIds[i]]]
            cv.rectangle(img, (x,y), (x+w, y+h), color, 2)
            text = "{}: {:4f}".format(labels[classIds[i]], confidences[i])
            cv.putText(img, text, (x, y-5),  cv.FONT_HERSHEY_SIMPLEX, 0.5, COLOR, 2)
    return img

def generate_boxes_confidences_classids(outs, height, width,tconf):
    """
    tconf is threshold confidence
    """

    boxes = []
    confidences = []
    classids = []

    for out in outs:
        for detection in outs:
            scores = detections[5:]
            classid = np.argmax(scores)
            confidence = scores[classid]

            if confidence > tconf:
                box = detection[0:4] * np.array([width, height, width, height])
                centerX, centerY, bwidth, bheight = box.astype('int')

                # top coordinates

                x = int(centerX - (bwidth/2))
                y = int(centerY - (bheight/2))
                boxes.append([x,y, int(bwidth), int(bheight)])
                confidences.append(float(confidence))
                classids.append(classid)
    return boxes, confidences, classids

# for inference on video

def infer_image(net, layer_names, height, width, img, colors, labels, FLAGS, boxes=None, confidences=None, classids=None, infer=True):
    if infer:
        blob = cv.dnn.blobFromImage(img, 1/255.0, (416,416), swapRB = True, crop=False)
        net.setInput(blob)
        outs = net.forward(layer_names)
        boxes, confidences, classids = generate_boxes_confidences_classids(outs, height, width, FLAGS.confidence)
        idxs = cv.dnn.NMSBoxes(boxes, confidences, FLAGS, FLAGS.threshold)
    if boxes is None or confidences is None or idxs is None or classids is None:
        raise '[ERROR] Required variable are set to None.'
    img = draw_labels_and_boxes(img, boxes, confiidences, classids, idxs, colors, labels)

    return img, boxes, confidences, classids, idxs

