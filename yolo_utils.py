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