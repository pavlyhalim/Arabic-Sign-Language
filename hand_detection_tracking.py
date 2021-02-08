import cv2 as cv
from scipy.spatial import distance
from utils import detector_utils as detector_utils
import numpy as np
from collections import OrderedDict
detection_graph, sess = detector_utils.load_inference_graph()
class Tracker:
    def __init__(self, maxLost = 30):           # maxLost: maximum object lost counted when the object is being tracked
        self.nextObjectID = 0                   # ID of next object
        self.objects = OrderedDict()            # stores ID:Locations
        self.lost = OrderedDict()               # stores ID:Lost_count
        
        self.maxLost = maxLost                  # maximum number of frames object was not detected.
        
    def addObject(self, new_object_location):
        self.objects[self.nextObjectID] = new_object_location    # store new object location
        self.lost[self.nextObjectID] = 0                         # initialize frame_counts for when new object is undetected
        
        self.nextObjectID += 1
    
    def removeObject(self, objectID):                          # remove tracker data after object is lost
        del self.objects[objectID]
        del self.lost[objectID]
    
    @staticmethod
    def getLocation(bounding_box):
        xlt, ylt, xrb, yrb = bounding_box
        return (int((xlt + xrb) / 2.0), int((ylt + yrb) / 2.0))
    
    def update(self,  detections):
        
        if len(detections) == 0:   # if no object detected in the frame
            lost_ids = list(self.lost.keys())
            for objectID in lost_ids:
                self.lost[objectID] +=1
                if self.lost[objectID] > self.maxLost: self.removeObject(objectID)
            
            return self.objects
        
        new_object_locations = np.zeros((len(detections), 2), dtype="int")     # current object locations
        
        for (i, detection) in enumerate(detections): new_object_locations[i] = self.getLocation(detection)
            
        if len(self.objects)==0:
            for i in range(0, len(detections)): self.addObject(new_object_locations[i])
        else:
            objectIDs = list(self.objects.keys())
            previous_object_locations = np.array(list(self.objects.values()))
            
            D = distance.cdist(previous_object_locations, new_object_locations) # pairwise distance between previous and current
            
            row_idx = D.min(axis=1).argsort()   # (minimum distance of previous from current).sort_as_per_index
            
            cols_idx = D.argmin(axis=1)[row_idx]   # index of minimum distance of previous from current
            
            assignedRows, assignedCols = set(), set()
            
            for (row, col) in zip(row_idx, cols_idx):
                
                if row in assignedRows or col in assignedCols:
                    continue
                
                objectID = objectIDs[row]
                self.objects[objectID] = new_object_locations[col]
                self.lost[objectID] = 0
                
                assignedRows.add(row)
                assignedCols.add(col)
                
            unassignedRows = set(range(0, D.shape[0])).difference(assignedRows)
            unassignedCols = set(range(0, D.shape[1])).difference(assignedCols)
            
            
            if D.shape[0]>=D.shape[1]:
                for row in unassignedRows:
                    objectID = objectIDs[row]
                    self.lost[objectID] += 1
                    
                    if self.lost[objectID] > self.maxLost:
                        self.removeObject(objectID)
                        
            else:
                for col in unassignedCols:
                    self.addObject(new_object_locations[col])
            
        return self.objects
model_info = {"config_path":"hand_inference_graph/hand_label_map.pbtxt",
              "model_weights_path":"hand_inference_graph/frozen_inference_graph.pb",
              "object_names": {0: 'Hand1',1: 'Hand2',2: 'Hand3'},
              "confidence_threshold": 0.5,
              "threshold": 0.4
             }

net = cv.dnn.readNetFromTensorflow(model_info["model_weights_path"], model_info["config_path"])
np.random.seed(12345)

bbox_colors = {key: np.random.randint(0, 255, size=(3,)).tolist() for key in model_info['object_names'].keys()}

maxLost = 5   # maximum number of object losts counted when the object is being tracked
tracker = Tracker(maxLost = maxLost)

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 600)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 800)
im_width, im_height = (cap.get(3), cap.get(4))
n = 0
while(True):
    
    ok, image = cap.read()
    try:
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    except:
        print("Error converting to RGB")
    
    if not ok:
        print("Cannot read the video feed.")
        break

    if n % 5== 0:

        boxes, scores, classes = detector_utils.detect_objects1(image,
                                                    detection_graph, sess)

    detections_bbox = []     # bounding box for detections
    
    boxess, confidences, classIDs = [], [], []
    
    for i in range(2):
        classID = classes[i]
        confidence = scores[i]

        if confidence > model_info['confidence_threshold']:
            (left, right, top, bottom) = (boxes[i][1] * im_width, boxes[i][3] * im_width,
                                boxes[i][0] * im_height, boxes[i][2] * im_height)
            
            width = right - left + 1
            height = bottom - top + 1
            boxess.append([int(left), int(top), int(width), int(height)])
            confidences.append(float(confidence))
            classIDs.append(int(classID))
    
    indices = cv.dnn.NMSBoxes(boxess, confidences, model_info["confidence_threshold"], model_info["threshold"])
    if len(indices)>0:
        for i in indices.flatten():
            x, y, w, h = boxess[i][0], boxess[i][1], boxess[i][2], boxess[i][3]
            
            detections_bbox.append((x, y, x+w, y+h))
            
            clr = [int(c) for c in bbox_colors[i]]
            cv.rectangle(image, (x, y), (x+w, y+h), clr, 2)
            
            label = "{}:{:.4f}".format(model_info["object_names"][i], confidences[i])
            (label_width, label_height), baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            y_label = max(y, label_height)
            p1 = (int(left), int(top))
            p2 = (int(right), int(bottom))
            cv.rectangle(image, (x, y_label-label_height),
                                 (x+label_width, y_label+baseLine), (255, 255, 255), cv.FILLED)
            cv.putText(image, label, (x, y_label), cv.FONT_HERSHEY_SIMPLEX, 0.5, clr, 2)
    if n % 5 ==0:
        objects = tracker.update(detections_bbox)           # update tracker based on the newly detected objects
    
    for (objectID, centroid) in objects.items():
        text = "ID {}".format(objectID)
        cv.putText(image, text, (centroid[0] - 10, centroid[1] - 10), cv.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2)
        cv.circle(image, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    n  += 1
    cv.imshow('ASL',
            cv.cvtColor(image, cv.COLOR_RGB2BGR))
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyWindow("image")