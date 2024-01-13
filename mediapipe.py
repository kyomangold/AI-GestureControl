import cv2
import numpy as np
from collections import namedtuple
from math import sin, cos, gcd

class HandRegion:
    """
        Attributes:
        pd_score : detection score
        pd_box : detection box [x, y, w, h], normalized [0,1] in the squared image
        pd_kps : detection keypoints coordinates [x, y], normalized [0,1] in the squared image
        rotation : rotation angle of rotated bounding rectangle with y-axis in radian
        norm_landmarks : 3D landmarks coordinates in the rotated bounding rectangle, normalized [0,1]
        world_landmarks : 3D landmark coordinates in meter
        handedness: float between 0. and 1., > 0.5 for right hand, < 0.5 for left hand,
        gesture: (optional, set in recognize_gesture() when use_gesture==True) string corresponding to recognized gesture ("ONE","TWO","THREE","FOUR","FIVE","FIST","OK","PEACE") 
                or None if no gesture has been recognized
        """
    def __init__(self, pd_score=None, pd_box=None, pd_kps=None):
        self.pd_score = pd_score # Palm detection score 
        self.pd_box = pd_box # Palm detection box [x, y, w, h] normalized
        self.pd_kps = pd_kps # Palm detection keypoints

    def get_rotated_world_landmarks(self):
        world_landmarks_rotated = self.world_landmarks.copy()
        sin_rot = sin(self.rotation)
        cos_rot = cos(self.rotation)
        rot_m = np.array([[cos_rot, sin_rot], [-sin_rot, cos_rot]])
        world_landmarks_rotated[:,:2] = np.dot(world_landmarks_rotated[:,:2], rot_m)
        return world_landmarks_rotated

    def print(self):
        attrs = vars(self)
        print('\n'.join("%s: %s" % item for item in attrs.items()))

class HandednessAverage:
    #More robustness by using average handeness

    def __init__(self):
        self._total_handedness = 0
        self._nb = 0
    def update(self, new_handedness):
        self._total_handedness += new_handedness
        self._nb += 1
        return self._total_handedness / self._nb
    def reset(self):
        self._total_handedness = self._nb = 0


SSDAnchorOptions = namedtuple('SSDAnchorOptions',[
        'num_layers',
        'min_scale',
        'max_scale',
        'input_size_height',
        'input_size_width',
        'anchor_offset_x',
        'anchor_offset_y',
        'strides',
        'aspect_ratios',
        'reduce_boxes_in_lowest_layer',
        'interpolated_scale_aspect_ratio',
        'fixed_anchor_size'])

# Starting from opencv 4.5.4, cv2.dnn.NMSBoxes output format changed
import re
cv2_version = cv2.__version__.split('.')
v0 = int(cv2_version[0])
v1 = int(cv2_version[1])
v2 = int(re.sub(r'\D+', '', cv2_version[2]))
if  v0 > 4 or (v0 == 4 and (v1 > 5 or (v1 == 5 and v2 >= 4))):
    def non_max_suppression(regions, nms_thresh):
        # cv2.dnn.NMSBoxes(boxes, scores, 0, nms_thresh) needs:
        # boxes = [ [x, y, w, h], ...] with x, y, w, h of type int
        # Currently, x, y, w, h are float between 0 and 1, so we arbitrarily multiply by 1000 and cast to int
        # boxes = [r.box for r in regions]
        boxes = [ [int(x*1000) for x in r.pd_box] for r in regions]        
        scores = [r.pd_score for r in regions]
        indices = cv2.dnn.NMSBoxes(boxes, scores, 0, nms_thresh) # Not using top_k=2 here because it does not give expected result. Bug ?
        return [regions[i] for i in indices]
else:
    def non_max_suppression(regions, nms_thresh):
        # cv2.dnn.NMSBoxes(boxes, scores, 0, nms_thresh) needs:
        # boxes = [ [x, y, w, h], ...] with x, y, w, h of type int
        # Currently, x, y, w, h are float between 0 and 1, so we arbitrarily multiply by 1000 and cast to int
        # boxes = [r.box for r in regions]
        boxes = [ [int(x*1000) for x in r.pd_box] for r in regions]        
        scores = [r.pd_score for r in regions]
        indices = cv2.dnn.NMSBoxes(boxes, scores, 0, nms_thresh) # Not using top_k=2 here because it does not give expected result. Bug ?
        return [regions[i[0]] for i in indices]

def rotated_rect_to_points(cx, cy, w, h, rotation):
    b = cos(rotation) * 0.5
    a = sin(rotation) * 0.5
    points = []
    p0x = cx - a*h - b*w
    p0y = cy + b*h - a*w
    p1x = cx + a*h - b*w
    p1y = cy - b*h - a*w
    p2x = int(2*cx - p0x)
    p2y = int(2*cy - p0y)
    p3x = int(2*cx - p1x)
    p3y = int(2*cy - p1y)
    p0x, p0y, p1x, p1y = int(p0x), int(p0y), int(p1x), int(p1y)
    return [[p0x,p0y], [p1x,p1y], [p2x,p2y], [p3x,p3y]]

def distance(a, b):
    """
    a, b: 2 points (in 2D or 3D)
    """
    return np.linalg.norm(a-b)

def angle(a, b, c):
    # https://stackoverflow.com/questions/35176451/python-code-to-calculate-angle-between-three-point-using-their-3d-coordinates
    # a, b and c : points as np.array([x, y, z]) 
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)

def find_isp_scale_params(size, resolution, is_height=True):
    """
    Find closest valid size close to 'size' and and the corresponding parameters to setIspScale()
    This function is useful to work around a bug in depthai where ImageManip is scrambling images that have an invalid size
    resolution: sensor resolution (width, height)
    is_height : boolean that indicates if the value 'size' represents the height or the width of the image
    Returns: valid size, (numerator, denominator)
    """
    # We want size >= 288 (first compatible size > lm_input_size)
    if size < 288:
        size = 288

    width, height = resolution

    # We are looking for the list on integers that are divisible by 16 and
    # that can be written like n/d where n <= 16 and d <= 63
    if is_height:
        reference = height 
        other = width
    else:
        reference = width 
        other = height
    size_candidates = {}
    for s in range(288,reference,16):
        f = gcd(reference, s)
        n = s//f
        d = reference//f
        if n <= 16 and d <= 63 and int(round(other * n / d) % 2 == 0):
            size_candidates[s] = (n, d)
            
    # What is the candidate size closer to 'size' ?
    min_dist = -1
    for s in size_candidates:
        dist = abs(size - s)
        if min_dist == -1:
            min_dist = dist
            candidate = s
        else:
            if dist > min_dist: break
            candidate = s
            min_dist = dist
    return candidate, size_candidates[candidate]

def recognize_gesture(hand):           
    # Finger states
    # state: -1=unknown, 0=close, 1=open
    d_3_5 = distance(hand.norm_landmarks[3], hand.norm_landmarks[5])
    d_2_3 = distance(hand.norm_landmarks[2], hand.norm_landmarks[3])
    angle0 = angle(hand.norm_landmarks[0], hand.norm_landmarks[1], hand.norm_landmarks[2])
    angle1 = angle(hand.norm_landmarks[1], hand.norm_landmarks[2], hand.norm_landmarks[3])
    angle2 = angle(hand.norm_landmarks[2], hand.norm_landmarks[3], hand.norm_landmarks[4])
    hand.thumb_angle = angle0+angle1+angle2
    if angle0+angle1+angle2 > 460 and d_3_5 / d_2_3 > 1.2: 
        hand.thumb_state = 1
    else:
        hand.thumb_state = 0

    if hand.norm_landmarks[8][1] < hand.norm_landmarks[7][1] < hand.norm_landmarks[6][1]:
        hand.index_state = 1
    elif hand.norm_landmarks[6][1] < hand.norm_landmarks[8][1]:
        hand.index_state = 0
    else:
        hand.index_state = -1

    if hand.norm_landmarks[12][1] < hand.norm_landmarks[11][1] < hand.norm_landmarks[10][1]:
        hand.middle_state = 1
    elif hand.norm_landmarks[10][1] < hand.norm_landmarks[12][1]:
        hand.middle_state = 0
    else:
        hand.middle_state = -1

    if hand.norm_landmarks[16][1] < hand.norm_landmarks[15][1] < hand.norm_landmarks[14][1]:
        hand.ring_state = 1
    elif hand.norm_landmarks[14][1] < hand.norm_landmarks[16][1]:
        hand.ring_state = 0
    else:
        hand.ring_state = -1

    if hand.norm_landmarks[20][1] < hand.norm_landmarks[19][1] < hand.norm_landmarks[18][1]:
        hand.little_state = 1
    elif hand.norm_landmarks[18][1] < hand.norm_landmarks[20][1]:
        hand.little_state = 0
    else:
        hand.little_state = -1

    # Gestures 
    if hand.thumb_state == 1 and hand.index_state == 1 and hand.middle_state == 1 and hand.ring_state == 1 and hand.little_state == 1:
        hand.gesture = "FIVE"
    elif hand.thumb_state == 0 and hand.index_state == 0 and hand.middle_state == 0 and hand.ring_state == 0 and hand.little_state == 0:
        hand.gesture = "FIST"
    elif hand.thumb_state == 1 and hand.index_state == 0 and hand.middle_state == 0 and hand.ring_state == 0 and hand.little_state == 0:
        hand.gesture = "OK" 
    elif hand.thumb_state == 0 and hand.index_state == 1 and hand.middle_state == 1 and hand.ring_state == 0 and hand.little_state == 0:
        hand.gesture = "PEACE"
    elif hand.thumb_state == 0 and hand.index_state == 1 and hand.middle_state == 0 and hand.ring_state == 0 and hand.little_state == 0:
        hand.gesture = "ONE"
    elif hand.thumb_state == 1 and hand.index_state == 1 and hand.middle_state == 0 and hand.ring_state == 0 and hand.little_state == 0:
        hand.gesture = "TWO"
    elif hand.thumb_state == 1 and hand.index_state == 1 and hand.middle_state == 1 and hand.ring_state == 0 and hand.little_state == 0:
        hand.gesture = "THREE"
    elif hand.thumb_state == 0 and hand.index_state == 1 and hand.middle_state == 1 and hand.ring_state == 1 and hand.little_state == 1:
        hand.gesture = "FOUR"
    else:
        hand.gesture = None
       