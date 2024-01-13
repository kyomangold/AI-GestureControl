import cv2
import numpy as np

LINES_HAND = [[0,1],[1,2],[2,3],[3,4], 
            [0,5],[5,6],[6,7],[7,8],
            [5,9],[9,10],[10,11],[11,12],
            [9,13],[13,14],[14,15],[15,16],
            [13,17],[17,18],[18,19],[19,20],[0,17]]

class HandTrackerRenderer:
    def __init__(self, 
                tracker,
                output=None):

        self.tracker = tracker

        # Rendering flags
        if self.tracker.use_lm:
            self.show_landmarks = True
        else:
            None

        if output is None:
            self.output = None
        else:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            self.output = cv2.VideoWriter(output,fourcc,self.tracker.video_fps,(self.tracker.img_w, self.tracker.img_h)) 

    def draw_hand(self, hand):
            dynamic_size = hand.rect_w_a / 400 # adapt size of landmarks to size of hand
            if hand.lm_score > self.tracker.lm_score_thresh:
                if self.show_landmarks:
                    lines = [np.array([hand.landmarks[point] for point in line]).astype(np.int32) for line in LINES_HAND]
                    color = (255, 0, 0)
                    cv2.polylines(self.frame, lines, False, color, int(1+dynamic_size*3), cv2.LINE_AA)
                    radius = int(1+dynamic_size*5)
                    if self.tracker.use_gesture:
                        # color depending on finger state (1=open, 0=close, -1=unknown)
                        color = { 1: (0,255,0), 0: (0,0,255), -1:(0,255,255)}
                        cv2.circle(self.frame, (hand.landmarks[0][0], hand.landmarks[0][1]), radius, color[-1], -1)
                        # draw thumb landmarks
                        for i in range(1,5):
                            cv2.circle(self.frame, (hand.landmarks[i][0], hand.landmarks[i][1]), radius, color[hand.thumb_state], -1)
                        # draw index finger landmarks
                        for i in range(5,9):
                            cv2.circle(self.frame, (hand.landmarks[i][0], hand.landmarks[i][1]), radius, color[hand.index_state], -1)
                        # draw middle finger landmarks
                        for i in range(9,13):
                            cv2.circle(self.frame, (hand.landmarks[i][0], hand.landmarks[i][1]), radius, color[hand.middle_state], -1)
                        # draw ring finger landmarks
                        for i in range(13,17):
                            cv2.circle(self.frame, (hand.landmarks[i][0], hand.landmarks[i][1]), radius, color[hand.ring_state], -1)
                        # draw little finger landmarks
                        for i in range(17,21):
                            cv2.circle(self.frame, (hand.landmarks[i][0], hand.landmarks[i][1]), radius, color[hand.little_state], -1)
                    else:
                        for x,y in hand.landmarks[:,:2]:
                            cv2.circle(self.frame, (int(x), int(y)), radius, color, -1)
        
    def draw(self, frame, hands):
        self.frame = frame
        for hand in hands:
            self.draw_hand(hand)
        return self.frame

    def exit(self):
        if self.output:
            self.output.release()
        cv2.destroyAllWindows()

    def waitKey(self, delay=1):
        cv2.imshow("Hand tracking", self.frame)
        if self.output:
            self.output.write(self.frame)
        key = cv2.waitKey(delay) 
        if key == 32: # Pause on space bar
            key = cv2.waitKey(0)
        elif key == ord('l') and self.tracker.use_lm:
            self.show_landmarks = not self.show_landmarks
        return key
