import sys
import datetime
from time import monotonic

ALL_POSES = ["ONE","TWO","THREE","FOUR","FIVE","FIST","PEACE","OK"]

# Default config parameters
DEFAULT_CONFIG = {
    'pose_params': 
    {
        "callback": "_DEFAULT_",
        "hand": "any",
        "trigger": "enter", 
        "first_trigger_delay": 0.3, 
        "next_trigger_delay": 0.3, 
        "max_missing_frames": 3,
    },

    'tracker': 
    { 
        'args': 
        {
            'pd_score_thresh': 0.6,
            'pd_nms_thresh': 0.3,
            'lm_score_thresh': 0.5, 
            'solo': True, # track only a single hand
            'internal_fps': 30,
            'internal_frame_height': 640,
            'use_gesture': True
        },
    },

    'renderer':
    {   
        'enable': False,

        'args':
        {
            'output': None,
        }
    }
}

class Event:
    def __init__(self, category, hand, pose_action, trigger):
        self.category = category
        self.hand = hand
        if hand:
            #self.handedness = hand.label
            self.pose = hand.gesture
        else:
            #self.handedness = None
            self.pose = None
        self.name = pose_action["name"]
        self.callback = pose_action["callback"]
        self.trigger = trigger
        self.time = datetime.datetime.now()
   
class PoseEvent(Event):
    def __init__(self, hand, pose_action, trigger):
        super().__init__("Pose", hand, pose_action, trigger = trigger)

class EventHist:
    def __init__(self, triggered=False, first_triggered=False, time=0, frame_nb=0):
        self.triggered = triggered
        self.first_triggered = first_triggered
        self.time = time
        self.frame_nb = frame_nb

def merge_dicts(d1, d2):
    #Merge 2 dictionaries. The 2nd dictionary's values overwrites those from the first
    return {**d1, **d2}

def config_handler(c1, c2):
    #Merge two configs c1 and c2 (where c1 is the default config and c2 the user defined config).
    merged_config = {}
    for k1,v1 in c1.items():
        if k1 in c2:
            if isinstance(v1, dict):
                merged_config[k1] = config_handler(v1, c2[k1])
            else:
                merged_config[k1] = c2[k1]
        else:
            merged_config[k1] = v1
    for k2,v2 in c2.items():
        if k2 not in c1:
            merged_config[k2] = v2
    return merged_config

class HandController:
    def __init__(self, config={}):
        self.config = config_handler(DEFAULT_CONFIG, config)

        # HandController runs callback functions defined in the calling app
        self.caller_globals = sys._getframe(1).f_globals

        # Parse pose configurations
        self.parse_poses()

        # Store previous poses 
        self.poses_hist = [EventHist() for i in range(len(self.pose_actions))]

        # Load HandTracker
        from hand_tracker_edge import HandTracker
       
        # Initialize tracker
        self.tracker = HandTracker(**self.config['tracker']['args'])

        # Activate renderer to show live video preview with hand skeleton
        self.use_renderer = self.config['renderer']['enable']
        if self.use_renderer:
            from hand_tracker_renderer import HandTrackerRenderer
            self.renderer = HandTrackerRenderer(self.tracker, **self.config['renderer']['args'])

        self.frame_nb = 0
        

    def parse_poses(self):
        mandatory_keys = ['name', 'pose']
        optional_keys = self.config['pose_params'].keys()
        self.pose_actions = []
        if 'pose_actions' in self.config:
            for pa in self.config['pose_actions']:
                pose = pa['pose']
                if pose == 'ALL':
                    pa['pose'] = ALL_POSES
                else:
                    pa['pose'] = [pose]
                optional_args = {k:pa.get(k, self.config['pose_params'][k]) for k in optional_keys}
                mandatory_args = { k:pa[k] for k in mandatory_keys}
                all_args = merge_dicts(mandatory_args, optional_args)
                self.pose_actions.append(all_args)
            
    def generate_events(self, hands):

        events = []

        # in solo mode: either hands=[] or hands=[hand]
        hand = hands[0] if hands else None

        for i, pa in enumerate(self.pose_actions):
            hist = self.poses_hist[i]
            trigger = pa['trigger']
            if hand and hand.gesture and \
                (hand.label == pa['hand'] or pa['hand'] == 'any') and \
                hand.gesture in pa['pose']:
                if trigger == "continuous":
                    events.append(PoseEvent(hand, pa, "continuous"))
                else: # trigger in ["enter", "enter_leave", "periodic"]:
                    if not hist.triggered:
                        if hist.time != 0 and (self.frame_nb - hist.frame_nb <= pa['max_missing_frames']):
                            if  hist.time and \
                                ((hist.first_triggered and self.now - hist.time > pa['next_trigger_delay']) or \
                                    (not hist.first_triggered and self.now - hist.time > pa['first_trigger_delay'])):
                                
                                if trigger == "enter" or trigger == "enter_leave":
                                    hist.triggered = True
                                    events.append(PoseEvent(hand, pa, "enter"))
                                else: # "periodic"
                                    hist.time = self.now
                                    hist.first_triggered = True
                                    events.append(PoseEvent(hand, pa, "periodic"))
                                
                        else:
                            hist.time = self.now
                            hist.first_triggered = False
                    else:
                        if self.frame_nb - hist.frame_nb > pa['max_missing_frames']:
                            hist.time = self.now
                            hist.triggered = False
                            hist.first_triggered = False
                            if trigger == "enter_leave":
                                events.append(PoseEvent(hand, pa, "leave"))
                hist.frame_nb = self.frame_nb

            else:
                if hist.triggered and self.frame_nb - hist.frame_nb > pa['max_missing_frames']:
                    hist.time = self.now
                    hist.triggered = False
                    hist.first_triggered = False 
                    if trigger == "enter_leave":
                        events.append(PoseEvent(hand, pa, "leave")) 
        return events    

    def process_events(self, events):
        for e in events:
            self.caller_globals[e.callback](e)

    def loop(self):
        while True:
            self.now = monotonic()
            frame, hands, _ = self.tracker.next_frame()
            if frame is None: break
            self.frame_nb += 1
            events = self.generate_events(hands)
            self.process_events(events)

            if self.use_renderer:
                frame = self.renderer.draw(frame, hands)
                key = self.renderer.waitKey(delay=1)
                if key == 27 or key == ord('q'):
                    break
        self.renderer.exit()
        self.tracker.exit()
            


