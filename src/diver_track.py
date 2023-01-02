from typing import List, Tuple, Deque

import rospy
from std_msgs.msg import Header 

from collections import deque
from math import sqrt
import numpy as np
from statistics import mode

from threading import Thread, Lock, Event

from proteus.environment.agent import Diver
from darknet_ros_msgs.msg import BoundingBox
from deeplabcut_ros.srv import ProcessDiverPose
from body_pose_msgs.msg import BodyPartEstimate, HumanPoseEstimate

bp_ids = ['head', 'sternum', 'l_shoulder', 'l_elbow', 'l_wrist',
          'r_shoulder', 'r_elbow', 'r_wrist', 'waist', 'l_hip', 
          'l_knee', 'l_ankle', 'r_hip', 'r_knee', 'r_ankle']

# This function returns the IOU of two bounding box messages.
def calculate_iou(boxA: BoundingBox, boxB: BoundingBox) -> float:
    xA = max(boxA.xmin, boxB.xmin)
    yA = max(boxA.ymin, boxB.ymin)
    xB = min(boxA.xmax, boxB.xmax)
    yB = min(boxA.ymax, boxB.ymax)

    intersection_area = max(0., xB- xA + 1) * max(0., yB- yA + 1)

    boxA_area = (boxA.xmax - boxA.xmin + 1) * (boxA.ymax - boxA.ymin + 1)
    boxB_area = (boxB.xmax - boxB.xmin + 1) * (boxB.ymax - boxB.ymin + 1)

    iou = intersection_area / float(boxA_area + boxB_area - intersection_area)
    return iou

# Calculate the distance between the center of two pose messages.
def calculate_center_dist(poseA: HumanPoseEstimate, poseB: HumanPoseEstimate) -> float:
    poseA_center = calculate_pose_center(poseA)
    poseB_center = calculate_pose_center(poseB)

    if poseA_center is None or poseB_center is None:
        return None # HACK lol

    return dist(poseA_center, poseB_center)

def dist(a, b):
    return sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def bp_dist(a,b):
    return dist([a.abs_x, a.abs_y], [b.abs_x, b.abs_y])

# Checks if a point is within the bounds of a bounding box.
def point_within_box(point: List[float], box: BoundingBox) -> bool:
    x,y=point
    return ((x > box.xmin) and (x < box.xmax)) and ((y > box.ymin) and (y < box.ymax))

# # Calculate the center of a bounding box (in pixel-relative coordinates)
# def calculate_box_center(box: BoundingBox, dims: List[float]) -> List[float]:
#     midx = float(box.xmax - box.xmin)/dims[0]
#     midy = float(box.ymax - box.ymin)/dims[1]

#     return [midx, midy]

# Calculate the center of a pose message.
def calculate_pose_center(pose:HumanPoseEstimate, pcutoff:float= 0.5) -> List[float]:
    xs = []
    ys = []

    for bp in pose.body_parts:
        if bp.confidence > pcutoff:
            xs.append(bp.abs_x)
            ys.append(bp.abs_y)

    if len(xs) > 0:
        x_mean = sum(xs)/float(len(xs))
        y_mean = sum(ys)/float(len(ys))

        return [x_mean, y_mean]

    else:
        return None

def calculate_box_center(box:BoundingBox) -> List[float]:

    w = box.xmax - box.xmin
    h = box.ymax - box.ymin

    center_x = box.xmin + (w/2)
    center_y = box.ymin + (h/2)

    return [center_x, center_y]


def calculate_catwa_bbox(boxes:Deque, times:Deque, conf_thresh : float = 0.75) -> BoundingBox:
    # This algorithm is a confidence-aware time-weighted average.
    # Basically, we need to calculate weights for each bounding box based on 
    # the time the box was observed and the confidence of the bbox.
    coords = np.zeros((len(boxes), 5))
    weights = np.zeros((len(boxes)))

    base_time = times[-1]

    for k, box in enumerate(boxes):
        coords[k,:] = [box.xmin, box.ymin, box.xmax, box.ymax, box.probability]

        time_since = (times[k] - base_time).to_sec() # The further we are from the oldest time (more recent), the higher the value
        conf = box.probability
        conf_term = conf if conf >= conf_thresh else (conf/2) # If conf is below threshold, reduce its effect
        weights[k] = time_since * conf_term

    # Once the weights are calculated, it's a simple weighted average for each coordinate.

    mean = np.average(coords, 0, weights)

    # Now we construct a new BBOx with the average coords.
    ret = BoundingBox()
    ret.xmin = int(mean[0])
    ret.ymin = int(mean[1])
    ret.xmax = int(mean[2])
    ret.ymax = int(mean[3])
    ret.probability = float(mean[4])
    ret.Class = 'diver'

    return ret

def calculate_catwa_pose(poses:Deque, times:Deque, conf_thresh:float = 0.75) -> HumanPoseEstimate:
    # This algorithm is a confidence-aware time-weighted average.
    # It's similar to the bbox algorithm, but in this case we need
    # to consider weights for each body part individually, since they each have a confidence. 

    n_bps = len(poses[0].body_parts)

    coords = np.zeros((len(poses), n_bps, 5))
    weights = np.zeros((len(poses),n_bps, 5))

    base_time = times[-1]

    for k, pose in enumerate(poses):
        time_since = (times[k] - base_time).to_sec() # The further we are from the oldest time (more recent), the higher the value
        for j, bp in enumerate(pose.body_parts):
            coords[k, j,:] = [bp.abs_x, bp.abs_y, bp.rel_x, bp.rel_y, bp.confidence]
            conf = bp.confidence
            conf_term = conf if conf >= conf_thresh else (conf/2) # If conf is below threshold, reduce its effect
            weights[k,j, :] = [(time_since * conf_term)] * 5

    # Once the weights are calculated, it's a simple weighted average for each coordinate.
    mean = np.average(coords, 0, weights)

    # Now we construct a new HumanPoseEstimate message with the average pose.
    ret = HumanPoseEstimate()
    conf_mean = 0
    for k, bp in enumerate(mean):
        part = BodyPartEstimate()
        part.id = bp_ids[k]
        part.abs_x = int(bp[0])
        part.abs_y = int(bp[1])
        part.rel_x = float(bp[2])
        part.rel_x = float(bp[3])
        part.confidence = float(bp[4])
        conf_mean += bp[4]

        ret.body_parts.append(part)
        
    ret.confidence = float(conf_mean/n_bps)

    return ret


# The diver track object adds tracking information on top of a PROTEUS diver object.
class DiverTrack(Thread):
    _iou_threshold = 0.25
    _conf_threshold = 0.75
    _time_thresh = 1.5
    _bbox_area_ratio = 0.72
    _pose_ratio_config = {'bust':{'bps' : ['head','sternum'], 'target_ratio':0.65, 'axis': 'y', 'weight':0.1}, 
                    'trunk':{'bps' : ['sternum','waist'], 'target_ratio':0.89, 'axis': 'y', 'weight':0.5}, 
                    'wingpsan':{'bps' : ['l_shoulder','r_shoulder'], 'target_ratio':0.4, 'axis': 'x', 'weight':0.2}, 
                    'hips':{'bps' : ['l_hip','r_hip'], 'target_ratio':0.2, 'axis': 'x', 'weight':0.2}}

    def __init__(self, name: str, bbox: BoundingBox, cur_img_header: Header, pose_srv: str, img_dims: List[float], queue_size: int = 10) -> None:
        super().__init__()
        self.diver = Diver(name) # ID

        # Queues of recent BBox and Pose messages.
        self.recent_bboxes = deque(maxlen=queue_size)
        self.recent_img_headers = deque(maxlen=queue_size)
        self.recent_bbox_times = deque(maxlen=queue_size)
        self.bbox_lock = Lock()

        self.recent_poses = deque(maxlen=queue_size)
        self.recent_pose_times = deque(maxlen=queue_size)
        self.pose_lock = Lock()

        # Add initial bounding box
        self.recent_bboxes.append(bbox)
        self.recent_img_headers.append(cur_img_header)
        self.recent_bbox_times.append(rospy.Time.now())

        # Add the service that we can use to get poses.
        self.pose_srv_str = pose_srv
        self.pose_service = rospy.ServiceProxy(self.pose_srv_str, ProcessDiverPose, persistent=True)

        # Add variables for the filtered bounding box and pose.
        self.filtered_bbox = None
        self.fbox_lock = Lock()
        self.filtered_pose = None
        self.fpose_lock = Lock()

        # Time last seen and whether or not the diver is currently seen.
        self.time_lock = Lock()
        self.last_seen = None
        self.currently_seen = True

        # Dimmensions of source image and DRP information.
        self.dims = img_dims
        self.center_point = None
        self.pseudodistance = None
        self.drp_lock = Lock()

        self.shutdown = Event()

    def run(self) -> None:
        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and not self.shutdown.is_set():
            rospy.logdebug(f"track {self.diver.readable_id} updating pose.")
            self.update_pose() 
            rospy.logdebug(f"track {self.diver.readable_id} updating recency.")
            self.update_seen()
            rospy.logdebug(f"track {self.diver.readable_id} updating filtered pose.")
            self.update_filtered()
            rospy.logdebug(f"track {self.diver.readable_id} updating drp.")
            self.update_relative_position()
            rospy.logdebug(f"track {self.diver.readable_id} sleeping.")
            rate.sleep()

    def close_track(self) -> None:
        self.pose_service.close()
        self.shutdown.set()

    # Accessor for latest bbox.
    def get_latest_bbox(self) -> BoundingBox:
        if len(self.recent_bboxes) > 0:
            self.bbox_lock.acquire()
            ret = self.recent_bboxes[0]
            self.bbox_lock.release()
            return ret
        else:
            return None

    def get_latest_img_header(self) -> Header:
        if len(self.recent_img_headers) > 0:
            self.bbox_lock.acquire()
            ret = self.recent_img_headers[0]
            self.bbox_lock.release()
            return ret
        else:
            return None

    # Accessor for latest pose
    def get_latest_pose(self) -> HumanPoseEstimate:
        if len(self.recent_poses) > 0:
            self.pose_lock.acquire()
            ret = self.recent_poses[0]
            self.pose_lock.release()
            return ret
        else:
            return None

    def get_filtered_bbox(self) -> BoundingBox:
        self.fbox_lock.acquire()
        ret = self.filtered_bbox
        self.fbox_lock.release()
        return ret

    def get_filtered_pose(self) -> HumanPoseEstimate:
        self.fpose_lock.acquire()
        ret = self.filtered_pose
        self.fpose_lock.release()
        return ret

    # Accessor for drp data.
    def get_relative_position(self) -> Tuple[List[float], float]:
        self.drp_lock.acquire()
        drp = [self.center_point, self.pseudodistance]
        self.drp_lock.release()
        return drp

    def calculate_bbox_confidence(self) -> float:
        if len(self.recent_bboxes) > 1:
            ious = []
            self.bbox_lock.acquire()
            for k, bbox in enumerate(self.recent_bboxes):
                if k == 0:
                    continue
                else:
                    ious.append(calculate_iou(self.recent_bboxes[k-1], bbox))
            
            self.bbox_lock.release()
            avg_iou = sum(ious)/ float(len(ious))
        else:
            avg_iou = 0

        return avg_iou

    def calculate_pose_confidence(self) -> float:
        if len(self.recent_poses) > 1:
            dists = []
            self.pose_lock.acquire()
            for k, pose in enumerate(self.recent_poses):
                if k == 0:
                    continue
                else:
                    d = calculate_center_dist(self.recent_poses[k-1], pose)
                    if d is not None:
                        dists.append(d)
            
            self.pose_lock.release()

            if len(dists) > 0:
                avg_dist = sum(dists)/ float(len(dists))
                max_dist = dist([0,0], self.dims)
                conf = 1.0 - (float(avg_dist) / float(max_dist))
            else:
                conf = 0.0
        else:
            conf = 0.0

        return conf

    # Calculate the confidence in the track based on the matches between the track elements.
    def calculate_track_confidence(self) -> float:
        c_bbox = self.calculate_bbox_confidence()
        c_pose = self.calculate_pose_confidence()
        return (c_bbox * c_pose)

    # Calculate DRP data
    def update_relative_position(self) -> None:
        if self.currently_seen:
            self.drp_lock.acquire()
            self.center_point = self.calculate_center_point()
            self.pseudodistance = self.calculate_pseudodistance()
            self.drp_lock.release()

    # Calculate the center point information
    def calculate_center_point(self) -> List[int]:

        pose = self.get_filtered_pose()
        box = self.get_filtered_bbox()
        p_center, b_center = [None, None]
        if pose is not None:
            p_center = calculate_pose_center(pose)
        if box is not None:
            b_center = calculate_box_center(box)

        # At this point, we will have calculated pose/bbox centers, if they're available.
        if p_center is not None and b_center is not None:
            rospy.logdebug(f"Using avg center (box:{b_center} pose:{p_center})")
            center_x = (b_center[0] + p_center[0])/2
            center_y = (b_center[1] + p_center[1])/2
            center = [center_x, center_y]
            center = [int(i) for i in center]
        elif b_center is not None:
            rospy.logdebug(f"Using box center (box:{b_center})")
            center = [int(i) for i in b_center]
        elif p_center is not None:
            rospy.logdebug(f"Using pose center (pose:{p_center})")
            center = [int(i) for i in p_center]
        else:
            return None

        return center

    def find_good_pose_pairs(self) -> bool:
        # We want to consider 3 things:
        # - Presence of required body point pairs in box, at high confidence
        # - The average confidence of the required point pairs
        pose = self.get_filtered_pose()
        box = self.get_filtered_bbox()
        good_pairs = {}

        if pose is not None and box is not None:
            bps = {'head': None, 'sternum': None, 'waist': None, 'l_shoulder': None, 'r_shoulder': None, 'l_hip': None, 'r_hip': None}
            for bp in pose.body_parts:
                if bp.id in bps.keys():
                    bps[bp.id] = bp

            for key, item in DiverTrack._pose_ratio_config.items():
                a, b = item['bps']
                bp_a = bps[a]
                bp_b = bps[b]

                conf_term = (bp_a.confidence >= DiverTrack._conf_threshold) and (bp_b.confidence >= DiverTrack._conf_threshold)
                presence_term = (point_within_box([bp_a.abs_x, bp_a.abs_y],box)) and (point_within_box([bp_b.abs_x, bp_b.abs_y], box))

                if conf_term and presence_term:
                    good_pairs[key] = [bp_a, bp_b]

        return good_pairs

    # Calculat the pseudodistances information
    def calculate_pseudodistance(self) -> float:
        now = rospy.Time.now()

        pose = self.get_filtered_pose()
        box = self.get_filtered_bbox()
        iw,ih = self.dims

        good_pairs =  self.find_good_pose_pairs()

        # We use pose if we have more than 2 good pairs to look at. Anything less than that, we don't bother.
        if len(good_pairs.keys()) >= 2:
            ratios = {}

            # Calculate ratios for each good pair.
            for key, bps in good_pairs.items():
                config = DiverTrack._pose_ratio_config[key]
                if config['axis'] == 'x':
                    a,b = bps
                    pair_width = bp_dist(a,b)
                    ratios[key] = (iw/pair_width) * (config['target_ratio']/4.0)
                    
                elif config['axis'] == 'y':
                    a,b = bps
                    pair_height = bp_dist(a,b)
                    ratios[key] = (ih/pair_height) * (config['target_ratio']/4.0)
            
            # Take a look at the variance of ratios
            rlist = np.array([val for _, val in ratios.items()])
            rlist = rlist[abs(rlist - np.mean(rlist)) < 1 * np.std(rlist)] 
            variance = np.var(rlist)

            # If the variance is low, we just do the simple average of the ratios.
            # The threshold for the variance is based on the range of PD (0.0 - 4.0)
            if variance <= 2.0:
                ratio = np.mean(rlist)

            # if if the variance is bad, we fall back to bbox estimate.
            else:
                box_width = abs(box.xmax - box.xmin)
                box_height = abs(box.ymax - box.ymin)
                box_area = float(box_width * box_height)
                image_area = float(iw * ih)

                ratio = (image_area/box_area) * (DiverTrack._bbox_area_ratio/4.00)

            rospy.logdebug(f"Pose pd -- Avg:{ratio} Var:{variance} Ratios:{ratios}")

        # If there's not enough good pose to work with, we just use bbox estimate.
        elif box is not None:
            box_width = abs(box.xmax - box.xmin)
            box_height = abs(box.ymax - box.ymin)
            box_area = float(box_width * box_height)
            image_area = float(iw * ih)

            ratio = (image_area/box_area) * (DiverTrack._bbox_area_ratio/4.00)
            rospy.logdebug(f"Box pd -- Area_ratio:{(image_area/box_area)}, created_target:{(DiverTrack._bbox_area_ratio/4.0)}, pd:{ratio}")

        # And lastly, if there was not enough good pose or bbox, we just return None, because we have no clue what's going on.
        else:
            ratio = None
  
        return ratio
        

    def update_pose(self):
        if ((rospy.Time.now() - self.recent_bbox_times[0]).to_sec() < DiverTrack._time_thresh):
            rospy.logdebug(f"Requesting pose for diver {self.diver.readable_id}")

            self.bbox_lock.acquire()
            box = self.recent_bboxes[0]
            img_header = self.recent_img_headers[0]
            self.bbox_lock.release()

            try:
                resp = self.pose_service(img_header, box)
            except rospy.ServiceException:
                rospy.logwarn(f"Service exception, pose service connection may have failed for track {self.diver.readable_id}")
                rospy.loginfo("Attempting reconnection of pose service, falling back on a non-persistent service")
                self.pose_service = rospy.ServiceProxy(self.pose_srv_str, ProcessDiverPose)
                return

            pose = HumanPoseEstimate()
            pose.body_parts = resp.pose.body_parts
            pose.confidence = resp.pose.confidence

            self.pose_lock.acquire()
            self.recent_pose_times.append(rospy.Time.now())
            self.recent_poses.append(pose)
            self.pose_lock.release()

    # Update recency information
    def update_seen(self) -> None:
        if len(self.recent_pose_times) > 0:
            self.last_seen = max(self.recent_bbox_times[0], self.recent_pose_times[0])
        else:
            self.last_seen = self.recent_bbox_times[0]

        self.currently_seen = ((rospy.Time.now() - self.last_seen).to_sec() < DiverTrack._time_thresh)

    # Calculate the filtered bbox and poses across recent observations using a time-weighted average.
    def update_filtered(self) -> None:
        if self.currently_seen:
            if len(self.recent_bboxes) > 1:
                self.bbox_lock.acquire()
                self.fbox_lock.acquire()
                self.filtered_bbox = calculate_catwa_bbox(self.recent_bboxes, self.recent_bbox_times)
                self.fbox_lock.release()
                self.bbox_lock.release()
            if len(self.recent_poses) > 1:
                self.pose_lock.acquire()
                self.fpose_lock.acquire()
                self.filtered_pose = calculate_catwa_pose(self.recent_poses, self.recent_pose_times)
                self.fpose_lock.release()
                self.pose_lock.release()    

    # Attempt to associate a bounding box with this track.
    def associate_bbox(self, candidates: List[BoundingBox], cur_img_header: Header, method: str = "max_iou") -> Tuple[bool, List[BoundingBox]]:
        # If there are no previous bounding boxes, we need to consider the most recent pose.
        if method.lower() == "max_iou":
            ious = []
            self.bbox_lock.acquire()
            for k, past in enumerate(self.recent_bboxes):
                ious.append([])
                for candidate in candidates:
                    ious[k].append(calculate_iou(candidate, past))

            self.bbox_lock.release()

            maxs = []
            for iou_list in ious:
                max_iou = max(iou_list)
                if max_iou > DiverTrack._iou_threshold:
                    maxs.append(iou_list.index(max_iou))
                else:
                    maxs.append(None)

            selected_idx = mode(maxs)
            if selected_idx is not None:
                selected_bbox = candidates.pop(selected_idx)
                self.recent_bboxes.append(selected_bbox)
                self.recent_img_headers.append(cur_img_header)
                self.recent_bbox_times.append(rospy.Time.now())
                ret = True

            else: # If the most common element in the list is None, we didn't find a good match.
                ret = False

            return [ret, candidates]

        else:
            raise NotImplementedError(f"Association method {method} not implemented in function associated_bbox.")


