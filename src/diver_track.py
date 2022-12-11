from typing import List, Tuple, Deque

import rospy
from collections import deque
from math import sqrt
import numpy as np
from statistics import mode

from threading import Thread, Lock, Event

from proteus.environment.agent import Diver
from darknet_ros_msgs.msg import BoundingBox
from deeplabcut_ros.srv import ProcessDiverPose
from body_pose_msgs.msg import BodyPartEstimate, HumanPoseEstimate

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

    return dist(poseA_center, poseB_center)

def dist(a, b):
    return sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

# Checks if a point is within the bounds of a bounding box.
def point_within_box(point: List[float], box: BoundingBox) -> bool:
    x,y=point
    return ((x > box.xmin) and (x < box.xmax)) and ((y > box.ymin) and (y < box.ymax))

# Calculate the center of a bounding box (in pixel-relative coordinates)
def calculate_box_center(box: BoundingBox, dims: List[float]) -> List[float]:
    midx = float(box.xmax - box.xmin)/dims[0]
    midy = float(box.ymax - box.ymin)/dims[1]

    return [midx, midy]

# Calculate the center of a pose message.
def calculate_pose_center(pose:HumanPoseEstimate) -> List[float]:
    xs = []
    ys = []

    for bp in pose.body_parts:
        xs.append(bp.rel_x)
        ys.append(bp.rel_y)

    x_mean = sum(xs)/float(len(xs))
    y_mean = sum(ys)/float(len(ys))

    return [x_mean, y_mean]


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
    for bp in mean:
        part = BodyPartEstimate()
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
    _dist_thereshold = 0.1
    _parts_threshold = 10
    _time_thresh = 3

    def __init__(self, name: str, bbox: BoundingBox, pose_srv: str, queue_size: int = 10) -> None:
        super().__init__()
        self.diver = Diver(name) # ID

        # Queues of recent BBox and Pose messages.
        self.recent_bboxes = deque(maxlen=queue_size)
        self.recent_bbox_times = deque(maxlen=queue_size)
        self.bbox_lock = Lock()

        self.recent_poses = deque(maxlen=queue_size)
        self.recent_pose_times = deque(maxlen=queue_size)
        self.pose_lock = Lock()

        # Add initial bounding box
        self.recent_bboxes.append(bbox)
        self.recent_bbox_times.append(rospy.Time.now())

        # Add the service that we can use to get poses.
        self.pose_service = rospy.ServiceProxy(pose_srv, ProcessDiverPose)

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
                    dists.append(calculate_center_dist(self.recent_poses[k-1], pose))
            
            self.pose_lock.release()
            avg_dist = sum(dists)/ float(len(dists))
        else:
            avg_dist = 0

        return avg_dist

    # Calculate the confidence in the track based on the matches between the track elements.
    def calculate_track_confidence(self) -> float:
        avg_iou = self.calculate_bbox_confidence()
        avg_dist = self.calculate_pose_confidence()
        return (avg_iou * (1 - avg_dist))

    # Calculate DRP data
    def update_relative_position(self) -> None:
        if self.currently_seen:
            self.drp_lock.acquire()
            self.center_point = self.calculate_center_point()
            self.pseudodistance = self.calculate_pseudodistance()
            self.drp_lock.release()

    # Calculate the center point information
    def calculate_center_point(self) -> List[float]:
        pass

    # Calculat the pseudodistances information
    def calculate_pseudodistance(self) -> float:
        pass

    def update_pose(self):
        if ((rospy.Time.now() - self.recent_bbox_times[0]).to_sec() < DiverTrack._time_thresh):
            rospy.logdebug(f"Requesting pose for diver {self.diver.readable_id}")

            self.bbox_lock.acquire()
            box = self.recent_bboxes[0]
            self.bbox_lock.release()

            resp = self.pose_service(box)

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
    def associate_bbox(self, candidates: List[BoundingBox], method: str = "max_iou") -> Tuple[bool, List[BoundingBox]]:
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
                self.recent_bbox_times.append(rospy.Time.now())
                ret = True

            else: # If the most common element in the list is None, we didn't find a good match.
                ret = False

            return [ret, candidates]

        else:
            raise NotImplementedError(f"Association method {method} not implemented in function associated_bbox.")


