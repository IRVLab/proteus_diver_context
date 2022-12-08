from typing import List

from rospy import Time
from collections import deque
from math import sqrt
from statistics import mode
from proteus.environment.agent import Diver
from darknet_ros_msgs.msg import BoundingBox
from body_pose_msgs.msg import HumanPoseEstimate

def calculate_iou(boxA: BoundingBox, boxB: BoundingBox) -> float:
    xA = max(boxA.xmax, boxB.xmax)
    yA = max(boxA.ymax, boxB.ymax)
    xB = min(boxA.xmin, boxB.xmin)
    yB = min(boxA.ymin, boxB.ymin)

    intersection_area = max(0, xB- xA + 1) * max(0, yB- yA + 1)

    boxA_area = (boxA.xmax - boxA.xmin + 1) * (boxA.ymax - boxA.ymin + 1)
    boxB_area = (boxB.xmax - boxB.xmin + 1) * (boxB.ymax - boxB.ymin + 1)

    iou = intersection_area / float(boxA_area + boxB_area - intersection_area)

    return iou

def calculate_center_dist(poseA: HumanPoseEstimate, poseB: HumanPoseEstimate) -> float:
    poseA_center = calculate_pose_center(poseA)
    poseB_center = calculate_pose_center(poseB)

    return sqrt((poseA_center[0] - poseB_center[0])**2 + (poseA_center[1] - poseB_center[1])**2)

def calculate_pose_center(pose:HumanPoseEstimate) -> List[float, float]:
    xs = []
    ys = []

    for bp in pose.body_parts:
        xs.append(bp.rel_x)
        ys.append(bp.rel_y)

    x_mean = sum(xs)/float(len(xs))
    y_mean = sum(ys)/float(len(ys))

    return [x_mean, y_mean]



class DiverTrack(object):
    _iou_threshold = 0.25
    _dist_thereshold = 10
    _time_thresh = 1

    def __init__(self, name: str, bbox: BoundingBox = None, pose: HumanPoseEstimate = None, queue_size: int = 10) -> None:
        self.diver = Diver(name)
        self.recent_bboxes = deque(queue_size)
        self.last_bbox_time = None
        self.recent_poses = deque(queue_size)
        self.last_pose_time = None

        self.last_seen = None
        self.currently_seen = True

        self.center_point = None
        self.pseudodistance = None

        if bbox:
            self.recent_bboxes.append(bbox)
            self.last_bbox_time = Time.now()
    
        if pose:
            self.recent_poses.append(bbox)
            self.last_pose_time = Time.now()


    def calculate_track_confidence(self) -> List[float, float, float]:
        ious = []
        for k, bbox in enumerate(self.recent_bboxes):
            if k == 0:
                continue
            else:
                ious.append(calculate_iou(self.recent_bboxes[k-1], bbox))
        
        avg_iou = sum(ious)/ float(len(ious))

        dists = []
        for k, pose in enumerate(self.recent_poses):
            if k == 0:
                continue
            else:
                ious.append(calculate_iou(self.recent_poses[k-1], pose))
        
        avg_dist = sum(dists)/ float(len(dists))

        return [(avg_iou * (1 - avg_dist)), avg_iou, avg_dist]


    def update_relative_position(self) -> None:
        if self.currently_seen:
            self.center_point = self.calculate_center_point()
            self.pseudodistance = self.calculate_pseudodistance()

    def calculate_center_point(self) -> List[float, float]:
        pass

    def calculate_pseudodistance(self) -> float:
        pass


    def associate_bbox(self, candidates: List[BoundingBox], method: str = "max_iou") -> List[bool, List[BoundingBox]]:
        if method.lower() == "max_iou":
            ious = []
            for k, past in enumerate(self.recent_bboxes):
                ious[k] = []
                for candidate in candidates:
                    ious[k].append(calculate_iou(candidate, past))

            maxs = []
            for iou_list in ious:
                max_iou = max(iou_list)
                if max_iou > DiverTrack._iou_threshold:
                    maxs.append(iou_list.index(max_iou))
                else:
                    maxs.append(None)

            selected_idx = mode(maxs)
            if selected_idx:
                selected_bbox = candidates.pop(selected_idx)
                self.recent_bboxes.append(selected_bbox)
                self.last_pose_time = Time.now()
                ret = True

            else: # If the most common element in the list is None, we didn't find a good match.
                ret = False

            
            return [ret, candidates]

        else:
            raise NotImplementedError(f"Association method {method} not implemented in function associated_bbox.")

    def associate_pose(self, candidates: List[HumanPoseEstimate], method: str = "avg_center") -> List[bool, List[HumanPoseEstimate]]:
        if method == "avg_center":
            dists = []
            for k, past in enumerate(self.recent_poses):
                dists[k] = []
                for candidate in candidates:
                    dists[k].append(calculate_center_dist(candidate, past))

            mins = []
            for dist_list in dists:
                min_dist = min(dist_list)
                if min_dist < DiverTrack._center_threshold:
                    mins.append(dist_list.index(min_dist))
                else:
                    mins.append(None)

            selected_idx = mode(mins)
            if selected_idx:
                selected_pose = candidates.pop(selected_idx)
                self.recent_poses.append(selected_pose)
                self.last_bbox_time = Time.now()
                ret = True

            else: # If the most common element in the list is None, we didn't find a good match.
                ret = False
            return [ret, candidates]

        else:
            raise NotImplementedError(f"Association method {method} not implemented in function associated_bbox.")

    def update_seen(self) -> None:
        self.last_seen =  min(self.last_bbox_time, self.last_pose_time)
        self.currently_seen = ((Time.now() - self.last_seen) < DiverTrack._time_thresh)

        
    