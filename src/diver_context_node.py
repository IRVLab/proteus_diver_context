#! /usr/bin/python3
import rospy
import random

from std_msgs.msg import Header
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image
from darknet_ros_msgs.msg import BoundingBoxes, BoundingBox
from body_pose_msgs.msg import HumanPoseEstimateGroup, HumanPoseEstimate,BodyPartEstimate
from deeplabcut_ros.srv import ProcessDiverPose
from proteus_msgs.msg import DiverGroup, Diver, RelativePosition, Pseudodistance

from diver_names import undersea_explorers
from diver_track import DiverTrack

# The DiverContextNode is responsible for keeping track of divers that have been recently seen.
class DiverContextNode(object):
    _conf_thresh = 0.75

    def __init__(self) -> None:
        rospy.init_node("proteus_diver_context_module", anonymous=False)

        rospy.loginfo("Setting up subscriptions and publishers.")
        # Get topic names from params and set up subscribers
        image_topic = rospy.get_param('dcm/img_topic', '/camera/image_raw')
        bbox_topic = rospy.get_param('dcm/bbox_topic', '/darknet_ros/bounding_boxes')
        # pose_topic = rospy.get_param('dcm/pose_topic', '/deeplabcut_ros/pose_estimates')

        # Get base image dimmensions
        self.img = rospy.wait_for_message(image_topic, Image)
        self.img_sub = rospy.Subscriber(image_topic, Image, self.img_cb)
        self.img_dims = [float(self.img.width), float(self.img.height)]

        self.bbox_sub = rospy.Subscriber(bbox_topic, BoundingBoxes, self.bbox_cb, queue_size=5)
        # self.pose_sub = rospy.Subscriber(pose_topic, HumanPoseEstimateGroup, self.pose_cb, queue_size=5)
        rospy.loginfo("Waiting for pose service...")
        rospy.wait_for_service('/deeplabcut_ros/process_diver_pose')
        self.pose_srv = rospy.ServiceProxy('/deeplabcut_ros/process_diver_pose', ProcessDiverPose)
        rospy.loginfo("Pose service proxy established.")
        self.head_ang_sub = None # This is left unimplemented for now, but if you feel like adding head angle estimation, go for it.

        # Now it's time to set up the publishers
        diver_topic = rospy.get_param('dcm/diver_topic', 'context/divers')
        self.diver_pub = rospy.Publisher(diver_topic, DiverGroup, queue_size=5)
        self.update_freq = rospy.get_param('dcm/update_frequency', 10)

        # Now, we need to create our variables that hold the most recent message
        self.last_bbox = None
        self.last_pose = None
        self.last_hang = None

        # Now, time to create our internal diver collection.
        self.queue_length = rospy.get_param('dmc/obs_queue_len', 10)
        self.stale_timeout = rospy.get_param('dmc/stale_timeout', 60)
        self.diver_tracks = {}
        self.names_in_use = []

    def img_cb(self, msg) -> None:
        self.img = msg

    # Record the most recent bounding box.
    def bbox_cb(self, msg) -> None:
        self.last_bbox = msg

    # # Reocrd the recent pose message.
    # def pose_cb(self, msg) -> None:
    #     self.last_pose = msg

    # Add a diver to the current tracks.
    def add_diver_track(self, bbox=None, pose=None) -> DiverTrack:
        if (bbox is not None) or (pose is not None):
            possible_names = [name for name in undersea_explorers if name not in self.names_in_use]
            name = random.choice(possible_names)
            self.names_in_use.append(name)
            self.diver_tracks[name] = DiverTrack(name, bbox=bbox, pose=pose, queue_size=self.queue_length, dims=self.img_dims)
            return self.diver_tracks[name]

    # Update all diver tracks.
    def update_diver_tracks(self) -> None:
        if self.last_bbox is not None:
                candidates = self.last_bbox.bounding_boxes
        else:
            candidates = []

        for key, track in self.diver_tracks.items():
            # If there are candidate bboxes, attempt to associate them
            if len(candidates) > 0:
                success, candidates = track.associate_bbox(candidates)

            track.get_pose(self.pose_srv, self.img) #Get pose for every track
            track.update_seen()
            track.update_relative_position()

        # At this point, any remaining candidates need a new diver, if they reach the confidence threshold
        for candidate in candidates:
            if candidate.probability > DiverContextNode._conf_thresh:
                self.add_diver_track(bbox=candidate).update_seen()

    def cull_stale(self) -> None:
        keys_to_cull = []
        for key, track in self.diver_tracks.items():
            if not track.currently_seen:
                if (rospy.Time.now() - track.last_seen).to_sec() < self.stale_timeout:
                    keys_to_cull.append(key)

        for key in keys_to_cull:
            self.diver_tracks.pop(key)
            self.names_in_use.pop(self.names_in_use.index(key))

    # Publish diver group.
    def publish_divers(self) -> None:
        msg = DiverGroup()
        msg.header = Header()

        for key, track in self.diver_tracks.items():
            d_msg = Diver()
            d_msg.diver_id = key
            d_msg.estimated_confidence = track.calculate_track_confidence()
            d_msg.last_seen = track.last_seen
            d_msg.currently_seen = track.currently_seen
            box = track.get_latest_bbox()
            if box is not None:
                d_msg.latest_bbox = box

            pose = track.get_latest_pose()
            if pose is not None:
                d_msg.latest_pose = pose

            # d_msg.location = RelativePosition()
            # cp, pd = track.get_relative_position()
            # d_msg.location.center_point_rel = Point()
            # d_msg.location.center_point_rel.x = cp[0]
            # d_msg.location.center_point_rel.y = cp[1]
            # d_msg.location.center_point_rel.z = 0.0
            # d_msg.location.distance = Pseudodistance()
            # d_msg.location.distance.distance_ratio = pd

            msg.divers.append(d_msg)

        msg.header.stamp = rospy.Time.now()
        self.diver_pub.publish(msg)


if __name__ == '__main__':
    dcn = DiverContextNode()

    rate = rospy.Rate(dcn.update_freq)
    while not rospy.is_shutdown():
        dcn.update_diver_tracks()
        dcn.cull_stale()
        dcn.publish_divers()
        rate.sleep()