#! /usr/bin/python3

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge

from colorhash import ColorHash

from geometry_msgs.msg import Point
from sensor_msgs.msg import Image, CompressedImage
from proteus_msgs.msg import DiverGroup, Diver, RelativePosition, Pseudodistance
from darknet_ros_msgs.msg import BoundingBox
from body_pose_msgs.msg import HumanPoseEstimate, BodyPartEstimate

def draw_diver(img, diver) -> None:
    color = ColorHash(diver.diver_id).rgb
    if diver.currently_seen:
        img = draw_name(img, diver.diver_id, [diver.latest_bbox.xmin, diver.latest_bbox.ymin], color)
        img = draw_bbox(img, diver.latest_bbox, color)
        img = draw_pose(img, diver.latest_pose, color)
        img = draw_drp(img, diver.location)

    return img

def draw_name(img, name, corner, color) -> None:
    img = cv2.putText(img, name, [corner[0], corner[1]-10], cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)
    word_length = len(name) * 5
    img = cv2.rectangle(img, [corner[0], corner[1]-35], [corner[0] + word_length, corner[1]], color, -1)
    return img

def draw_bbox(img, box, color) -> None:
    if box:
        img = cv2.rectangle(img, [box.xmin, box.ymin], [box.xmax, box.ymax], color)
    return img

def draw_pose(img, pose, color) -> None:
    if pose:
        for bp in pose.body_part:
            img = cv2.circle(img, [bp.abs_x, bp.abs_y], radius=2, color=color, thickness=-1)
    return img

def draw_drp(img, drp) -> None:
    if drp:
        pass
    return img

class DiverVisualizationNode(object):
    def __init__(self) -> None:
        rospy.loginfo("Setting up subscriptions and publisers.")
        # Get topic names from params and set up subscribers
        image_topic = rospy.get_param('dcm/diver_topic', 'context/divers')
        diver_topic = rospy.get_param('dcm/image_topic', '/camera/image_raw')

        self.image_sub = rospy.Subscriber(image_topic, Image, self.image_cb, queue_size=5)
        self.diver_sub = rospy.Subscriber(diver_topic, DiverGroup, self.diver_cb, queue_size=5)
        
        vis_topic = rospy.get_param('dvm/vis_topic', 'context/diver_image')
        self.vis_image_pub = rospy.Publisher(vis_topic, CompressedImage, queue_size=5)

        # Now it's time to set up the publishers
        diver_topic = rospy.get_param('dcm/diver_topic', 'context/divers')
        self.diver_pub = rospy.Publisher(diver_topic, DiverGroup, queue_size=5)
        self.update_freq = rospy.get_param('dcm/update_frequency', 10)

        self.last_img = None
        self.last_divers = None

        self.bridge = CvBridge()
    
    def image_cb(self, msg: Image) -> None:
        self.last_img = self.bridge.imgmsg_to_cv2(msg)

    def diver_cb(self, msg: DiverGroup) -> None:
        self.last_divers = msg

    def create_diver_vis_img(self):
        img = self.last_img
        for diver in self.last_divers.divers:
            img = draw_diver(img, diver)
        return img

    def update(self) -> None:
        img = self.create_diver_vis_img()
        msg = self.bridge.cv2_to_compressed_imgmsg(img)
        self.vis_image_pub.publish(msg)

if __name__ == '__main__':
    dvn = DiverVisualizationNode()
    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        dvn.update()
        rate.sleep()
    