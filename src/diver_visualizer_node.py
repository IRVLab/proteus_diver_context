#! /usr/bin/python3

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge

from colorhash import ColorHash
from collections import deque

from geometry_msgs.msg import Point
from sensor_msgs.msg import Image, CompressedImage
from proteus_msgs.msg import DiverGroup, Diver, RelativePosition, Pseudodistance
from darknet_ros_msgs.msg import BoundingBox
from body_pose_msgs.msg import HumanPoseEstimate, BodyPartEstimate

def draw_latest_diver(img, diver) -> None:
    color = ColorHash(diver.diver_id).rgb
    if diver.currently_seen:
        img = draw_name(img, diver.diver_id, [diver.latest_bbox.xmin, diver.latest_bbox.ymin], color)
        img = draw_bbox(img, diver.latest_bbox, color)
        img = draw_pose(img, diver.latest_pose, color)
        img = draw_drp(img, diver.location)

    return img

def draw_filtered_diver(img, diver) -> None:
    color = ColorHash(diver.diver_id).rgb
    if diver.currently_seen:
        img = draw_name(img, diver.diver_id, [diver.filtered_bbox.xmin, diver.filtered_bbox.ymin], color)
        img = draw_bbox(img, diver.filtered_bbox, color)
        img = draw_pose(img, diver.filtered_pose, color)
        img = draw_drp(img, diver.location)

    return img

def draw_name(img, name, corner, color) -> None:
    word_length = len(name) * 25
    img = cv2.rectangle(img, (corner[0]-5, corner[1]-40), (corner[0] + word_length, corner[1]), color, -1)
    img = cv2.putText(img, name, (corner[0]+5, corner[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0), 2)
    
    return img

def draw_bbox(img, box, color) -> None:
    if box is not None:
        img = cv2.rectangle(img, (box.xmin, box.ymin), (box.xmax, box.ymax), color, 5)
    return img

def draw_pose(img, pose, color) -> None:
    if pose is not None:
        for bp in pose.body_parts:
            if bp.visible:
                img = cv2.circle(img, (bp.abs_x, bp.abs_y), radius=5, color=color, thickness=-1)
                img = cv2.circle(img, (bp.abs_x, bp.abs_y), radius=6, color=(0,0,0), thickness=1)
                img = cv2.putText(img, bp.id, (bp.abs_x+10, bp.abs_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2)
    return img

def draw_drp(img, drp) -> None:
    if drp is not None:
        pass
    return img

class DiverVisualizationNode(object):
    def __init__(self) -> None:
        rospy.init_node('diver_visulizer')
        rospy.loginfo("Setting up subscriptions and publishers.")
        # Get topic names from params and set up subscribers
        diver_topic = rospy.get_param('dcm/diver_topic', 'context/divers')
        image_topic = rospy.get_param('dcm/image_topic', '/camera/image_raw')

        self.image_sub = rospy.Subscriber(image_topic, Image, self.image_cb, queue_size=5)
        self.diver_sub = rospy.Subscriber(diver_topic, DiverGroup, self.diver_cb, queue_size=5)
        
        vis_topic = rospy.get_param('dvm/vis_topic', 'context/diver_image')
        self.vis_image_pub_latest = rospy.Publisher(vis_topic + '/latest', Image, queue_size=5)
        self.vis_image_pub_filtered = rospy.Publisher(vis_topic + '/filtered', Image, queue_size=5)

        # Now it's time to set up the publishers
        diver_topic = rospy.get_param('dcm/diver_topic', 'context/divers')
        self.diver_pub = rospy.Publisher(diver_topic, DiverGroup, queue_size=5)
        self.update_freq = rospy.get_param('dcm/update_frequency', 10)

        self.last_img = None
        self.last_divers = None

        self.bridge = CvBridge()
    
    def image_cb(self, msg: Image) -> None:
        self.last_img = msg

    def diver_cb(self, msg: DiverGroup) -> None:
        self.last_divers = msg

    def create_diver_vis_img(self, img_msg, mode):
        if mode == "latest":
            img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
            for diver in self.last_divers.divers:
                img = draw_latest_diver(img, diver)
            return img

        elif mode == "filtered":
            img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
            for diver in self.last_divers.divers:
                img = draw_filtered_diver(img, diver)
            return img


    def get_best_img_match(self, timestamp):
        raise DeprecationWarning("NOT USED ANYMORE")
        return None

        # I'm leaving the below implementation in here, because I think it's worth looking at, 
        # But it doesn't seem to be working right now, so I'm not gonna use it.
        best_img = None
        min_time = 100

        for img in self.recent_imgs:
            t_dist = (img.header.stamp - timestamp ).to_sec()

            if t_dist > 0 and t_dist < min_time:
                min_time = t_dist
                best_img = img

        # If we can't find a good match, just return the most recent image.
        if best_img is None:
            rospy.logdebug(f"No best match found.")
            return self.bridge.imgmsg_to_cv2(self.recent_imgs.pop(), desired_encoding="bgr8")
        else:
            rospy.logdebug(f"Found best match, time_dist {min_time}")
            self.recent_imgs.remove(best_img)
            return self.bridge.imgmsg_to_cv2(best_img, desired_encoding="bgr8")

    def update(self) -> None:
        if self.last_divers is not None and self.last_img is not None:
            latest_img = self.create_diver_vis_img(self.last_img, mode='latest')
            filtered_img = self.create_diver_vis_img(self.last_img, mode='filtered')

            latest_msg = self.bridge.cv2_to_imgmsg(latest_img, encoding="bgr8")
            filtered_msg = self.bridge.cv2_to_imgmsg(filtered_img, encoding="bgr8")
            
            self.vis_image_pub_latest.publish(latest_msg)
            self.vis_image_pub_filtered.publish(filtered_msg)

        elif self.last_img is not None:
            self.vis_image_pub_latest.publish(self.last_img)
            self.vis_image_pub_filtered.publish(self.last_img)
        else:
            return

if __name__ == '__main__':
    dvn = DiverVisualizationNode()
    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        dvn.update()
        rate.sleep()
    