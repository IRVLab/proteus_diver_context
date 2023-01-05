#! /usr/bin/python3

# /data/proteus_ws/proteus/bin/python

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
        img = draw_drp(img, diver.relative_position)

    return img

def draw_filtered_diver(img, diver) -> None:
    color = ColorHash(diver.diver_id).rgb
    if diver.currently_seen:
        img = draw_name(img, diver.diver_id, [diver.filtered_bbox.xmin, diver.filtered_bbox.ymin], color)
        img = draw_bbox(img, diver.filtered_bbox, color)
        img = draw_pose(img, diver.filtered_pose, color)
        img = draw_drp(img, diver.relative_position)

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
        cp = drp.center_point_abs
        img = cv2.circle(img, (int(cp.x), int(cp.y)), radius=7, color=(0,200,100), thickness=-1)
        pd = drp.distance.distance_ratio
    
        if pd > drp.distance.edge:
            b = (255/3.00) * (pd/1.0) # The further we are, the more blue there is.
            selected_color = (b, 255, 255)
        elif pd < drp.distance.personal:
            r = (255/1.0) * ((pd+1)/1.00) # The closer we are after the personal space level, the more red there is.
            b = (255/3.00) * (pd/1.0) # The further we are, the more blue there is.
            selected_color = (b, 0, r)
        else:
            b = (255/3.00) * (pd/1.0) # The further we are, the more blue there is.
            selected_color = (b,50,0)
        img = cv2.circle(img, (int(cp.x), int(cp.y)), radius=int(7 + (10*pd)), color=selected_color, thickness=4)
        img = cv2.putText(img, "{:.2f}".format(pd), (int(cp.x)+15, int(cp.y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,100), 2)

    return img

class DiverVisualizationNode(object):
    def __init__(self) -> None:
        rospy.init_node('diver_visulizer')
        rospy.loginfo("Setting up subscriptions and publishers.")
        # Get topic names from params and set up subscribers
        diver_topic = rospy.get_param('dcm/diver_topic', 'context/divers')
        image_topic = rospy.get_param('dcm/img_topic', '/camera/image_raw')

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
        self.img_q = deque(maxlen=50)
        self.last_divers = None

        self.bridge = CvBridge()
    
    def image_cb(self, msg: Image) -> None:
        self.last_img = msg
        # self.img_q.append(msg)

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


    def get_best_img_match(self, divers):
        sum_seq = 0
        n = 0
        for d in divers.divers:
            if d.currently_seen:
                sum_seq += d.latest_img_header.seq
                n += 1

        if n > 0:
            avg_seq = int(sum_seq/n)
            for img in self.img_q:
                seq = img.header.seq
                if seq == avg_seq:
                    return img
            
        # If we haven't gotten to one yet, just return the most recent image.
        return self.img_q[0]
            

    def update(self) -> None:

        if self.last_divers is not None and self.last_img is not None:
            # img  = self.get_best_img_match(self.last_divers)
            img = self.last_img
            latest_img = self.create_diver_vis_img(img, mode='latest')
            filtered_img = self.create_diver_vis_img(img, mode='filtered')

            latest_msg = self.bridge.cv2_to_imgmsg(latest_img, encoding="bgr8")
            filtered_msg = self.bridge.cv2_to_imgmsg(filtered_img, encoding="bgr8")
            
            self.vis_image_pub_latest.publish(latest_msg)
            self.vis_image_pub_filtered.publish(filtered_msg)

        else:
            return

if __name__ == '__main__':
    dvn = DiverVisualizationNode()
    rate = rospy.Rate(10)

    while not rospy.is_shutdown():
        dvn.update()
        rate.sleep()