#!/usr/bin/env python

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32
from scipy.spatial import KDTree

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 50 # Number of waypoints we will publish. You can change this number
MAX_DECEL = 0.5

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below


        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        
        self.pose = None
        self.all_wps = None
        self.twod_wps = None
        self.tree_wps = None
        self.stopline_wp_idx = -1

        self.loop()
        
    def loop(self):
        rate = rospy.Rate(20)
        
        while not rospy.is_shutdown():
            if self.pose and self.tree_wps:
                closest_wp_ids = self.get_closest_wp_ids()
                self.publish_wps( closest_wp_ids)
            rate.sleep()
    
    def get_closest_wp_ids(self):
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        closest_id = self.tree_wps.query([x, y], 1)[1]
        
        closest_coord = self.twod_wps[closest_id]
        prev_coord = self.twod_wps[closest_id - 1]
        
        closest_arr = np.array(closest_coord)
        prev_arr = np.array(prev_coord)
        pos_arr = np.array([x, y])
        
        val = np.dot(closest_arr - prev_arr, pos_arr - closest_arr)
        
        if val > 0:
            closest_id = (closest_id + 1) % len(self.twod_wps)
            
        return closest_id
    
    def publish_wps(self, closest_idx):
        #lane = Lane()
        #lane.header = self.all_wps.header
        #lane.waypoints = self.all_wps.waypoints[closest_idx:closest_idx + LOOKAHEAD_WPS]
        lane = self.generate_lane(closest_idx)
        self.final_waypoints_pub.publish(lane)
        
    def generate_lane(self, closest_idx):
        lane = Lane()
        farthest_idx = closest_idx + LOOKAHEAD_WPS
        horizon_wps = self.all_wps.waypoints[closest_idx:farthest_idx]
        
        if self.stopline_wp_idx == -1 or (self.stopline_wp_idx >= farthest_idx):
            lane.waypoints = horizon_wps
        else:
            lane.waypoints = self.decelerate_wps(horizon_wps, closest_idx)
            
        return lane
            
    def decelerate_wps(self, wps, closest_idx):
        temp = []
        stop_idx = max(self.stopline_wp_idx - closest_idx - 4, 0)
        for i, wp in enumerate(wps):
            p = Waypoint()
            p.pose = wp.pose
           
            dist = self.distance(wps, i, stop_idx)
            vel = math.sqrt(2*MAX_DECEL*dist) + (i / LOOKAHEAD_WPS)
            if vel < 1:
                vel = 0
                
            p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
            temp.append(p)
            
        return temp
        

    def pose_cb(self, msg):
        # TODO: Implement
        self.pose = msg

    def waypoints_cb(self, waypoints):
        # TODO: Implement
        self.all_wps = waypoints
        if not self.twod_wps:
            self.twod_wps = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.tree_wps = KDTree(self.twod_wps)
            

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        print msg.data
        self.stopline_wp_idx = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
