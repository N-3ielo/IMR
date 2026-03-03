"""
Copyright (c) 2026 The uos_feeg6043_build Authors.
Authors: Blair Thornton, Sam Fenton, Miquel Massot 

All rights reserved. Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""
import numpy as np
import argparse
from datetime import datetime
import time
from pathlib import Path
from scipy.spatial.transform import Rotation as R

from drivers.aruco_udp_driver import ArUcoUDPDriver
from zeroros import Subscriber, Publisher
from zeroros.messages import RBLaserScan, Vector3Stamped, Pose, PoseStamped, Header, Quaternion
from zeroros.datalogger import DataLogger
from zeroros.rate import Rate

# add more libraries here
from model_feeg6043 import ActuatorConfiguration
from model_feeg6043 import rigid_body_kinematics
from model_feeg6043 import RangeAngleKinematics
from model_feeg6043 import TrajectoryGenerate
from model_feeg6043 import feedback_control

from math_feeg6043 import Inverse, HomogeneousTransformation
from math_feeg6043 import l2m
from math_feeg6043 import Vector

class LaptopPilot:
    def __init__(self, simulation):
        # network for sensed pose
        aruco_params = {
            "port": 50001,  # Port to listen Arena1: 50001; Arena2: 50002 (CHANGE THIS to match the Arena you are testing in)
            "marker_id": 20,  # Marker ID to listen to (CHANGE THIS to your marker ID)            
        }
        self.robot_ip = "192.168.90.1"

        # handles different time reference, network amd aruco parameters for simulator
        self.sim_time_offset = 0 #used to deal with webots timestamps
        self.sim_init = False #used to deal with webots timestamps
        self.simulation = simulation

        if self.simulation:
            aruco_params = {
                "port": 50000,  # Port to listen to (DO NOT CHANGE)
                "marker_id": 0,  # Marker ID to listen to (CHANGE THIS to your marker ID)            
            }
            self.robot_ip = "127.0.0.1"          
            aruco_params['marker_id'] = 0  #Ovewrites Aruco marker ID to 0 (needed for simulation)
            self.sim_init = True #used to deal with webots timestamps

            #Creates a fake Aruco log
            filename_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = Path("logs")
            log_dir.mkdir(parents=True, exist_ok=True) #Auto-create folder if missing
            self.groundtruth_log = log_dir / f"{filename_time}_pseudo_aruco.csv"
            with self.groundtruth_log.open('w') as f:
                f.write("epoch [s],elapsed [s],x [m],y [m],z [m],roll [deg],pitch [deg],yaw [deg],broadcast\n")
        
        self.broadcast = None # stores aruco broadcast timestamps to flag assignment in pseudo aruco (simulated) data. Not used for physical system
        self.pending_groundtruth = None   # holds the last aruco sample until the next callback. Only used for simulation. Not used for physical system
        self.stop_flag = False # a flag to stop wheels when code ends
        
        print("Connecting to robot with IP", self.robot_ip)
        self.aruco_driver = ArUcoUDPDriver(aruco_params, parent=self)

        ############# INITIALISE ATTRIBUTES ##########        
        # path
        self.initialise_pose = True # False once the pose is initialised 
        self.northings_path = [0,1.2,1.2,0.2,0.2]
        self.eastings_path = [0,0.2,1.2,1.2,0]
        self.relative_path = True        

        # model pose
        self.est_pose_northings_m = None
        self.est_pose_eastings_m = None
        self.est_pose_yaw_rad = None

        # velocity, acceleration and turning radius
        self.v = 0.1
        self.a = 0.1/3
        self.arc_radius = 0.2
        self.accept_radius = 0.5

        # control parameters        
        self.tau_s = 1 # s to remove along track error
        self.L = 0.2 # m distance to remove normal and angular error
        self.v_max = 1 # fastest the robot can go
        self.w_max = np.pi/2 # fastest the robot can turn
        self.initialise_control = True # False once control gains is initialised 

        # measured pose
        self.measured_pose_timestamp_s = None
        self.measured_pose_northings_m = None
        self.measured_pose_eastings_m = None
        self.measured_pose_yaw_rad = None

        # wheel speed commands
        self.cmd_wheelrate_right = None
        self.cmd_wheelrate_left = None 

        # encoder/actual wheel speeds
        self.measured_wheelrate_right = None
        self.measured_wheelrate_left = None   

        # lidar
        self.lidar_timestamp_s = None
        self.lidar_data = None
        self.corners = [] # list of corners locations. Append to this using self.corners.append([n,e])
        
        # lidar
        lidar_xb = 0.04 # location of lidar centre in b-frame primary axis
        lidar_yb = 0.0 # location of lidar centre in b-frame secondary axis
        self.lidar = RangeAngleKinematics(lidar_xb, lidar_yb)

        # modelling parameters
        wheel_distance = 0.16 # measure this
        wheel_diameter = 0.07 # measure this
        self.ddrive = ActuatorConfiguration(wheel_distance/2, wheel_diameter) #look at your tutorial and see how to use this
        
        ###############################################################        
        
        self.datalog = DataLogger(log_dir="logs")
        # Wheels speeds in rad/s are encoded as a Vector3 with timestamp, 
        # with x for the right wheel and y for the left wheel.        
        self.wheel_speed_pub = Publisher(
            "/wheel_speeds_cmd", Vector3Stamped, ip=self.robot_ip
        )

        self.true_wheel_speed_sub = Subscriber(
            "/true_wheel_speeds",Vector3Stamped, self.true_wheel_speeds_callback,ip=self.robot_ip,
        )
        self.lidar_sub = Subscriber(
            "/lidar", RBLaserScan, self.lidar_callback, ip=self.robot_ip
        )
        self.groundtruth_sub = Subscriber(
         "/groundtruth", PoseStamped, self.groundtruth_callback, ip=self.robot_ip
        )

    def stopcommand(self):        
        print("Wheels stopping")
        r = Rate(10.0)

        self.stop_flag = True

        stop_msg = Vector3Stamped() # initially 0
        
        for i in range(10):
            stop_msg.vector.x=0
            stop_msg.vector.y=0
            self.wheel_speed_pub.publish(stop_msg)            
            r.sleep()

        self.lidar_sub.stop()
        self.true_wheel_speed_sub.stop()

        print("Data saved in ",self.datalog.filename)
                 
    def true_wheel_speeds_callback(self, msg):
        print("Received sensed wheel speeds: R=", msg.vector.x,", L=", msg.vector.y)

        self.measured_wheelrate_right = msg.vector.x
        self.measured_wheelrate_left = msg.vector.y
        self.datalog.log(msg, topic_name="/true_wheel_speeds")

    def lidar_callback(self, msg):
        # This is a callback function that is called whenever a message is received        
        print("Received lidar message", msg.header.seq)

        if self.sim_init == True:
            self.sim_time_offset = datetime.utcnow().timestamp()-msg.header.stamp
            self.sim_init = False     

        msg.header.stamp += self.sim_time_offset

        self.lidar_timestamp_s = msg.header.stamp #we want the lidar measurement timestamp here

        # b to e frame
        p_eb = Vector(3)
        p_eb[0] = self.est_pose_northings_m if self.est_pose_northings_m is not None else 0 #robot pose northings
        p_eb[1] = self.est_pose_eastings_m if self.est_pose_eastings_m is not None else 0 #robot pose eastings
        p_eb[2] = self.est_pose_yaw_rad if self.est_pose_yaw_rad is not None else 0 #robot pose yaw

        # m to e frame
        self.lidar_data = np.zeros((len(msg.ranges), 2))

        z_lm = Vector(2)
        # for each map measurement
        for i in range(len(msg.ranges)):
            z_lm[0] = msg.ranges[i]
            z_lm[1] = msg.angles[i]

            t_em = self.lidar.rangeangle_to_loc(p_eb, z_lm)

            self.lidar_data[i,0] = float(t_em[0])
            self.lidar_data[i,1] = float(t_em[1])

        # this filters out any NaN
        self.lidar_data = self.lidar_data[~np.isnan(self.lidar_data).any(axis=1)]
        self.datalog.log(msg, topic_name="/lidar")

    def groundtruth_callback(self, msg):
        """Log the ground-truth sample using the broadcast timestamp snapshot
        from the previous loop iteration, then buffer the current sample."""
        t = msg.header.stamp
        n = msg.pose.position.x
        e = msg.pose.position.y
        d = msg.pose.position.z
        q = [msg.pose.orientation.x, msg.pose.orientation.y,
             msg.pose.orientation.z, msg.pose.orientation.w]
        r = R.from_quat(q)  # [x, y, z, w]
        roll, pitch, yaw = r.as_euler('xyz', degrees=True)
        yaw = np.mod(yaw, 360.0)

        # 1) If we have a previous sample buffered, log it now against the *previous* broadcast snapshot.
        if self.pending_groundtruth is not None:
            t_prev, n_prev, e_prev, d_prev, roll_prev, pitch_prev, yaw_prev = self.pending_groundtruth

            # Precise check: True if the previous ground-truth timestamp equals the broadcast timestamp
            is_broadcast_prev = (t_prev == self.broadcast)            

            with self.groundtruth_log.open('a') as f:
                f.write(
                    f"{t_prev},{t_prev - self.start_time},{n_prev},{e_prev},{d_prev},"
                    f"{roll_prev},{pitch_prev},{yaw_prev},{is_broadcast_prev}\n"
                )        
        self.pending_groundtruth = (t, n, e, d, roll, pitch, yaw)


    
    def pose_parse(self, msg, aruco = False):
        # parser converts pose data to a standard format for logging
        time_stamp = msg[0]
        

        if aruco == True:
            if self.sim_init == True: 
                self.sim_time_offset = datetime.utcnow().timestamp()-msg[0]               
                self.sim_init = False                                         
                
            # self.sim_time_offset is 0 if not a simulation. Deals with webots dealing in elapse timeself.sim_time_offset
            print(
                "Received position update from",
                datetime.utcnow().timestamp() - msg[0],
                "seconds ago",
            )            

        pose_msg = PoseStamped() 
        pose_msg.header = Header()
        pose_msg.header.stamp = time_stamp
        pose_msg.pose.position.x = msg[1]
        pose_msg.pose.position.y = msg[2]
        pose_msg.pose.position.z = 0

        quat = Quaternion()        
        if self.simulation == False and aruco == True: quat.from_euler(0, 0, np.deg2rad(msg[6]))
        else: quat.from_euler(0, 0, msg[6])
        pose_msg.pose.orientation = quat            
        

        return pose_msg
    
    def generate_trajectory(self):
        # pick waypoints as current pose relative or absolute northings and eastings
        if self.relative_path == True:
            for i in range(len(self.northings_path)):
                self.northings_path[i] += self.est_pose_northings_m #offset by current northings
                self.eastings_path[i] += self.est_pose_eastings_m #offset by current eastings

            # convert path to matrix and create a trajectory class instance
            C = l2m([self.northings_path, self.eastings_path])        
            self.path = TrajectoryGenerate(C[:, 0], C[:, 1])        
            
            # set trajectory variables (velocity, acceleration and turning arc radius)
            self.path.path_to_trajectory(self.v, self.a) #velocity and acceleration
            self.path.turning_arcs(self.arc_radius) #turning radius
            self.path.wp_id=0 #initialises the next waypoint


    def run(self, time_to_run=-1):
        self.start_time = datetime.utcnow().timestamp()
        
        try:
            r = Rate(10.0)
            while True:
                current_time = datetime.utcnow().timestamp()
                if time_to_run > 0 and current_time - self.start_time > time_to_run:
                    print("Time is up, stopping…")                    
                    break
                self.infinite_loop()
                r.sleep()
        except KeyboardInterrupt:
            print("KeyboardInterrupt received, stopping…")
        except Exception as e:
            print("Exception: ", e)
        finally:
            self.stopcommand()


    def infinite_loop(self):
        """Main control loop

        Your code should go here.
        """
        # > Sense < #
        # get the latest position measurements
        aruco_pose = self.aruco_driver.read() 

        wheel_speed_msg = Vector3Stamped()
        wheel_speed_msg.vector.x = 0
        wheel_speed_msg.vector.y = 0   

        q = Vector(2)
        q[0] = self.measured_wheelrate_right if self.measured_wheelrate_right is not None else 0
        q[1] = self.measured_wheelrate_left if self.measured_wheelrate_left is not None else 0
        u = self.ddrive.fwd_kinematics(q)

        if aruco_pose is not None:

            # converts aruco date to zeroros PoseStamped format                         
            msg = self.pose_parse(aruco_pose, aruco = True)
            self.broadcast = msg.header.stamp 

            # reads sensed pose for local use 
            self.measured_pose_timestamp_s = msg.header.stamp
            self.measured_pose_northings_m = msg.pose.position.x
            self.measured_pose_eastings_m = msg.pose.position.y
            _, _, self.measured_pose_yaw_rad = msg.pose.orientation.to_euler()
            self.measured_pose_yaw_rad = self.measured_pose_yaw_rad % (np.pi*2) # manage angle wrapping

            # logs the data
            self.datalog.log(msg, topic_name="/aruco")

            ###### wait for the first sensor info to initialize the pose ######
            if self.initialise_pose == True:
                self.est_pose_northings_m = self.measured_pose_northings_m
                self.est_pose_eastings_m = self.measured_pose_eastings_m
                self.est_pose_yaw_rad = self.measured_pose_yaw_rad

                # generates trajectory
                self.generate_trajectory()

                # get current time and determine timestep
                self.t_prev = datetime.utcnow().timestamp() #initialise the time
                self.t = 0 #elapsed time
                time.sleep(0.1) #wait for approx a timestep before proceeding

                # path and trajectory are initialised
                self.initialise_pose = False

        if self.initialise_pose != True:
   
            # determine the time step
            t_now = datetime.utcnow().timestamp()
            dt = t_now - self.t_prev
            self.t += dt
            self.t_prev = t_now

            # take current pose estimate and update by twist
            p_robot = Vector(3)
            p_robot[0] = self.est_pose_northings_m
            p_robot[1] = self.est_pose_eastings_m
            p_robot[2] = self.est_pose_yaw_rad

            p_robot = rigid_body_kinematics(p_robot, u, dt)
            p_robot[2] = p_robot[2] % (2 * np.pi)


            #################### Trajectory sample #################################    

            #feedforward control: check wp progress and sample reference trajectory
            self.path.wp_progress(self.t, p_robot, self.accept_radius) # fill turning radius
            p_ref, u_ref = self.path.p_u_sample(self.t) #sample the path at the current elapsetime (i.e., seconds from start of motion modelling)

            # feedback control: get pose change to desired trajectory from body
            dp = p_ref - p_robot #compute difference between reference and estimated pose in the $e$-frame
            dp[2] = (dp[2] + np.pi) % (2 * np.pi) - np.pi # handle angle wrapping for yaw

            H_eb = HomogeneousTransformation(p_robot[0:2], p_robot[2])
            ds = Inverse(H_eb.H_R)@dp # rotate the $e$-frame difference to get it in the $b$-frame (Hint: dp_b = H_be.H_R @ dp_e)

            # compute control gains for the initial condition (where the robot is stationalry)
            self.k_s = 1/self.tau_s #ks
            if self.initialise_control == True:
                self.k_n = 2 * u_ref[0]/ (self.L**2) #kn
                self.k_g = u_ref[0]/self.L #kg
                self.initialise_control = False # maths changes a bit after the first iteration

            # update the controls
            du = feedback_control(ds, self.k_s, self.k_n, self.k_g)

            # total control
            u = u_ref + du # combine feedback and feedforward control twist components

            # update control gains for the next timestep
            self.k_n = 2*u[0]/self.L**2 #kn
            self.k_g = u[0]/self.L #kg

            # ensure within performance limitation
            if u[0] > self.v_max: u[0] = self.v_max
            if u[0] < -self.v_max: u[0] = -self.v_max
            if u[1] > self.w_max: u[1] = self.w_max
            if u[1] < -self.w_max: u[1] = -self.w_max

            # actuator commands                 
            q = self.ddrive.inv_kinematics(u)            

            wheel_speed_msg = Vector3Stamped()
            wheel_speed_msg.vector.x = q[0,0] # Right wheelspeed rad/s
            wheel_speed_msg.vector.y = q[1,0] # Left wheelspeed rad/s

            self.est_pose_northings_m = float(p_robot[0])
            self.est_pose_eastings_m = float(p_robot[1])
            self.est_pose_yaw_rad = float(p_robot[2])

            # self.est_pose_northings_m = float(p_ref[0])
            # self.est_pose_eastings_m = float(p_ref[1])
            # self.est_pose_yaw_rad = float(p_ref[2])

        # > Think < #
        ################################################################################
        #  TODO: Implement your state estimation

        if self.est_pose_northings_m is not None:
            msg = self.pose_parse([datetime.utcnow().timestamp(),self.est_pose_northings_m,self.est_pose_eastings_m,0,0,0,self.est_pose_yaw_rad])
            self.datalog.log(msg, topic_name="/est_pose")
        ################################################################################
        #  TODO: Implement your controller here                                        #

        # wheel_speed_msg = Vector3Stamped()
        # wheel_speed_msg.vector.x = 1.5 * np.pi  # Right wheel 1 rev/s = 1*pi rad/s
        # wheel_speed_msg.vector.y = 2 * np.pi  # Left wheel 1 rev/s = 2*pi rad/s

        # self.cmd_wheelrate_right = wheel_speed_msg.vector.x
        # self.cmd_wheelrate_left = wheel_speed_msg.vector.y
        ################################################################################

        # > Act < #
        # Send commands to the robot        
        if self.stop_flag == False: self.wheel_speed_pub.publish(wheel_speed_msg)
        self.datalog.log(wheel_speed_msg, topic_name="/wheel_speeds_cmd")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--time",
        type=float,
        default=-1,
        help="Time to run an experiment for. If negative, run forever.",
    )
    parser.add_argument(
        "--simulation",
        action="store_true",
        help="Run in simulation mode. Defaults to False",
    )

    args = parser.parse_args()

    laptop_pilot = LaptopPilot(args.simulation)
    laptop_pilot.run(args.time)
