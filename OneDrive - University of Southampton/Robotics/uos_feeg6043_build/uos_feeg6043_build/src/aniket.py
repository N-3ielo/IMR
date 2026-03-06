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
import copy

from drivers.aruco_udp_driver import ArUcoUDPDriver
from zeroros import Subscriber, Publisher
from zeroros.messages import RBLaserScan, Vector3Stamped, Pose, PoseStamped, Header, Quaternion
from zeroros.datalogger import DataLogger
from zeroros.rate import Rate

# add more libraries here
from model_feeg6043 import ActuatorConfiguration
from math_feeg6043 import Vector
from model_feeg6043 import rigid_body_kinematics
from model_feeg6043 import RangeAngleKinematics
from model_feeg6043 import TrajectoryGenerate
from math_feeg6043 import l2m
from model_feeg6043 import feedback_control
from math_feeg6043 import Inverse, HomogeneousTransformation

#Libraries for Particle Filtering
from model_feeg6043 import kde_probability, systematic_resample, kde_probability
from math_feeg6043 import wrapped_mean, wrapped_std
from math_feeg6043 import polar2cartesian
from sklearn.neighbors import KernelDensity



class LaptopPilot:
    def __init__(self, simulation):
        # network for sensed pose
        aruco_params = {
            "port": 50001,  # Port to listen Arena1: 50001; Arena2: 50002 (CHANGE THIS to match the Arena you are testing in)
            "marker_id": 24,  # Marker ID to listen to (CHANGE THIS to your marker ID)            
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
        self.northings_path = []
        self.eastings_path = []        

        self.initialise_pose = True # False once the pose is initialised

        # model pose
        self.est_pose_northings_m = None
        self.est_pose_eastings_m = None
        self.est_pose_yaw_rad = None

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
        lidar_xb = 0.1 # location of lidar centre in b-frame primary axis
        lidar_yb = 0 # location of lidar centre in b-frame secondary axis
        self.lidar = RangeAngleKinematics(lidar_xb, lidar_yb)

        # modelling parameters
        wheel_distance = 0.17/2 # measure this
        wheel_diameter = 0.07 # measure this
        self.ddrive = ActuatorConfiguration(wheel_distance, wheel_diameter) #look at your tutorial and see how to use this

        self.northings_path = [0, 1.5, 1.5, 0, 0] # create a list of waypoints
        self.eastings_path = [0, 0, 1.5, 1.5, 0] # create a list of waypoints
        self.relative_path = True # False if you want it to be absolute

        self.v = 0.1 #m/s
        self.a = 0.1/3 #takes 3s to get to 0.1m/s
        self.radius = 0.3 #m
        self.accept_radius = 0.2 #m

        # control parameters        
        self.tau_s = 0.5 # s to remove along track error 1
        self.L = 0.2 # m distance to remove normal and angular error
        self.v_max = 1 # fastest the robot can go
        self.w_max = np.deg2rad(90) # fastest the robot can turn
        self.initialise_control = True # False once control gains is initialised

        # PF Parameters
        self.N = 100
        self.northings_std = 0.1   #m
        self.eastings_std  = 0.1 #m
        self.g_std   = np.deg2rad(1)   #rad
        self.x_dot_std = 0.3 #m/s
        self.g_dot_std = np.deg2rad(0.1) #rad/s
       
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

        # update wheel rates
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
           
        self.lidar_data = np.zeros((len(msg.ranges), 2)) #specify length of the lidar data
        self.lidar_data[:,0] = msg.ranges # use ranges as a placeholder, workout northings in Task 4
        self.lidar_data[:,1] = msg.angles # use angles as a placeholder, workout eastings in Task 4
        self.datalog.log(msg, topic_name="/lidar")

        # b to e frame
        p_eb = Vector(3)
        p_eb[0] = self.est_pose_northings_m #robot pose northings (see Task 3)
        p_eb[1] = self.est_pose_eastings_m #robot pose eastings (see Task 3)
        p_eb[2] = self.est_pose_yaw_rad #robot pose yaw (see Task 3)

        # m to e frame
        self.lidar_data = np.zeros((len(msg.ranges), 2))        
                       
        z_lm = Vector(2)        
        # for each map measurement
        for i in range(len(msg.ranges)):
            z_lm[0] = msg.ranges[i]
            z_lm[1] = msg.angles[i]
               
            t_em = self.lidar.rangeangle_to_loc(p_eb, z_lm) # see tutotial

            self.lidar_data[i,0] = t_em[0]
            self.lidar_data[i,1] = t_em[1]

        # this filters out any
        self.lidar_data = self.lidar_data[~np.isnan(self.lidar_data).any(axis=1)]

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
            self.path.turning_arcs(self.radius) #turning radius
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

    ##################################################################################### Particle Filter Function ############################################################################################################
   
    # Measurements
    class Measurement:
        def __init__(self):
            self.timestamp = np.array([None])
            self.northings = np.array([None])
            self.eastings = np.array([None])
            self.northings_std = np.array([None])
            self.eastings_std = np.array([None])  

    # Define Initial States and Weights
       
    class Particles:

        def __init__(self, N):
            # Particle Filter parameters
            self.N = N
            self.northings = np.array([None] * N)  # State - to be estimated
            self.eastings = np.array([None] * N)   # State - to be estimated
            self.gamma = np.array([None] * N)     # State provided with noise
            self.x_dot = np.array([None] * N)   # State provided with noise
            self.gamma_dot = np.array([None] * N)  # State provided with noise
            self.weight = np.ones(N)/float(N)  # Particle weights  
       

   
    # Initialise the States

    def initialise_particle_distribution(self, particles, centre = [0,0], radius = 1, heading = 0):

            # sample angles and ranges
            theta = np.random.uniform(0, np.radians(360), particles.N)
            r = np.sqrt(np.random.uniform(0, radius, particles.N))

            # convert to cartesian    
            northings, eastings = polar2cartesian(r, theta)
       
            # add centre offset
            particles.northings  = northings + centre[0]
            particles.eastings = eastings + centre[1]
       
            # particles could be pointing anywhere
            particles.gamma = np.random.uniform(heading - np.deg2rad(10) , heading + np.deg2rad(10), particles.N)

    # Predict the next States

    def discrete_motion_model(self, particles, gamma, u, dt, process_noise):

        g_std = np.sqrt(process_noise[0]) # we choose to use variance, to make inputs equivalent to EKF
        x_dot_std = np.sqrt(process_noise[1])
        g_dot_std = np.sqrt(process_noise[2])
   
        for i in range(particles.N):

            p = Vector(3)
       
            # noise is not added to the location of the particles as the distribution already represents the noise and this is what our PF will update based on sensor observations
            p[0] = particles.northings[i]
            p[1] = particles.eastings[i]

            # We treat the other states as auxilliary (Rao-Blackwellisation), where noise is random sampled from a distribution and added  
            if gamma != None: p[2] = gamma + np.random.normal(scale = g_std) % (2 * np.pi)
            else: p[2] = particles.gamma[i] + np.random.normal(scale = g_std) % (2 * np.pi)
       
            u_noise = Vector(2)
            u_noise[0] = u[0] + np.random.normal(scale = x_dot_std)
            u_noise[1] = u[1] + np.random.normal(scale = g_dot_std)
       
            # note rigid_body_kinematics already handles the exception dynamics of w=0
            p = rigid_body_kinematics(p, u_noise, dt)    

            # update the particles and store the information
            particles.northings[i] = p[0,0]
            particles.eastings[i] = p[1,0]
            particles.gamma[i] = p[2,0]
            particles.x_dot[i] = u_noise[0,0]
            particles.gamma_dot[i] = u_noise[1,0]    


    # Particle Weight Update

    def pf_measurement_probability(self, particles, measurement):    
        # calculate likelihood of each particle given the observation
        delta = np.sqrt(measurement.northings_std * measurement.eastings_std)
        probability = []

        for i in range(particles.N):
            particle_to_measurement=(particles.northings[i]-measurement.northings)**2+(particles.eastings[i]-measurement.eastings)**2
            num = np.exp(-(particle_to_measurement)/(2*delta**2))
            den = np.sqrt(2*np.pi*delta**2)
            probability.append(num/den)  
       
        return probability
       
    # Prior Probability

    def pf_kde_probability(self, particles, sigma_resolution=1.0, sampling_resolution=None):
        """
        KDE-based scoring on particle positions.

        Parameters
        ----------
        particles : object
            Must expose:
            - particles.northings : array-like (N,)
            - particles.eastings  : array-like (N,)
            - particles.weight    : array-like (N,)  [optional but recommended]
        sigma_resolution : float
            Isotropic Gaussian bandwidth for KDE (same units as northings/eastings).
        sampling_resolution : int or None
            If None, returns density at each particle location.
            Else, evaluates KDE on a full 2D grid of size (sampling_resolution x sampling_resolution)
            spanning the min/max of the particles and returns the (northings, eastings) at the grid argmax.

        Returns
        -------
        If sampling_resolution is None:
            density : ndarray (N,)
                KDE density (not probability) at each particle location.
        Else:
            est_northings : float
            est_eastings  : float
                Coordinates of the grid point with maximum KDE density.
        """
        # Assemble data matrix (N, 2) as (northing, easting)
        locations = np.vstack([particles.northings, particles.eastings]).T

        # Optional sample weights (ensure 1-D if present)
        w = getattr(particles, "weight", None)
        if w is not None:
            w = np.asarray(w).ravel()

        # Fit KDE once (same model used in both branches)
        kde = KernelDensity(kernel='gaussian', bandwidth=float(sigma_resolution))
        kde.fit(locations, sample_weight=w)

        if sampling_resolution is None:
            # Return density (not probability) at the particle locations
            log_density = kde.score_samples(locations)
            density = np.exp(log_density)
            return density

        # ---- Full 2D Cartesian grid over the particle bounds ----
        ny = int(sampling_resolution)
        nx = int(sampling_resolution)

        n_min = float(np.min(locations[:, 0]))
        n_max = float(np.max(locations[:, 0]))
        e_min = float(np.min(locations[:, 1]))
        e_max = float(np.max(locations[:, 1]))

        grid_n = np.linspace(n_min, n_max, num=ny)
        grid_e = np.linspace(e_min, e_max, num=nx)

        # Build full grid and evaluate KDE everywhere
        GE, GN = np.meshgrid(grid_e, grid_n, indexing='xy')  # GE: easting, GN: northing
        grid_pts = np.column_stack([GN.ravel(), GE.ravel()])  # (ny*nx, 2) as (northing, easting)

        log_density_grid = kde.score_samples(grid_pts).reshape(GN.shape)

        # Argmax on the grid
        ij = np.unravel_index(np.argmax(log_density_grid), log_density_grid.shape)
        est_northings = float(GN[ij])
        est_eastings  = float(GE[ij])
   
        return est_northings, est_eastings


    def pf_normalise_weights(self, weights):
        ws = sum(weights)
        for i in range(len(weights)):
            weights[i] /= ws    
        return weights

    def pf_update(self, particles, measurement):
        tau = self.pf_measurement_probability(particles, measurement)
   
        sigma_resolution = np.sqrt(measurement.northings_std * measurement.eastings_std)
        prior_tau = self.pf_kde_probability(particles, sigma_resolution)

        particles.weight *= tau/prior_tau
        self.pf_normalise_weights(particles.weight)

        return particles
   
    def neff(self, particles):
        particles.weight /= np.sum(particles.weight) # normalises the weights
        return 1/(particles.N * np.sum(np.square(particles.weight)))

    def pf_resample(self, particles, jitter, verbose = False):
        """Resample particles using systematic resample"""
       
        if verbose == True: print('effective particles:', self.neff(particles))
        if self.neff(particles) < 0.5:
            # Get the indexes of the particles to resample
            indexes = systematic_resample(particles.weight)
            if verbose == True: print('Resampling needed, sampled particles:',indexes)
               
            # Copy all the particles, to overwrite "particles"
            particles_copy = copy.deepcopy(particles)
           
            angle = np.random.uniform(0, 2*np.pi, particles.N)
            radius = np.random.normal(0, jitter, particles.N)
           
            northings, eastings = polar2cartesian(radius,angle)
           
            # Overwrite "particles" with the copied ones and correct indices
            for i in range(particles.N):
                particles.northings[i] = copy.deepcopy(particles_copy.northings[indexes[i]])+northings[i]
                particles.eastings[i] = copy.deepcopy(particles_copy.eastings[indexes[i]])+eastings[i]
           
            # Reset the weights to equally probable
            particles.weight = np.ones(particles.N)/particles.N
        else:
            if verbose == True: print('No resampling needed')

   
   

    def infinite_loop(self):
        """Main control loop

        Your code should go here.
        """
        # > Sense < #
        # get the latest position measurements
        aruco_pose = self.aruco_driver.read()  

        wheel_speed_msg = Vector3Stamped()
        wheel_speed_msg.vector.x = 0 # Right wheelspeed rad/s
        wheel_speed_msg.vector.y = 0 # Left wheelspeed rad/s

        ################### Motion Model ##############################
        # convert true wheel speeds in to twist
        q = Vector(2)            
        if self.measured_wheelrate_right is not None: q[0] = self.measured_wheelrate_right # wheel rate rad/s (measured)
        if self.measured_wheelrate_left is not None: q[1] = self.measured_wheelrate_left # wheel rate rad/s (measured)
        u = self.ddrive.fwd_kinematics(q)  

        self.gamma = None
     
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

            self.gamma = self.measured_pose_yaw_rad

            ###### wait for the first sensor info to initialize the pose ######
            if self.initialise_pose == True:

                self.est_pose_northings_m = self.measured_pose_northings_m
                self.est_pose_eastings_m = self.measured_pose_eastings_m
                self.est_pose_yaw_rad = self.measured_pose_yaw_rad

                # get current time and determine timestep
                self.t_prev = datetime.utcnow().timestamp() #initialise the time
                self.t = 0 #elapsed time
                time.sleep(0.1) #wait for approx a timestep before proceeding
               
                # path and tragectory are initialised
                self.initialise_pose = False
                self.generate_trajectory()

                # Initialise PF
                self.particles = self.Particles(self.N)
                self.initialise_particle_distribution(self.particles, centre=[-0., 0.], radius = np.sqrt(4*self.northings_std*self.eastings_std), heading = self.est_pose_yaw_rad)
                self.particles.gamma = [0]*self.particles.N
           
       

        if self.initialise_pose != True:  

            #determine the time step
            t_now = datetime.utcnow().timestamp()        
               
            dt = t_now - self.t_prev #timestep from last estimate
            self.t += dt #add to the elapsed time
            self.t_prev = t_now #update the previous timestep for the next loop

            ################################################################################ PF Implementation ################################################################################
           
            self.discrete_motion_model(self.particles, self.gamma, u, dt, [self.g_std**2, self.x_dot_std**2, self.g_dot_std**2])


            # pack into our measurement class
            self.measurement = self.Measurement()
            self.measurement.timestamp = self.measured_pose_timestamp_s
            self.measurement.northings = self.measured_pose_northings_m
            self.measurement.eastings = self.measured_pose_eastings_m
            self.measurement.northings_std = self.northings_std
            self.measurement.eastings_std = self.eastings_std  

            self.particles = self.pf_update(self.particles, self.measurement)
            self.pf_resample(self.particles, np.sqrt(self.measurement.northings_std * self.measurement.eastings_std), verbose = False)
           
            sigma_resolution = np.sqrt(self.northings_std * self.eastings_std)
            sampling_resolution = 100

            self.est_pose_northings_m, self.est_pose_eastings_m = kde_probability(self.particles, sigma_resolution, sampling_resolution)
            self.est_pose_yaw_rad = wrapped_mean(self.particles.gamma)
             ################################################################################ Trajectory Control ################################################################################

            # take current pose estimate and update by twist
            p_robot = Vector(3)
            p_robot[0] = self.est_pose_northings_m
            p_robot[1] = self.est_pose_eastings_m
            p_robot[2] = self.est_pose_yaw_rad
                               
            # p_robot = rigid_body_kinematics(p_robot, u, dt)
            # p_robot[2] = p_robot[2] % (2 * np.pi)  # deal with angle wrapping      

            ####################################### Trajectory sample #################################    
            # feedforward control: check wp progress and sample reference trajectory
            self.path.wp_progress(self.t, p_robot, self.accept_radius) # fill turning radius
            p_ref, u_ref = self.path.p_u_sample(self.t) #sample the path at the current elapsetime (i.e., seconds from start of motion modelling)  

            # feedback control: get pose change to desired trajectory from body
            dp = p_ref - p_robot #compute difference between reference and estimated pose in the $e$-frame
            dp[2] = (dp[2] + np.pi) % (2 * np.pi) - np.pi # handle angle wrapping for yaw
            H_eb = HomogeneousTransformation(p_robot[0:2],p_robot[2])
            ds=Inverse(H_eb.H_R)@dp # rotate the $e$-frame difference to get it in the $b$-frame (Hint: dp_b = H_be.H_R @ dp_e)

            # compute control gains for the initial condition (where the robot is stationalry)
            self.k_s = 1 / self.tau_s #ks
            if self.initialise_control == True:
                self.k_n = 2 * u_ref[0] / (self.L**2) #kn
                self.k_g = u_ref[0] / self.L #kg
                self.initialise_control = False # maths changes a bit after the first iteration

            # update the controls
            du = feedback_control(ds, self.k_s, self.k_n, self.k_g)

            # total control
            u = u_ref + du # combine feedback and feedforward control twist components

            # update control gains for the next timestep
            self.k_n = 2 * u[0] / (self.L**2) #kn
            self.k_g = u[0] / self.L #kg

            # ensure within performance limitation
            if u[1]>self.w_max: u[1]=self.w_max
            if u[1]<-self.w_max: u[1]=-self.w_max
            if u[0]>self.v_max: u[0]=self.v_max
            if u[0]<-self.v_max: u[0]=-self.v_max

           

            # actuator commands                
            q = self.ddrive.inv_kinematics(u)            

            wheel_speed_msg = Vector3Stamped()
            wheel_speed_msg.vector.x = q[0,0] # Right wheelspeed rad/s
            wheel_speed_msg.vector.y = q[1,0] # Left wheelspeed rad/s

            # update for show_laptop.py            
            # self.est_pose_northings_m = float(p_robot[0])
            # self.est_pose_eastings_m = float(p_robot[1])
            # self.est_pose_yaw_rad = float(p_robot[2])

           
            msg = self.pose_parse([datetime.utcnow().timestamp(),self.est_pose_northings_m,self.est_pose_eastings_m,0,0,0,self.est_pose_yaw_rad])
            self.datalog.log(msg, topic_name="/est_pose")


           
       

            #



        # > Think < #
        ############################################S####################################
        # #  TODO: Implement your state estimation
        # self.est_pose_northings_m = 1 # modify with your estimates
        # self.est_pose_eastings_m = 1 # modify with your estimates
        # self.est_pose_yaw_rad = 0 # modify with your estimates
       
        # msg = self.pose_parse([datetime.utcnow().timestamp(),self.est_pose_northings_m,self.est_pose_eastings_m,0,0,0,self.est_pose_yaw_rad])
        # self.datalog.log(msg, topic_name="/est_pose")
        # ################################################################################
        #  TODO: Implement your controller here                                        #

        # wheel_speed_msg = Vector3Stamped()
        # wheel_speed_msg.vector.x = 1 * np.pi  # Right wheel 1 rev/s = 1*pi rad/s
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