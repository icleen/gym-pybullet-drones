import numpy as np

from gymnasium import spaces

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class TargetAviary(BaseRLAviary):
    """Multi-agent RL problem: leader-follower."""

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=1,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 action_steps: int = 1,
                 action_obs: bool = False,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM,
                 use_residual: bool = False,
                 ):
        """Initialization of a multi-agent RL environment.

        Using the generic multi-agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        num_drones : int, optional
            The desired number of drones in the aviary.
        neighbourhood_radius : float, optional
            Radius used to compute the drones' adjacency matrix, in meters.
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        action_steps: int, optional
            The number of steps to take when given an action [default: 1]
        action_obs: bool, optional
            Whether to include the base actions in the observation (only valid if use_residual is True) [default: False]
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)
        use_residual: bool, optional
            Whether to use a PID controller to get base actions with the 
            given actions added

        """
        act = ActionType('pid')
        num_drones = 1
        self.EPISODE_LEN_SEC = 5
        self.max_steps = 10
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         action_steps=action_steps,
                         action_obs=action_obs,
                         gui=gui,
                         record=record, 
                         obs=obs,
                         act=act,
                         use_residual=use_residual,
                         )
        
        print('initial pose:', self.INIT_XYZS)

        self.target_pose_threshold = 0.15
        self.reward_dist_threshold = 0.15
        self._target_reset()

        self.temp_cnt = 0
        self.temp_max = 10000

    ################################################################################
    
    def step(self, action):
        """
        This is currently only setup for one drone so will break with more than one
        """
        tot_rew = 0
        action_threshold = 0.05
        # if self.use_residual:
        #     action = action
        maxact = np.max(np.abs(action), 1).squeeze()
        stepsize = min(action_threshold, maxact)
        num_steps = self.max_steps
        if maxact != 0:
            num_steps = int(np.ceil(maxact / stepsize))
            if self.action_obs:
                num_steps = min(num_steps, self.max_steps)
            stepsize = maxact / num_steps
        action_poses = np.zeros((num_steps, 3))
        if maxact != 0:
            pts = np.arange(num_steps) + 1
            action_steps = action / num_steps
            action_poses[:, :] = (action_steps.reshape(-1, 1) @ pts.reshape(1, -1)).T
        self.action_buffer.append(action)
        for ai in range(num_steps):
            rpm = np.zeros((self.NUM_DRONES, 4))
            for k in range(self.NUM_DRONES):
                rpm[k, :] = self.compute_control(k, action_poses[ai, :])
                rpm[k, :] = np.clip(rpm[k, :], 0, self.MAX_RPM)
            # self.action_buffer.append(action_poses[ai, :])
            obs, reward, terminated, truncated, info = super(BaseRLAviary, self).step(rpm)
            if ai == 0:
                tot_rew = reward * 0
            tot_rew += reward
            if terminated or truncated:
                break
        # terminated = True
        # truncated = True
        return obs, tot_rew, terminated, truncated, info

    def _preprocessAction(self,
                          action
                          ):
        """Pre-processes the action passed to `.step()` into motors' RPMs.

        Parameter `action` is processed differenly for each of the different
        action types: the input to n-th drone, `action[n]` can be of length
        1, 3, or 4, and represent RPMs, desired thrust and torques, or the next
        target position to reach using PID control.

        Parameter `action` is processed differenly for each of the different
        action types: `action` can be of length 1, 3, or 4 and represent 
        RPMs, desired thrust and torques, the next target position to reach 
        using PID control, a desired velocity vector, etc.

        Parameters
        ----------
        action : ndarray
            The input action for each drone in RPMs.

        Returns
        -------
        ndarray
            (NUM_DRONES, 4)-shaped array of ints containing to clipped RPMs
            commanded to the 4 motors of each drone.

        """
        return action
    
    ################################################################################

    def compute_control(self, k, act_k=None):
        """Pre-processes the action passed to `.step()` into motors' RPMs.

        Parameter `action` is processed differenly for each of the different
        action types: the input to n-th drone, `action[n]` can be of length
        1, 3, or 4, and represent RPMs, desired thrust and torques, or the next
        target position to reach using PID control.

        Parameter `action` is processed differenly for each of the different
        action types: `action` can be of length 1, 3, or 4 and represent 
        RPMs, desired thrust and torques, the next target position to reach 
        using PID control, a desired velocity vector, etc.

        Parameters
        ----------
        k : int
            The index for which drone to get controls for.
        act_k : ndarray
            The array of the target location for the PID conroller to move to.

        Returns
        -------
        ndarray
            (4,)-shaped array of ints containing to clipped RPMs
            commanded to the 4 motors of the drone.

        """
        state_k = self._getDroneStateVector(k)
        if k == 0:
            target = self.target_poses[self.target_idx]
            if np.linalg.norm(state_k[:3] - target[:3]) < 0.1:
                self.target_idx += 1
                self.target_idx = self.target_idx % len(self.target_poses)
            target = self.target_poses[self.target_idx]
        if act_k is not None:
            if self.use_residual:
                target += act_k
            else:
                target = act_k
                target[:3] += state_k[:3]
                target[2] = max(self.INIT_XYZS[0, 2], target[2])
        # print(target)
        # import pdb; pdb.set_trace()
        next_pos = self._calculateNextStep(
            current_position=state_k[0:3],
            destination=target,
            step_size=1,
        )
        rpm_k, _, _ = self.ctrl[k].computeControl(
            control_timestep=self.CTRL_TIMESTEP,
            cur_pos=state_k[0:3],
            cur_quat=state_k[3:7],
            cur_vel=state_k[10:13],
            cur_ang_vel=state_k[13:16],
            target_pos=next_pos
        )
        return rpm_k

    ################################################################################

    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        if (self.reward_accomp == 1).all():
            return 0
        if self._computeDroneFail():
            return -10000
        state = self._getDroneStateVector(0)
        ridx = int(np.sum(self.reward_accomp))
        dist = np.linalg.norm(state[0:3] - self.reward_poses[ridx])**2
        return -dist
        if dist < self.reward_dist_threshold:
            self.reward_accomp[ridx] += 1
        reward = (self.reward_accomp - 1) * 2
        reward = reward.sum() - dist
        return reward

    ################################################################################
    
    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        term = (self.reward_accomp == 1).all()
        if self.print_fail_reasons and term:
            print('terminated')
        return term

    ################################################################################
    
    def _computeTruncated(self):
        """Computes the current truncated value.

        Returns
        -------
        bool
            Whether the current episode timed out.

        """
        if self._computeDroneFail():
            if self.print_fail_reasons:
                print('drone fail')
            return True
        if self._computeDroneTooFar():
            if self.print_fail_reasons:
                print('drone too far')
            return True
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            if self.print_fail_reasons:
                print('episode length')
            return True
        else:
            return False
        
    def _computeDroneFail(self):
        if self.step_counter < 110:
            return False
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        for i in range(self.NUM_DRONES):
            if (abs(states[i][7]) > .6 or abs(states[i][8]) > .6 # Truncate when a drone is too tilted
                or states[i][2] < 0.02
            ):
                if self.print_fail_reasons:
                    print(states[i][:9])
                    print(self.step_counter)
                return True
        return False
    
    def _computeDroneTooFar(self):
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        maxdist = 10
        for i in range(self.NUM_DRONES):
            # if (abs(states[i][0]) > 2.0 or abs(states[i][1]) > 2.0 or states[i][2] > 2.0 # Truncate when a drones is too far away
            #  or abs(states[i][7]) > .4 or abs(states[i][8]) > .4 # Truncate when a drone is too tilted
            # ):
            if (abs(states[i][0]) > maxdist or abs(states[i][1]) > maxdist or states[i][2] > maxdist or abs(states[i][0]) < -maxdist or abs(states[i][1]) < -maxdist or states[i][2] < -maxdist # Truncate when a drones is too far away
            ):
                return True
        return False

    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {
            'target': self.target_poses[self.target_idx], 
            'reward_pose': self.reward_poses[int(np.sum(self.reward_accomp))],
            'drone_pose': self._getDroneStateVector(0),
        }

    def reset(self,
              seed : int = None,
              options : dict = None):
        """Resets the environment.

        Parameters
        ----------
        seed : int, optional
            Random seed.
        options : dict[..], optional
            Additinonal options, unused

        Returns
        -------
        ndarray | dict[..]
            The initial observation, check the specific implementation of `_computeObs()`
            in each subclass for its format.
        dict[..]
            Additional information as a dictionary, check the specific implementation of `_computeInfo()`
            in each subclass for its format.

        """
        self._target_reset()

        return super().reset(seed, options)
    
    def _target_reset(self):
        self.target_idx = 0

        self.target_pose = np.array([1.5, 1, self.INIT_XYZS[0, 2] + 0.1])
        target_diff = self.target_pose - self.INIT_XYZS[0, :3]
        target_dist = np.linalg.norm(target_diff)
        # TODO: randomize the target pose

        if True:
            R = .3
            PERIOD = 10
            NUM_WP = self.CTRL_FREQ * PERIOD
            self.target_poses = np.zeros((NUM_WP, 3))
            for i in range(NUM_WP):
                self.target_poses[i, :] = R*np.cos((i/NUM_WP)*(2*np.pi)+np.pi/2)+self.INIT_XYZS[0, 0], R*np.sin((i/NUM_WP)*(2*np.pi)+np.pi/2)-R+self.INIT_XYZS[0, 1], self.INIT_XYZS[0, 2]
        else:
            PERIOD = 10
            NUM_WP = self.CTRL_FREQ * PERIOD
            self.target_poses = np.zeros((NUM_WP, 3))
            pts = np.linspace(0, 1, NUM_WP)
            ys = np.sin(pts * np.pi) * target_dist
            xs = pts * target_dist
            # xs = pts * target_diff[0]
            # ys = pts * target_diff[1]
            zs = pts * target_diff[2] + self.INIT_XYZS[0, 2]
            xyz = np.stack((xs, ys, zs), 1)

            base_axis = self.INIT_XYZS[0, :3].copy()
            base_axis = np.array([1, 0, 0])
            costheta = np.dot(target_diff, base_axis) / target_dist
            sintheta = np.linalg.norm(np.cross(target_diff, base_axis)) / target_dist
            rot = np.array([
                [costheta, -sintheta],
                [sintheta, costheta]
            ])
            xyz[:, :2] = xyz[:, :2] @ rot.T
            xyz[:, 0] += self.INIT_XYZS[0, 0]
            xyz[:, 1] += self.INIT_XYZS[0, 1]
            self.target_poses = xyz
            # print(xyz[-1])
            # import pdb; pdb.set_trace()

        self.TARGET_POS = self.target_poses[self.target_idx]

        self.reward_poses = np.array([self.target_pose])
        self.reward_accomp = np.zeros(len(self.reward_poses))

    def _observationSpace(self):
        """Returns the observation space of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (NUM_DRONES,H,W,4) or (NUM_DRONES,12) depending on the observation type.

        """
        if self.OBS_TYPE == ObservationType.RGB:
            return spaces.Box(low=0,
                              high=255,
                              shape=(self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0], 4), dtype=np.uint8)
        elif self.OBS_TYPE == ObservationType.KIN:
            ############################################################
            #### OBS SPACE OF SIZE 12
            #### Observation vector ### X        Y        Z       Q1   Q2   Q3   Q4   R       P       Y       VX       VY       VZ       WX       WY       WZ       TX        TY        TZ       TQ1   TQ2   TQ3   TQ4
            lo = -np.inf
            hi = np.inf
            obs_lower_bound = np.array([[lo,lo,0, lo,lo,lo,lo,lo,lo,lo,lo,lo, lo,lo,0] for i in range(self.NUM_DRONES)])
            obs_upper_bound = np.array([[hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,hi,hi, hi,hi,hi] for i in range(self.NUM_DRONES)])
            #### Add targets to observation space ################
            # obs_lower_bound = np.hstack([obs_lower_bound, np.array([[lo,lo,0, lo,lo,lo,lo]])])
            # obs_upper_bound = np.hstack([obs_upper_bound, np.array([[hi,hi,hi,hi,hi,hi,hi]])])
            # obs_lower_bound = np.hstack([obs_lower_bound, np.array([[lo,lo,0]])])
            # obs_upper_bound = np.hstack([obs_upper_bound, np.array([[hi,hi,hi]])])
            #### Add action buffer to observation space ################
            act_lo = -1
            act_hi = +1
            for i in range(self.ACTION_BUFFER_SIZE):
                obs_lower_bound = np.hstack([obs_lower_bound, np.array([[act_lo,act_lo,act_lo] for i in range(self.NUM_DRONES)])])
                obs_upper_bound = np.hstack([obs_upper_bound, np.array([[act_hi,act_hi,act_hi] for i in range(self.NUM_DRONES)])])
            if self.action_obs:
                for ai in range(self.max_steps):
                    obs_lower_bound = np.hstack([obs_lower_bound, np.array([[act_lo,act_lo,act_lo] for i in range(self.NUM_DRONES)])])
                    obs_upper_bound = np.hstack([obs_upper_bound, np.array([[act_hi,act_hi,act_hi] for i in range(self.NUM_DRONES)])])
            return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)
            ############################################################
        else:
            print("[ERROR] in BaseRLAviary._observationSpace()")
    
    ################################################################################

    def _computeObs(self):
        """Returns the current observation of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (NUM_DRONES,H,W,4) or (NUM_DRONES,12) depending on the observation type.

        """
        comped_obs = None
        if self.OBS_TYPE == ObservationType.RGB:
            if self.step_counter%self.IMG_CAPTURE_FREQ == 0:
                for i in range(self.NUM_DRONES):
                    self.rgb[i], self.dep[i], self.seg[i] = self._getDroneImages(i,
                                                                                 segmentation=False
                                                                                 )
                    #### Printing observation to PNG frames example ############
                    if self.RECORD:
                        self._exportImage(img_type=ImageType.RGB,
                                          img_input=self.rgb[i],
                                          path=self.ONBOARD_IMG_PATH+"drone_"+str(i),
                                          frame_num=int(self.step_counter/self.IMG_CAPTURE_FREQ)
                                          )
            comped_obs = np.array([self.rgb[i] for i in range(self.NUM_DRONES)]).astype('float32')
        elif self.OBS_TYPE == ObservationType.KIN:
            ############################################################
            #### OBS SPACE OF SIZE 12
            #### Add targets to observation space ################
            obs_12 = np.zeros((self.NUM_DRONES,15))
            for i in range(self.NUM_DRONES):
                #obs = self._clipAndNormalizeState(self._getDroneStateVector(i))
                obs = self._getDroneStateVector(i)
                # obs_12[i, :] = np.hstack([obs[0:3], obs[7:10], obs[10:13], obs[13:16]]).reshape(12,)
                obs_12[i, :] = np.hstack([obs[0:3], obs[7:10], obs[10:13], obs[13:16], self.reward_poses[int(np.sum(self.reward_accomp))]]).reshape(15,)
            ret = np.array([obs_12[i, :] for i in range(self.NUM_DRONES)]).astype('float32')
            # #### Add targets to observation space ################
            # import pdb; pdb.set_trace()
            # ret = np.concatenate(ret, self.reward_poses[int(np.sum(self.reward_accomp))], 0)
            #### Add action buffer to observation #######################
            for i in range(self.ACTION_BUFFER_SIZE):
                ret = np.hstack([ret, np.array([self.action_buffer[i][j, :] for j in range(self.NUM_DRONES)])])
            if self.action_obs:
                act_obs = np.zeros((self.max_steps, 3))
                max_idx = min(
                    len(self.target_poses), self.target_idx + self.max_steps
                )
                act_obs[:max_idx - self.target_idx] = self.target_poses[self.target_idx:max_idx]
                if self.target_idx + self.max_steps > max_idx:
                    act_obs[max_idx - self.target_idx:] += self.target_poses[-1]
                ret = np.hstack((ret, act_obs.reshape(1, -1)))
            comped_obs = ret
            ############################################################
        else:
            print("[ERROR] in BaseRLAviary._computeObs()")

        self.comped_obs = comped_obs
        return comped_obs        
