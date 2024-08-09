import numpy as np

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class SinAviary(BaseRLAviary):
    """Multi-agent RL problem: leader-follower."""

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=2,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
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
        num_drones = 1
        self.EPISODE_LEN_SEC = 20
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
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
            The input action for each drone, to be translated into RPMs.

        Returns
        -------
        ndarray
            (NUM_DRONES, 4)-shaped array of ints containing to clipped RPMs
            commanded to the 4 motors of each drone.

        """
        self.action_buffer.append(action)
        rpm = np.zeros((self.NUM_DRONES, 4))
        for k in range(self.NUM_DRONES):
            act_k = action[k, :]
            #if self.temp_cnt < self.temp_max:
            #    self.temp_cnt += 1
            #    act_k *= 0
            target = self.HOVER_RPM * 0.2 * act_k
            if self.use_residual:
                rpm[k, :] = self.compute_control(k)
                # rpm[k, :] = np.clip(rpm[k, :], 0, self.MAX_RPM)
            if self.ACT_TYPE == ActionType.RPM:
                # import pdb; pdb.set_trace()
                rpm[k, :] += target
            elif self.ACT_TYPE == ActionType.PID:
                rpm[k, :] = self.compute_control(k, act_k * 0.25)
            elif self.ACT_TYPE == ActionType.VEL:
                state = self._getDroneStateVector(k)
                if np.linalg.norm(act_k[0:3]) != 0:
                    v_unit_vector = act_k[0:3] / np.linalg.norm(act_k[0:3])
                else:
                    v_unit_vector = np.zeros(3)
                temp, _, _ = self.ctrl[k].computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                        cur_pos=state[0:3],
                                                        cur_quat=state[3:7],
                                                        cur_vel=state[10:13],
                                                        cur_ang_vel=state[13:16],
                                                        target_pos=state[0:3], # same as the current position
                                                        target_rpy=np.array([0,0,state[9]]), # keep current yaw
                                                        target_vel=self.SPEED_LIMIT * np.abs(act_k[3]) * v_unit_vector # target the desired velocity vector
                                                        )
                rpm[k, :] += temp
            elif self.ACT_TYPE == ActionType.ONE_D_RPM:
                rpm[k, :] += np.repeat(target, 4)
            elif self.ACT_TYPE == ActionType.ONE_D_PID:
                state = self._getDroneStateVector(k)
                res, _, _ = self.ctrl[k].computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                        cur_pos=state[0:3],
                                                        cur_quat=state[3:7],
                                                        cur_vel=state[10:13],
                                                        cur_ang_vel=state[13:16],
                                                        target_pos=state[0:3]+0.1*np.array([0,0,act_k[0]])
                                                        )
                rpm[k, :] += res
            else:
                print("[ERROR] in BaseRLAviary._preprocessAction()")
                exit()
            rpm[k, :] = np.clip(rpm[k, :], 0, self.MAX_RPM)

        
        return rpm
    
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
        return (self.reward_accomp == 1).all()

    ################################################################################
    
    def _computeTruncated(self):
        """Computes the current truncated value.

        Returns
        -------
        bool
            Whether the current episode timed out.

        """
        if self._computeDroneFail():
            return True
        if self.step_counter / self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False
        
    def _computeDroneFail(self):
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        for i in range(self.NUM_DRONES):
            # if (abs(states[i][0]) > 2.0 or abs(states[i][1]) > 2.0 or states[i][2] > 2.0 # Truncate when a drones is too far away
            #  or abs(states[i][7]) > .4 or abs(states[i][8]) > .4 # Truncate when a drone is too tilted
            # ):
            if (abs(states[i][0]) > 50. or abs(states[i][1]) > 50. or states[i][2] > 50. # Truncate when a drones is too far away
             or abs(states[i][7]) > .4 or abs(states[i][8]) > .4 # Truncate when a drone is too tilted
            ):
                # import pdb; pdb.set_trace()
                return True
            # if self.step_counter > 10 and states[i][2] < 0.1:
            #     return True
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
        return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years

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

        PERIOD = 10
        DEFAULT_CONTROL_FREQ_HZ = 12
        NUM_WP = DEFAULT_CONTROL_FREQ_HZ * PERIOD
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

        
