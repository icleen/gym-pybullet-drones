import numpy as np

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class ChaserAviary(BaseRLAviary):
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
        num_drones = 2
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
        self.target_idx = 0
        self.target_poses = np.array([
            [1, 1, 1],
            [1, 2, 0.5],
            [2, 2, 1.5],
            [3, 3, 0.5],
            [4, 3, 1],
            [4, 4, 1],
            [4, 5, 1],
            [4, 6, 1],
            [4, 7, 1.2],
            [5, 7, 1.5],
            [6, 7, 1.3],
            [7, 7, 1],
        ])
        # self.target_poses = [np.arange(tpose[i-1], tpose[i], step_size) for i in range(1, len(self.target_poses))]
        R = .5
        PERIOD = 10
        DEFAULT_CONTROL_FREQ_HZ = 12
        NUM_WP = DEFAULT_CONTROL_FREQ_HZ * PERIOD
        self.target_poses = np.zeros((NUM_WP, 3))
        for i in range(NUM_WP):
            self.target_poses[i, 0] = R * np.cos((i / NUM_WP) * (2 * np.pi) + np.pi / 2) + self.INIT_XYZS[0, 0]
            self.target_poses[i, 1] = R * np.sin((i / NUM_WP) * (2 * np.pi) + np.pi/2) - R + self.INIT_XYZS[0, 1]
            self.target_poses[i, 2] = self.INIT_XYZS[0, 2] + 0.5
            # self.target_poses[i, 2] += i / 10 if i < 10 else 1
        self.TARGET_POS = self.target_poses[self.target_idx]
        self.target_pose_threshold = .15

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
            if k == 0:
                rpm[k, :] = self.compute_control(k)
            else:
                continue
                act_k = action[k - 1, :]
                target = self.HOVER_RPM * 0.05 * act_k
                if self.use_residual:
                    rpm[k, :] = self.compute_control(k)
                    rpm[k, :] = np.clip(rpm[k, :], 0, self.MAX_RPM)
                if self.ACT_TYPE == ActionType.RPM:
                    rpm[k, :] += target
                elif self.ACT_TYPE == ActionType.PID:
                    rpm_k = self.compute_control(k, act_k)
                    rpm[k, :] += rpm_k
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
            target = self.TARGET_POS
        else:
            target = self._getDroneStateVector(0)[0:3]
            if act_k is not None:
                target += act_k
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
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        ret = -np.linalg.norm(states[0, :3] - states[1, :3])**2
        return ret

    ################################################################################
    
    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        if np.linalg.norm(states[0, :3] - self.TARGET_POS[:3]) < 0.1:
            self.target_idx += 1
            self.target_idx = self.target_idx % len(self.target_poses)
            self.TARGET_POS = self.target_poses[self.target_idx]
        dist = np.linalg.norm(states[0, :3] - states[1, :3])
        if dist < self.target_pose_threshold:
            return True
        else:
            return False

    ################################################################################
    
    def _computeTruncated(self):
        """Computes the current truncated value.

        Returns
        -------
        bool
            Whether the current episode timed out.

        """
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        for i in range(self.NUM_DRONES):
            # if (abs(states[i][0]) > 2.0 or abs(states[i][1]) > 2.0 or states[i][2] > 2.0 # Truncate when a drones is too far away
            #  or abs(states[i][7]) > .4 or abs(states[i][8]) > .4 # Truncate when a drone is too tilted
            # ):
            if (abs(states[i][0]) > 50. or abs(states[i][1]) > 50. or states[i][2] > 50. # Truncate when a drones is too far away
             or abs(states[i][7]) > .4 or abs(states[i][8]) > .4 # Truncate when a drone is too tilted
            ):
                return True
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
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
