import time
import multiprocessing
import numpy as np
from avp_stream import VisionProStreamer
import scipy.spatial.transform
from typing import Tuple
from dataclasses import dataclass


class AVPExpert:
    """
    This class provides an interface to the Apple Vision Pro to teleoperate a robotic hand.
    """

    def __init__(self, avp_ip: str = None):
        
        AVP_IP = avp_ip or "10.93.181.127"
        PINCH_THRESHOLD = 0.02
        BUFFER_TIME = 0.06
        STREAM_PERIOD = 0.001
        
        # Manager to handle shared state between processes
        self.manager = multiprocessing.Manager()
        self.latest_data = self.manager.dict()
        self.latest_data["action"] = [0.0] * 6
        self.latest_data["grasping"] = [0, 0, 0, 0]
        
        # This controls whether the user want to intervene
        # When left finger pinches, the user can control the robot
        self.latest_data["is_intervening"] = False
        
        buffer_count = int(BUFFER_TIME / STREAM_PERIOD)
        assert buffer_count > 0, "Buffer time must be greater than the stream period."
        
        self.left_pinching_buffer = self.manager
        
        
        # This should mitigate noie 
        self.left_pinching_buffer = self.manager.list()
   
        # Start a process to continuously read from the AVP
        self.process = multiprocessing.Process(
            target=self._read_avp,
            args=(AVP_IP, buffer_count, STREAM_PERIOD, PINCH_THRESHOLD)
        )
        
        self.process.daemon = True
        self.process.start()

    def _read_avp(self, ip: str, buffer_count: float, stream_period: float, pinch_threshold: float):
        
        # If this buffer is full, we know that the person pinched wtih the left hand
        # and we can set the flag to intervene
        pinch_left_buffer = 0
        
        stream = VisionProStreamer(ip = ip, record = True)
        
        while stream.latest is None: 
            pass 

        while stream:
            data = stream.latest
            action = [0.0] * 6
            
            state_right = data["right_fingers"]  # shape (25,4,4)
            state_right = state_right[0]  # shape (4,4)
            pinch_right = data["right_pinch_distance"] # float
            pinch_left = data["left_pinch_distance"] # float
                        
            right_wrist = data['right_wrist'][0]
                        
            # Check if the user is pinching
            if pinch_right < pinch_threshold:
                pinching_right = True
            else:
                pinching_right = False
                
            if pinch_left < pinch_threshold:
                pinch_left_buffer += 1
            else:
                pinch_left_buffer -= 1
                
            if pinch_left_buffer >= buffer_count:
                pinch_left_buffer = buffer_count
            if pinch_left_buffer <= 0:
                pinch_left_buffer = 0
                
            if pinch_left_buffer >= 1:
                self.latest_data["is_intervening"] = True
            else:
                self.latest_data["is_intervening"] = False
                
            # if pinch_left_buffer >= buffer_count:
            #     self.latest_data["is_intervening"] = not self.latest_data["is_intervening"]
            #     pinch_left_buffer = 0
                
            # Extract translation (x, y, z) from the matrix
            translation = right_wrist[:3, 3]  # Last column of the matrix (ignoring the 4th row)
            # Extract rotation (roll, pitch, yaw) from the matrix
            rotation = scipy.spatial.transform.Rotation.from_matrix(right_wrist[:3, :3]).as_euler('xyz', degrees=False)
            
            action = [
                -translation[0], -translation[1], translation[2],  # y, x, z
                -rotation[0], -rotation[1], -rotation[2]          # roll, pitch, yaw
            ]
            # Update the shared state
            self.latest_data["action"] = action
            self.latest_data["grasping"] = pinching_right
            
            time.sleep(stream_period)

    def get_action(self) -> Tuple[np.ndarray, list]:
        """Returns the latest action and pinch distance of the AVP."""
        action = self.latest_data["action"]
        grasping = self.latest_data["grasping"]
        return np.array(action), grasping
    
    def is_intervening(self) -> bool:
        return self.latest_data["is_intervening"]
    
    def close(self):
        self.process.terminate()