import time
import multiprocessing
import numpy as np
from avp_stream import VisionProStreamer
import scipy.spatial.transform
from typing import Tuple
from dataclasses import dataclass
from enum import Enum

class AVPExpert:
    """
    This class provides an interface to the Apple Vision Pro to teleoperate a robotic hand.
    """

    def __init__(self, avp_ip: str = "10.93.181.127"):
        self.avp_ip = avp_ip 
        
  
        # Manager to handle shared state between processes
        self.manager = multiprocessing.Manager()
        self.latest_data = self.manager.dict()
        self.latest_data["action"] = [0.0] * 6
        self.latest_data["pinching"] = [0, 0, 0, 0]
             
        # This controls whether the user want to intervene
        # When left finger pinches, the user can control the robot
        self.latest_data["pinching_left"] = False
   
        # Start a process to continuously read from the AVP
        self.process = multiprocessing.Process(target=self._read_avp)
        self.process.daemon = True
        self.process.start()

    def _read_avp(self):
        self.stream = VisionProStreamer(ip = self.avp_ip, record = True)

        while True:
            data = self.stream.latest
            action = [0.0] * 6
            
            state_right = data["right_fingers"]  # shape (25,4,4)
            state_right = state_right[0]  # shape (4,4)
            pinch_right = data["right_pinch_distance"] # float
            pinch_left = data["left_pinch_distance"] # float
                        
            right_wrist = data['right_wrist'][0]
                        
            # Check if the user is pinching
            if pinch_right < 0.01:
                pinching_right = True
            else:
                pinching_right = False
                
            if pinch_left < 0.01:
                self.latest_data["pinching_left"] = True
            else:
                self.latest_data["pinching_left"] = False
                
                
            # Extract translation (x, y, z) from the matrix
            translation = right_wrist[:3, 3]  # Last column of the matrix (ignoring the 4th row)
            # Extract rotation (roll, pitch, yaw) from the matrix
            rotation = scipy.spatial.transform.Rotation.from_matrix(right_wrist[:3, :3]).as_euler('xyz', degrees=False)
            
            action = [
                -translation[1], translation[0], translation[2],  # y, x, z
                -rotation[0], -rotation[1], -rotation[2]          # roll, pitch, yaw
            ]
            # Update the shared state
            self.latest_data["action"] = action
            self.latest_data["pinching"] = pinching_right

    def get_action(self) -> Tuple[np.ndarray, list]:
        """Returns the latest action and pinch distance of the AVP."""
        action = self.latest_data["action"]
        pinching = self.latest_data["pinching"]
        return np.array(action), pinching
    
    def is_intervening(self) -> bool:
        return self.latest_data["pinching_left"]
    
    def close(self):
        self.process.terminate()