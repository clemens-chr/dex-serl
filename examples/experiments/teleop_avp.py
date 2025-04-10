# tele_avp.py

import requests
import numpy as np
import time
from scipy.spatial.transform import Rotation # Import Rotation

# --- Use the actual AVP Streamer ---
try:
    from avp_stream import VisionProStreamer
except ImportError:
    print("ERROR: Failed to import VisionProStreamer. Make sure avp_stream package is installed.")
    exit(1)

# --- Configuration ---
# AVP Configuration (Using your provided details - corrected IP from previous version)
AVP_IP = "192.168.1.10"  # Use the IP you confirmed was working
RECORD_AVP_DATA = True    # Set based on your requirement

# Robot and Control Configuration
FRANKA_SERVER_URL = "http://127.0.0.2:5000" # From your franka_server flags
PINCH_THRESHOLD = 0.015
CONTROL_LOOP_HZ = 50 # Frequency to check AVP data and potentially send commands

# --- State Variables ---
pinch_active = False
reference_avp_pose = None     # Will store the 7D AVP pose [x,y,z,qx,qy,qz,qw] on pinch start
reference_franka_pose = None  # Will store the 7D Franka pose on pinch start

# --- Helper Functions ---
def get_franka_pose(server_url):
    """Gets the current pose [x,y,z,qx,qy,qz,qw] from the Franka server."""
    try:
        response = requests.post(f"{server_url}/getpos")
        response.raise_for_status() # Raise an exception for bad status codes
        data = response.json()
        pose = np.array(data['pose'])
        if pose.shape == (7,):
            return pose
        else:
            print(f"Error: Received Franka pose has unexpected shape: {pose.shape}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error getting Franka pose: {e}")
        return None
    except (KeyError, ValueError) as e: # Catch potential JSON errors
        print(f"Error processing Franka pose response: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during get_franka_pose: {e}")
        return None


def send_franka_pose(server_url, pose_command):
    """Sends a target pose [x,y,z,qx,qy,qz,qw] to the Franka server."""
    if pose_command is None or pose_command.shape != (7,):
         print(f"Error: Invalid pose command shape for sending: {pose_command.shape if pose_command is not None else 'None'}")
         return False
    try:
        payload = {"arr": pose_command.tolist()}
        # print(f"Debug: Sending payload: {payload}") # Uncomment for deep debug
        response = requests.post(f"{server_url}/pose", json=payload)
        response.raise_for_status()
        # print(f"Debug: Response status: {response.status_code}, text: {response.text}") # Uncomment for deep debug
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error sending Franka pose: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during send_franka_pose: {e}")
        return False

def get_7d_pose_from_avp_matrix(matrix: np.ndarray) -> np.ndarray | None:
    """Extracts [x,y,z,qx,qy,qz,qw] from a 4x4 AVP matrix."""
    matrix = np.squeeze(matrix) # Ensure it's a 2D array
    if matrix is None or matrix.shape != (4, 4):
        print(f"Error: Invalid AVP matrix shape for pose extraction: {matrix.shape if matrix is not None else 'None'}")
        return None
    try:
        position = matrix[:3, 3]
        rotation_matrix = matrix[:3, :3]
        if np.linalg.det(rotation_matrix) < 0.1: # Check determinant is close to 1
             print(f"Warning: Possibly invalid rotation matrix (det={np.linalg.det(rotation_matrix)}). Using identity orientation.")
             # Fallback to identity quaternion or handle as error
             orientation_quat = np.array([0.0, 0.0, 0.0, 1.0]) # w is last in scipy
        else:
             orientation_quat = Rotation.from_matrix(rotation_matrix).as_quat() # [x,y,z,w]
        return np.concatenate([position, orientation_quat])

    except Exception as e:
        print(f"Error converting AVP matrix to 7D pose: {e}")
        return None


# --- Main Control Loop ---
if __name__ == "__main__":
    print("Initializing AVP Streamer...")
    avp_streamer = None
    try:
        avp_streamer = VisionProStreamer(ip=AVP_IP, record=RECORD_AVP_DATA)
        print(f"AVP Streamer initialized for IP {AVP_IP}. Waiting for first data...")
        time.sleep(2) # Give it time to connect and get first data
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to initialize AVP Streamer: {e}")
        exit(1)

    print("Starting Teleoperation Loop...")
    print(f"Connecting to Franka Server at: {FRANKA_SERVER_URL}")
    print(f"Control active when Left Pinch Distance <= {PINCH_THRESHOLD}")

    last_loop_time = time.time()

    try:
        while True:
            current_time = time.time()
            if current_time - last_loop_time < (1.0 / CONTROL_LOOP_HZ):
                time.sleep(0.001) # Sleep briefly if looping too fast
                continue
            last_loop_time = current_time

            # 1. Get latest AVP data
            try:
                avp_data = avp_streamer.latest
                if not avp_data or "right_wrist" not in avp_data or "left_pinch_distance" not in avp_data:
                    print("\rWaiting for valid AVP data...", end="")
                    time.sleep(0.1)
                    continue

                current_avp_matrix = avp_data["right_wrist"]
                left_pinch = avp_data["left_pinch_distance"]
                
                print(f"\rLeft Pinch Distance: {left_pinch:.4f}", end="") # For debugging

                # Ensure matrix is a numpy array
                if not isinstance(current_avp_matrix, np.ndarray):
                    current_avp_matrix = np.array(current_avp_matrix)

                # Extract the full 7D pose from the AVP matrix
                current_avp_pose_7d = get_7d_pose_from_avp_matrix(current_avp_matrix)

                if current_avp_pose_7d is None:
                    print("Error: Could not get valid 7D pose from AVP matrix. Skipping frame.")
                    continue

            except Exception as e:
                print(f"\nError getting or processing data from AVP streamer: {e}")
                time.sleep(0.5)
                continue


            is_pinching_now = left_pinch <= PINCH_THRESHOLD and left_pinch > 0.0 # When it is not detected, it is 0.0

            # 2. Handle Pinch State Transitions and Control
            if is_pinching_now and not pinch_active:
                # --- Pinch Started ---
                print(f"\n--- Pinch START DETECTED (Dist: {left_pinch:.4f}) ---")
                current_franka_pose = get_franka_pose(FRANKA_SERVER_URL)

                # Need both AVP and Franka poses to establish reference
                if current_avp_pose_7d is not None and current_franka_pose is not None:
                    pinch_active = True
                    reference_avp_pose = current_avp_pose_7d.copy() # Store 7D AVP pose
                    reference_franka_pose = current_franka_pose.copy() # Store 7D Franka pose

                    print(f"  Reference AVP Pose (7D): {np.array2string(reference_avp_pose, precision=3)}")
                    print(f"  Reference Franka Pose (7D): {np.array2string(reference_franka_pose, precision=3)}")
                else:
                    print("  ERROR: Could not get valid AVP or Franka pose to start control.")


            elif is_pinching_now and pinch_active:
                # --- Pinch Continues ---
                if reference_avp_pose is None or reference_franka_pose is None:
                    print("\nWARN: Pinch active but reference points are missing. Resetting state.")
                    pinch_active = False # Reset state to force re-acquiring references
                    continue

                # --- Calculate Translation Delta ---
                current_avp_pos = current_avp_pose_7d[:3]
                reference_avp_pos = reference_avp_pose[:3]
                delta_translation = current_avp_pos - reference_avp_pos

                # --- Calculate Orientation Delta ---
                current_avp_quat = current_avp_pose_7d[3:] # qx,qy,qz,qw
                reference_avp_quat = reference_avp_pose[3:]

                # Use Scipy Rotation objects for robust calculation
                try:
                    r_ref_avp = Rotation.from_quat(reference_avp_quat)
                    r_curr_avp = Rotation.from_quat(current_avp_quat)

                    # Delta rotation: transforms reference AVP orientation to current AVP orientation
                    r_delta = r_curr_avp * r_ref_avp.inv()

                except ValueError as e:
                     print(f"\nError creating Rotation object from AVP quat: {e}. Using identity delta.")
                     r_delta = Rotation.identity()

                # --- Adjust for transformation offsets ---
                delta_x = delta_translation[0]
                delta_y = delta_translation[1]
                delta_z = delta_translation[2]
                delta_translation[0] = delta_y * 0.5 # Scale down for Franka control
                delta_translation[1] = -delta_x * 0.5 # Scale down for Franka control
                delta_translation[2] = delta_z * 0.5 # Scale down for Franka control
                target_franka_pos = reference_franka_pose[:3] + delta_translation
                

                reference_franka_quat = reference_franka_pose[3:]
                try:
                    r_ref_franka = Rotation.from_quat(reference_franka_quat)
                    
                    try:
                        d_roll_orig, d_pitch_orig, d_yaw_orig = r_delta.as_euler('xyz', degrees=False)
                    except ValueError:
                        print("Warning: Gimbal lock or invalid Euler sequence for r_delta? Setting deltas to 0.")
                        d_roll_orig, d_pitch_orig, d_yaw_orig = 0.0, 0.0, 0.0   
                                         
                    r_delta_franka_component = Rotation.from_euler('x', d_roll_orig)
                    
                    r_target_franka = r_delta_franka_component * r_ref_franka # Apply delta in Franka's base frame


                    # Apply the delta rotation calculated from AVP to the Franka reference orientation
                    # r_target_franka = r_delta * r_ref_franka

                    target_franka_quat = r_target_franka.as_quat() # [qx, qy, qz, qw]

                except ValueError as e:
                    print(f"\nError creating Rotation object from Franka quat: {e}. Keeping original orientation.")
                    target_franka_quat = reference_franka_quat

                # ONLY TRANSLATION FOR NOW
                # target_franka_quat = reference_franka_quat

                # --- Combine into the full pose command ---
                pose_command = np.concatenate([target_franka_pos, target_franka_quat])

                # --- DEBUGGING: Print deltas and command ---
                
                clear_line = "\033[K"
                green_start = "\033[92m"
                yellow_start = "\033[93m"
                color_end = "\033[0m"

                delta_rot_euler = r_delta.as_euler('xyz', degrees=True) # For easier visualization
                
                print(f"\r{clear_line}" # Start with carriage return and clear line
                      f"P:{left_pinch:.3f} | "
                      f"{green_start}" # Green for AVP delta
                      f"dT:[{delta_translation[0]:+.3f},{delta_translation[1]:+.3f},{delta_translation[2]:+.3f}] | "
                      f"dR:[{delta_rot_euler[0]:+.1f},{delta_rot_euler[1]:+.1f},{delta_rot_euler[2]:+.1f}]" # Euler, less precise
                      f"{color_end} | "
                      f"{yellow_start}" # Yellow for Franka command
                      f"CmdP:[{pose_command[0]:.3f},{pose_command[1]:.3f},{pose_command[2]:.3f}] | "
                      f"CmdQ:[{pose_command[3]:.2f},{pose_command[4]:.2f},{pose_command[5]:.2f},{pose_command[6]:.2f}]" # Quat, less precise
                      f"{color_end}",
                      end="") # Keep end=""
                
            
                success = send_franka_pose(FRANKA_SERVER_URL, pose_command)
                if not success:
                    print("\nERROR: Failed to send pose command to Franka Server!")
                    

            elif not is_pinching_now and pinch_active:
                # --- Pinch Ended ---
                print(f"\n--- Pinch END DETECTED (Dist: {left_pinch:.4f}) ---")
                pinch_active = False
                reference_avp_pose = None
                reference_franka_pose = None
                print("  Resetting references.")


            # else: # Not pinching now and wasn't pinching before
                # print("\rWaiting for pinch...", end="") # Can be noisy, enable if needed
                # pass # Do nothing if not pinching

    except KeyboardInterrupt:
        print("\nControl loop interrupted by user.")
    finally:
        print("Exiting teleoperation script.")
        # Add any cleanup code here if needed
        if avp_streamer and hasattr(avp_streamer, 'close'):
             try:
                 print("Closing AVP streamer...")
                 avp_streamer.close()
             except Exception as e:
                 print(f"Error closing AVP streamer: {e}")