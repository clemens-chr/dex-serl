""" Test the spacemouse output. """
import time
import numpy as np
from franka_env.teleop.avp_expert import AVPExpert


def test_avp():
    """Test the AVPExpert class.

    This interactive test prints the coordinates of the AVP at a rate of 10Hz.
    Also prints whether the user wants to intervene (pinching_left).
    Prints whether gripper is closing or opening (pinching_right).

    """
    avp = AVPExpert()
    with np.printoptions(precision=3, suppress=True):
        while True:
            action, pinching = avp.get_action()
            print(f"AVP Action: {action}, Pinching: {pinching}, Intervention: {avp.is_intervening()}")
            time.sleep(0.1)


def main():
    """Call avp test."""
    test_avp()


if __name__ == "__main__":
    main()
