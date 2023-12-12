import rospy
from sensor_msgs.msg import Joy
from std_msgs.msg import Bool
from move_base_msgs.msg import MoveBaseAction
import actionlib

class DeadmanSwitch:
    def __init__(self):

        self.bool_msg = None
        self.pub = rospy.Publisher('e_stop',
                              Bool,
                              queue_size=1)
        

        self.joy_topic = '/bluetooth_teleop/joy'
        rospy.Subscriber(self.joy_topic,
                         Joy,
                         self.joy_callback)


    def joy_callback(self, joy_msg):
        
        # If the PS4 joystick loses connection, true is published to e_stop topic
        # Setting robot into idle mode
        if joy_msg.axes[2] == 0.0 and joy_msg.axes[-1] == 0.0:
            self.bool_msg = True
        
        # Manually shutdown the node
        if joy_msg.buttons[3] == 1.0:
            rospy.signal_shutdown("deadman switch active ..")
            self.bool_msg = True

        # R2 button as a switch for e_stop
        if joy_msg.axes[-1] == -1.0:
            self.bool_msg = False
        else:
            self.bool_msg = True
        
        self.pub.publish(self.bool_msg)


if __name__ == '__main__':
    try:

        rospy.init_node('deadman_switch')
        node = DeadmanSwitch()
        rospy.spin()
        # joy_to_bool_subscriber()
    except rospy.ROSInterruptException:
        pass