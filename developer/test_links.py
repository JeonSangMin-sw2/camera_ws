import rby1_sdk as rby
try:
    robot = rby.create_robot("192.168.30.1:50051", "a")
    robot.connect()
    print("Joint names:", robot.model().robot_joint_names)
except Exception as e:
    print("Unable to connect:", e)
