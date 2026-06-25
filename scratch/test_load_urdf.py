import rby1_sdk.dynamics as rd

urdf_path = "/home/rainbow/sdk/rby1-sdk/models/rby1m/urdf/model_v1.3.urdf"
robot_config = rd.load_robot_from_urdf(urdf_path, "base")
print("Robot loaded successfully:", robot_config)

dyn_robot = rd.Robot(robot_config)
print("Robot dynamics solver created successfully:", dyn_robot)
print("DOF:", dyn_robot.get_dof())
print("Joint names:", dyn_robot.get_joint_names())
