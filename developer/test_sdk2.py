import rby1_sdk as rby
import sys

robot = rby.create_robot("192.168.30.1:50051", "a")
robot.connect()
model = robot.model()

methods = [m for m in dir(model) if not m.startswith('_')]
print("Model methods:", methods)

state = robot.get_state()
print("State object methods:", [m for m in dir(state) if not m.startswith('_')])

rob_methods = [m for m in dir(robot) if not m.startswith('_')]
print("Robot methods:", rob_methods)
