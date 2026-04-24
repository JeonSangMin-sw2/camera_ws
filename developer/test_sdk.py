import rby1_sdk as rby
robot = rby.create_robot("192.168.30.1:50051", "a")
robot.connect()
model = robot.model()
print("dir(model):", dir(model))
# Try to find link info
