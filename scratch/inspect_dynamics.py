import rby1_sdk
import inspect

for name, obj in inspect.getmembers(rby1_sdk):
    if "dynamics" in name.lower() or "model" in name.lower() or "kinematics" in name.lower():
        print(name)
