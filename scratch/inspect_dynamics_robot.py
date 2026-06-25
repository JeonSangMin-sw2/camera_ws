import rby1_sdk.dynamics as rd

for cls in [rd.Robot, rd.Robot_18, rd.Robot_24, rd.Robot_26]:
    print(f"\n=== Class {cls.__name__} ===")
    for name in dir(cls):
        if not name.startswith('_'):
            print(name)
