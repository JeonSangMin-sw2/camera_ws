import rby1_sdk
import inspect

print(rby1_sdk.dynamics)
for name, obj in inspect.getmembers(rby1_sdk.dynamics):
    print(name)
