import rby1_sdk
import inspect

print("--- Model_M members ---")
for name, obj in inspect.getmembers(rby1_sdk.Model_M):
    print(name)
