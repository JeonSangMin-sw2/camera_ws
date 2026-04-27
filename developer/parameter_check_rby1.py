import rby1_sdk as rby
import time
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

np.set_printoptions(suppress=True, precision=3, linewidth=300)
ROBOT_ADDRESS = "127.0.0.1:50051"
robot = rby.create_robot(ROBOT_ADDRESS, "m")

if not robot.connect():
    logging.error("Could not connect to robot")
    exit(1)

# Power on is typically required for FT sensors
if not robot.is_power_on(".*"):
    logging.info("Powering on...")
    if not robot.power_on(".*"):
        logging.error("Failed to power on")
        exit(1)

logging.info("Robot connected and powered. Skipping motor activation for sensor monitoring.")

def monitor_ft_sensor(robot, duration_sec=10, rate_hz=10):
    print(f"\nMonitoring FT sensor data for {duration_sec} seconds...")
    print(f"{'Arm':<10} | {'Force (N)':<25} | {'Torque (Nm)':<25}")
    print("-" * 70)

    def callback(rs):
        # Format force and torque vectors for display
        rf = np.array2string(rs.ft_sensor_right.force, precision=2, separator=', ')
        rt = np.array2string(rs.ft_sensor_right.torque, precision=2, separator=', ')
        lf = np.array2string(rs.ft_sensor_left.force, precision=2, separator=', ')
        lt = np.array2string(rs.ft_sensor_left.torque, precision=2, separator=', ')

        # Clear line and print current values (using carriage return to update in place)
        print(f"\rRight FT: F {rf:<20} | T {rt:<20}\nLeft  FT: F {lf:<20} | T {lt:<20}", end="")
        # Move cursor up to overwrite on next iteration
        print("\033[A", end="") 

    robot.start_state_update(callback, rate=rate_hz)
    
    try:
        time.sleep(duration_sec)
    except KeyboardInterrupt:
        pass
    finally:
        robot.stop_state_update()
        print("\n" * 2) # Move cursor down after monitoring
        print("FT sensor monitoring stopped.")

# Briefly monitor FT sensor data
monitor_ft_sensor(robot, duration_sec=10)
