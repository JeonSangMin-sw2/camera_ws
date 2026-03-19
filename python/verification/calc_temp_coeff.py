import csv

def manual_regression(x, y):
    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_x2 = sum(xi**2 for xi in x)
    sum_xy = sum(x[i]*y[i] for i in range(n))
    
    denominator = n * sum_x2 - sum_x**2
    if denominator == 0:
        return 0, 0
    
    m = (n * sum_xy - sum_x * sum_y) / denominator
    b = (sum_y - m * sum_x) / n
    return m, b

def calculate_coefficients(filepath):
    temp = []
    x = []
    y = []
    z = []
    roll = []
    pitch = []
    yaw = []
    
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 7:
                try:
                    t = float(parts[0])
                    # 온도가 34도 미만인 쓰레기값이나 비정상값 필터링
                    if t < 30 or t > 50:
                        continue
                    temp.append(t)
                    x.append(float(parts[1]))
                    y.append(float(parts[2]))
                    z.append(float(parts[3]))
                    roll.append(float(parts[4]))
                    pitch.append(float(parts[5]))
                    yaw.append(float(parts[6]))
                except ValueError:
                    pass

    if len(temp) == 0:
        print("No valid data found.")
        return

    datasets = [
        ('x (mm)', x),
        ('y (mm)', y),
        ('z (mm)', z),
        ('roll (deg)', roll),
        ('pitch (deg)', pitch),
        ('yaw (deg)', yaw)
    ]
    
    print(f"Total valid samples: {len(temp)}")
    print("Linear Regression Results (Offset = Slope * (Current_Temp - 41.0)):")
    print("Value_Compensated = Value_Measured - Offset\n")
    
    for label, data in datasets:
        slope, intercept = manual_regression(temp, data)
        # 41도를 기준으로 보정한다고 가정할 때의 값을 계산하기 위해 기준점 출력
        val_at_41 = slope * 41.0 + intercept
        
        print(f"--- {label} ---")
        print(f"Slope (Coefficient per °C): {slope:.6f}")
        # print(f"Intercept at 0°C: {intercept:.6f}")
        print(f"Estimated value at 41°C: {val_at_41:.6f}")
        print()
        
if __name__ == "__main__":
    calculate_coefficients("log.txt")
