import math

def get_discrete_angle(angle):
    # [no_action, move_left, move_right, move_down, move_up]
    if math.pi >= angle > 3/4*math.pi:
        return 1
    if 3/4*math.pi >= angle > math.pi/4:
        return 4
    if math.pi/4 >= angle > -math.pi/4:
        return 2
    if -math.pi/4 >= angle > -5/4*math.pi:
        return 3
    if -5/4*math.pi >= angle >= -math.pi:
        return 1
    else:
        return 0

def get_discrete_speed_8(speed):
    discrete_speed = 1
    speed = float(speed)
    if speed >= 0.875:  # 1
        discrete_speed = 0
    if 0.625 < speed <= 0.875:  # 0.75
        discrete_speed = 1
    if 0.365 < speed <= 0.625:  # 0.5
        discrete_speed = 2
    if 0.125 < speed <= 0.365:  # 0.25
        discrete_speed = 3
    if -0.125 < speed <= 0.125:  # 0
        discrete_speed = 4
    if -0.365 < speed <= -0.125:  # -0.25
        discrete_speed = 5
    if -0.625 < speed <= -0.365:  # -0.5
        discrete_speed = 6
    if -0.875 < speed <= -0.625:  # -0.75
        discrete_speed = 7
    if speed <= -0.875:  # -1
        discrete_speed = 8
    return discrete_speed

def get_discrete_speed(speed):
    discrete_speed = 1
    speed = float(speed)
    if speed >= 0.75:  # 1
        discrete_speed = 0
    if 0.25 < speed <= 0.75:  # 0.5
        discrete_speed = 1
    if -0.25 < speed <= 0.25:  # 0
        discrete_speed = 2
    if -0.75 < speed <= -0.25:  # -0.5
        discrete_speed = 3
    if speed <= -0.75:  # -1
        discrete_speed = 4
    return discrete_speed

def get_distance(xA, yA, xB, yB):
    return math.sqrt(math.pow(xA - xB, 2) + math.pow(yA - yB, 2))

def get_agent_distance(poseA, poseB):
    return get_distance(poseA[0], poseA[1], poseB[0], poseB[1])

def get_points_from_segment(xA, yA, xB, yB):
    theta = math.atan2(yB - yA, xB - xA)
    dist = math.sqrt((math.pow(xB - xA, 2) + math.pow(yB - yA, 2)))
    rho = []
    step = 0.5
    r = 0
    while r < dist:
        rho.append(r)
        r += step
    rho.append(dist)
    x = []
    y = []
    for r in rho:
        x.append(r * math.cos(theta) + xA)
        y.append(r * math.sin(theta) + yA)

    return x, y