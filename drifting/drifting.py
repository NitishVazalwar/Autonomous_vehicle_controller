"""drifting controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from vehicle import Driver
from controller import Lidar, Camera
import cv2
import numpy as np

# create the Robot instance.
robot = Driver()

# initialize cameras
front_camera = robot.getCamera("front_camera")
front_camera.enable(30)

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())

# initialize  and enable lidar
lidar = robot.getLidar("Sick LMS 291")
#lidar.enablePointCloud()
lidar.enable(timestep)

# set lidar global variables
lidarWidth = lidar.getHorizontalResolution()
sideLidarCount = int(lidarWidth/4)
frontLidarCount = 8

# set global variables
maxSpeed = 70.0
cameraOnlySpeed = 35.0
turnSpeed = 35.0
brakeIntensity = 0
turnDist = 25.0
cameraSteerAngle = 0
lidarSteerAngle = 0
steerAngle = 0
steerAngleDB = 0.1
turnAngle = 0.3
curSpeed = maxSpeed
lidarSideLengthLimit = 6
lidarFrontLengthLimit = 40
lidarRailCheckMin = 19
sideDistDB = 0.2
cX = 0
cY = 0

    
# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step() != -1:

    # Read camera data
    front_image_array = np.asarray(front_camera.getImageArray(), dtype = np.uint8)
    hsv = cv2.cvtColor(front_image_array, cv2.COLOR_BGR2HSV)
    img_rotate_90_clockwise = cv2.rotate(hsv, cv2.ROTATE_90_CLOCKWISE)
    final_image = cv2.flip(img_rotate_90_clockwise, 1)
    rows, columns, _ = final_image.shape
    
    tar_filter = cv2.inRange(final_image, (0, 5, 40), (15, 30, 90)) #Greenlight HSV
    shadow_filter = cv2.inRange(final_image, (0, 50, 20), (15, 95, 40))
    tar_filter += shadow_filter
    kernel = np.ones((3,3), np.uint8)
    tar_filter = cv2.erode(tar_filter, kernel, iterations = 3)
    tar_filter = cv2.dilate(tar_filter, kernel, iterations = 9)
    
    if np.sum(tar_filter) > 0:
        M = cv2.moments(tar_filter)
        cY = int(M["m10"]/M["m00"])
        cX = int(M["m01"]/M["m00"])
        
    # set camera steer angle
    if abs(cY - (columns/2)) > 0:
        cameraSteerAngle = (cY - (columns/2))/(columns/4)
        if cameraSteerAngle > 1:
            cameraSteerAngle = 1
        elif cameraSteerAngle < -1:
            cameraSteerAngle = -1
    elif np.sum(tar_filter) == 0:
        cameraSteerAngle = steerAngle
    else:
        cameraSteerAngle = 0
    
    # Read lidar values
    lidarValues = lidar.getRangeImage()
    
    # create arrays for lidar groups
    lidarLeft = []
    lidarRight = []
    lidarFront = []
    lidarLeftRailCheckArray = []
    lidarRightRailCheckArray = []
    
    # populate left and right arrays
    for i in range(sideLidarCount):
        j = lidarWidth - 1 - i
        left = lidarValues[i]
        right = lidarValues[j]
        if left < lidarSideLengthLimit:
            lidarLeft.append(left)
        if right < lidarSideLengthLimit:
            lidarRight.append(right)
    
    # populate front array    
    for k in range(frontLidarCount):
        point = int(lidarWidth / 2) - frontLidarCount + k
        front = lidarValues[point]
        if front < lidarFrontLengthLimit:
            lidarFront.append(front)
            
    # populate arrays to check for presence of guard rails
    for point in range(int(lidarWidth / 2)):
        leftPoint = point
        rightPoint = (lidarWidth - 1) - point
        lidarLeftRailCheckArray.append(lidarValues[leftPoint])
        lidarRightRailCheckArray.append(lidarValues[rightPoint])
    
    # calculate length of average lidar point to each side
    lidarLeftRailCheck = sum(lidarLeftRailCheckArray) / len(lidarLeftRailCheckArray)
    lidarRightRailCheck = sum(lidarRightRailCheckArray) / len(lidarRightRailCheckArray)
    
    # calculate distance to nearest obstacle in front    
    if len(lidarFront) > 0:
        frontDist = sum(lidarFront) / len(lidarFront)
    else:
        frontDist = lidarFrontLengthLimit
    
    # calculate distance to nearest obstacle on either side    
    if len(lidarLeft) > 0:
        leftDist = sum(lidarLeft) / len(lidarLeft)
    elif frontDist > turnDist:
        leftDist = rightDist
    else:
        leftDist = lidarSideLengthLimit
        
    if len(lidarRight) > 0:
        rightDist = sum(lidarRight) / len(lidarRight)
    elif frontDist > turnDist:
        rightDist = leftDist
    else:
        rightDist = lidarSideLengthLimit
    
    # Adjust steer angle based on difference between left and right distances    
    if leftDist - rightDist > sideDistDB:
        lidarSteerAngle = (rightDist / leftDist) - 1
        if lidarSteerAngle < -1:
            lidarSteerAngle = -1
    elif rightDist - leftDist > sideDistDB:
        lidarSteerAngle = 1 - (leftDist / rightDist)
        if lidarSteerAngle > 1:
            lidarSteerAngle = 1
    else:
        lidarSteerAngle = 0
    
    # switch to camera control speed if no guard rail detected
    if lidarLeftRailCheck >= lidarRailCheckMin and lidarRightRailCheck >= lidarRailCheckMin:
        curSpeed = cameraOnlySpeed
        if robot.getCurrentSpeed() > cameraOnlySpeed:
            brakeIntensity = 1
        else:
            brakeIntensity = 0
        
    # set steer angle based on lidar and camera calulated angles
    if abs(cameraSteerAngle) >= steerAngleDB and abs(lidarSteerAngle) < steerAngleDB:
        curspeed = turnSpeed
        steerAngle = cameraSteerAngle
        brakeIntensity = abs(steerAngle)
    elif abs(cameraSteerAngle) < steerAngleDB and abs(lidarSteerAngle) >= steerAngleDB:
        steerAngle = lidarSteerAngle
    elif abs(cameraSteerAngle) >= steerAngleDB and abs(lidarSteerAngle) >= steerAngleDB:
        steerAngle = (cameraSteerAngle + lidarSteerAngle)/2
    else:
        steerAngle = 0
        
    # set brake intensity and speed based on steer angle and approaching obstacles    
    if frontDist < turnDist and robot.getCurrentSpeed() > turnSpeed:
        curSpeed = turnSpeed
        brakeIntensity = 1 - (frontDist / lidarFrontLengthLimit)
    elif steerAngle > steerAngleDB:
        brakeIntensity = abs(steerAngle) - steerAngleDB
    elif frontDist >= turnDist and steerAngle < steerAngleDB and lidarLeftRailCheck < lidarRailCheckMin and lidarRightRailCheck < lidarRailCheckMin:
        curSpeed = maxSpeed
        brakeIntensity = 0
        
    # limit output values for brakeIntensity and steerAngle
    # reduce requested speed if breaking
    if brakeIntensity > 0:
        curSpeed = turnSpeed
        if brakeIntensity > 1:
            brakeIntensity = 1
    
    if steerAngle > 1:
        steerAngle = 1
    elif steerAngle < -1:
        steerAngle = -1
        
    # send commands to actuators
    #robot.setGear(gear)
    robot.setBrakeIntensity(brakeIntensity)
    robot.setSteeringAngle(steerAngle)
    robot.setCruisingSpeed(curSpeed)
    
    if curSpeed==turnSpeed:
        robot.setGear(1)
        robot.setThrottle(.8)
    pass
# Enter here exit cleanup code.


