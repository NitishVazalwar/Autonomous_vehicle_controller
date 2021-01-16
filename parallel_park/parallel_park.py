"""parallel_park controller."""

from vehicle import Driver
from controller import Lidar, Camera, Gyro
import cv2
import numpy as np


# create the Robot instance.
driver = Driver()

timestep = int(driver.getBasicTimeStep())
lidar = driver.getLidar("Sick LMS 291")
lidar.enable(timestep)
lidar.enablePointCloud()
lidarWidth = lidar.getHorizontalResolution()
lidarMaxRange = lidar.getMaxRange()

gyro = Gyro("gyro")
gyro.enable(timestep)



# ss = cv2.CascadeClassifier('stop_sign.xml')
# k = 0
'''
Stage -1 : Backup if started at middle 
Stage 0 : Identify side of parking and distance
Stage 1 : Identify stretch for close up
Stage 2 : Close up
Stage 2.1 : Find park spot

'''
'''Variables for Stage 1'''
end_time = 0
start_time = 0
previous_value = 80
new_value = 80
lidar_error_factor = 0

'''Variables for stage 2'''
min_dist = 2
k = 0

'''Variables for stage 2.1'''
start_time1 = 0 
end_time1 = 0

def park_right(driver):
    driver.setCruisingSpeed(5)
    for i in range(250):
        driver.step()
    driver.setCruisingSpeed(0)
    for i in range(100):
        driver.step()
    driver.setSteeringAngle(.8)
    for i in range(150):
        driver.step()
    driver.setCruisingSpeed(-5)
    for i in range(150):
        driver.step()
    driver.setSteeringAngle(0)
    for i in range(150):
        driver.step()
    driver.setSteeringAngle(-.8)
    for i in range(150):
        driver.step()
    driver.setCruisingSpeed(0)
    driver.setSteeringAngle(0)
    
    
def park_left(driver):
    driver.setCruisingSpeed(5)
    for i in range(250):
        driver.step()
    driver.setCruisingSpeed(0)
    for i in range(100):
        driver.step()
    driver.setSteeringAngle(-.8)
    for i in range(150):
        driver.step()
    driver.setCruisingSpeed(-5)
    for i in range(150):
        driver.step()
    driver.setSteeringAngle(0)
    for i in range(150):
        driver.step()
    driver.setSteeringAngle(.8)
    for i in range(150):
        driver.step()
    driver.setCruisingSpeed(0)
    driver.setSteeringAngle(0)
    
    
    
def close_up_from_right(driver, distance):
    driver.setSteeringAngle(.8)
    for i in range(50):
        driver.step()
    driver.setCruisingSpeed(-5)
    for i in range(50):
        driver.step()
    driver.setSteeringAngle(0)
    straight_dist = int(120 * distance)
    for i in range(straight_dist):
        driver.step()
    driver.setSteeringAngle(-.8)
    for i in range(100):
        driver.step()
    driver.setCruisingSpeed(0)
    driver.setSteeringAngle(0)
    

def close_up_from_left(driver, distance):
    driver.setSteeringAngle(-.8)
    for i in range(50):
        driver.step()
    driver.setCruisingSpeed(-5)
    for i in range(50):
        driver.step()
    driver.setSteeringAngle(0)
    straight_dist = int(120 * distance)
    for i in range(straight_dist):
        driver.step()
    driver.setSteeringAngle(.8)
    for i in range(100):
        driver.step()
    driver.setCruisingSpeed(0)
    driver.setSteeringAngle(0)


stage = -1
park_at_right = False
park_at_left = False
parkable_space = 0
# brake_flag = 04
while driver.step()!= -1:
            
    lidarImg = lidar.getRangeImage()
    pointCloud = lidar.getPointCloud()
    pointCount = len(lidarImg)
    
    sideImgRight = lidarImg[179]
    sideImgLeft = lidarImg[0]
    # print(sideImg)
    lidar_array = np.array(lidarImg)
    

    if stage == -1:
        if sideImgRight < 20 or sideImgLeft < 20:
            driver.setCruisingSpeed(-5)
        else:
            driver.setCruisingSpeed(0)
            stage = 0


    elif stage == 0:
        # print("stage 0")
        if 3 <= sideImgRight < 20:
            distance = sideImgRight
            stage = 1
            print("Parking at right")
            print("Backing up")
            park_at_right = True
        else:
            driver.setCruisingSpeed(5)

        if 3 <= sideImgLeft < 20:
            distance = sideImgLeft
            stage = 1
            print("Parking at Left")
            print("Backing up")
            park_at_left = True
        else:
            driver.setCruisingSpeed(5)
        

    elif stage == 1:
        
        if park_at_right is True:
            new_value_trials = sideImgRight
            if abs(previous_value - new_value_trials) > 10:
                lidar_error_factor += 1
                if lidar_error_factor  > 10:
                    new_value = new_value_trials
                    lidar_error_factor = 0
                else:
                    pass
            else:
                lidar_error_factor = 0
            
            # print(previous_value, new_value)
            if previous_value >= 20 and new_value < 20:
                start_time = driver.getTime()
                previous_value = new_value
                
            elif previous_value < 20 and new_value >= 20:
                end_time = driver.getTime()
                previous_value = new_value
                
            else:
                pass
            if end_time > start_time and end_time > 0 and start_time > 0:
                duration = end_time - start_time
                no_parking_distance = duration * 5 #No units and speed referenced
                print("No parking distance ", no_parking_distance)
                if no_parking_distance > 5:
                    stage = 2
                    ("Closing up")
            elif start_time > end_time and end_time > 0 and start_time > 0:
                duration = start_time - end_time
                parking_distance = duration * 5 #No units and speed referenced
                print("Parking distance ", parking_distance)

        if park_at_left is True:
            new_value_trials = sideImgLeft
            if abs(previous_value - new_value_trials) > 10:
                lidar_error_factor += 1
                if lidar_error_factor  > 10:
                    new_value = new_value_trials
                    lidar_error_factor = 0
                else:
                    pass
            else:
                lidar_error_factor = 0
            
            # print(previous_value, new_value)
            if previous_value >= 20 and new_value < 20:
                start_time = driver.getTime()
                previous_value = new_value
                print("Start time ", start_time)
            elif previous_value < 20 and new_value >= 20:
                end_time = driver.getTime()
                previous_value = new_value
                print("End Time ", end_time)
            else:
                pass
            if end_time > start_time and end_time > 0 and start_time > 0:
                duration = end_time - start_time
                no_parking_distance = duration * 5 #No units and speed referenced
                print("No parking distance ", no_parking_distance)
                if no_parking_distance > 5:
                    stage = 2
                    ("Closing up")
            elif start_time > end_time and end_time > 0 and start_time > 0:
                duration = start_time - end_time
                parking_distance = duration * 5 #No units and speed referenced
                print("Parking distance ", parking_distance)


    elif stage == 2:
        if park_at_right is True:
            driver.setCruisingSpeed(-5)
            sideImgRight_trials = lidarImg[179]
            if sideImgRight_trials < 20:
                k += 1
                if k > 10:
                    sideImgRight = sideImgRight_trials
                else:
                    sideImgRight = 80
            else:
                k = 0   
            if sideImgRight < 20 and sideImgRight > min_dist:
                distance = sideImgRight - min_dist
                close_up_from_right(driver, distance)
                stage = 2.1
                print("Finding parking spots")
        
        
        if park_at_left is True:
            driver.setCruisingSpeed(-5)
            sideImgLeft_trials = lidarImg[0]
            if sideImgLeft_trials < 20:
                k += 1
                if k > 10:
                    sideImgLeft = sideImgLeft_trials
                else:
                    sideImgLeft = 80
            else:
                k = 0   
            if sideImgLeft < 20 and sideImgLeft > min_dist:
                distance = sideImgLeft - min_dist
                close_up_from_left(driver, distance)
                stage = 2.1
                print("Finding parking spots")
                
        
    elif stage == 2.1:
        driver.setCruisingSpeed(5)
        if park_at_right is True:
            new_value_trials = sideImgRight
            if abs(previous_value - new_value_trials) > 10:
                lidar_error_factor += 1
                if lidar_error_factor  > 10:
                    new_value = new_value_trials
                    lidar_error_factor = 0
                else:
                    pass
            else:
                lidar_error_factor = 0
            # print(previous_value, new_value_trials)
            # print(previous_value, new_value)
            if previous_value >= 20 and new_value < 20:
                start_time1 = driver.getTime()
                previous_value = new_value
                
            elif previous_value < 20 and new_value >= 20:
                end_time1 = driver.getTime()
                previous_value = new_value
                
            else:
                pass
            if end_time1 > start_time1 and end_time1 > 0 and start_time1 > 0:
                duration = end_time1 - start_time1
                no_parking_distance = duration * 5 #No units and speed referenced
                print("No parking distance ", no_parking_distance)
                
            elif start_time1 > end_time1 and end_time1 > 0 and start_time1 > 0:
                duration = start_time1 - end_time1
                parking_distance = duration * 5 #No units and speed referenced
                print("Parking distance ", parking_distance)
                if parking_distance > 20:
                    stage = 3
        
        
        if park_at_left is True:
            new_value_trials = sideImgLeft
            if abs(previous_value - new_value_trials) > 10:
                lidar_error_factor += 1
                if lidar_error_factor  > 10:
                    new_value = new_value_trials
                    lidar_error_factor = 0
                else:
                    pass
            else:
                lidar_error_factor = 0
            # print(previous_value, new_value_trials)
            # print(previous_value, new_value)
            if previous_value >= 20 and new_value < 20:
                start_time1 = driver.getTime()
                previous_value = new_value
                
            elif previous_value < 20 and new_value >= 20:
                end_time1 = driver.getTime()
                previous_value = new_value
                
            else:
                pass
            if end_time1 > start_time1 and end_time1 > 0 and start_time1 > 0:
                duration = end_time1 - start_time1
                no_parking_distance = duration * 5 #No units and speed referenced
                print("No parking distance ", no_parking_distance)
                
            elif start_time1 > end_time1 and end_time1 > 0 and start_time1 > 0:
                duration = start_time1 - end_time1
                parking_distance = duration * 5 #No units and speed referenced
                print("Parking distance ", parking_distance)
                if parking_distance > 20:
                    stage = 3
                
    elif stage == 3:
        print("Parking started")
        driver.setCruisingSpeed(5)
        if park_at_right is True:
            park_right(driver)
            stage = 4
            print("Finishing up")
        elif park_at_left is True:
            park_left(driver)
            stage = 4
            print("Finishing up")
    
    elif stage == 4:
        
        if lidarImg[90] > 1:
            driver.setCruisingSpeed(2)
        else:
            driver.setCruisingSpeed(0)
    pass







