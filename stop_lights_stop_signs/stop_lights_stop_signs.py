"""stop_lights_stop_signs controller."""

from vehicle import Driver
from controller import Lidar, Camera
import cv2
import numpy as np



driver = Driver()
front_camera = Camera("front_camera")
front_camera.enable(30)

redFlag = 0
redsum = 0
i = 0
y = 0
ss = cv2.CascadeClassifier('stop_sign.xml')
k = 0
stop_flag = 0
start_time = 0
duration = 0

def stopCar(driver):
    driver.setCruisingSpeed(0)
    for i in range(300):
        driver.step()

# driver.setGear(gear)
driver.setCruisingSpeed(30)

while driver.step()!= -1:
    front_image_array = np.asarray(front_camera.getImageArray(), dtype = np.uint8)
    hsv = cv2.cvtColor(front_image_array, cv2.COLOR_BGR2HSV)
    img_rotate_90_clockwise = cv2.rotate(hsv, cv2.ROTATE_90_CLOCKWISE)
    final_image = cv2.flip(img_rotate_90_clockwise, 1)
    rows, columns, _ = final_image.shape

    #  Observing roads
    tar_filter = cv2.inRange(final_image, (0, 5, 40), (15, 30, 90)) 
    shadow_filter = cv2.inRange(final_image, (0, 50, 20), (15, 95, 40))
    tar_filter += shadow_filter
    kernel = np.ones((3,3), np.uint8)
    tar_filter = cv2.erode(tar_filter, kernel, iterations=3)
    tar_filter = cv2.dilate(tar_filter, kernel, iterations=9)
    
    #Watching for Red light
    red_filter = cv2.inRange(final_image, (100, 170, 210), (150, 210, 255))
    red_filter = cv2.dilate(red_filter, kernel, iterations=8)
    redsumtrials = np.sum(red_filter)
    
    yellow_filter = cv2.inRange(final_image, (90, 130, 250), (95, 138, 255))
    yellow_filter = cv2.dilate(yellow_filter, kernel, iterations=8)
    yellowsumtrials = np.sum(yellow_filter)
    
    green_filter = cv2.inRange(final_image, (25, 180, 235), (55, 210, 255))
    green_filter = cv2.dilate(green_filter, kernel, iterations=8)
    green_sum = np.sum(green_filter)
    
    gray = cv2.cvtColor(final_image, cv2.COLOR_BGR2GRAY)
    SS = ss.detectMultiScale(gray, 1.1, 2)
    
    # driver.setCruisingSpeed(30)
    if stop_flag == 0:
        for (x, y, w, x) in SS:
            stop_flag = 1
            print("stop Sign Detected ", k)
            k += 1
            if stop_flag == 1:
                stop_flag = 2
                start_time = driver.getTime()
                stopCar(driver)
                

    if start_time > 0:
        end_time = driver.getTime()
        duration = end_time - start_time
    print(duration)
    if duration > 15:
        stop_flag = 0
        start_time = 0
    
    if redsumtrials  > 78000:
        i += 1
        if i > 3:
            redsum = redsumtrials
        else:
            redsum = 0
    else:
        i = 0
        redsum = 0
        
    if yellowsumtrials  < 45000:
        y += 1
        if y > 5:
            yellow_sum = yellowsumtrials
        else:
            yellow_sum = 50000
    else:
        y = 0
        yellow_sum = 50000
        
        
    print(redsumtrials, yellow_sum, green_sum)
    
    if redsum > 78000 or yellow_sum > 40000:
        redFlag = 1
        driver.setCruisingSpeed(0)
    elif redsum == 0 and yellow_sum == 0:
        driver.setCruisingSpeed(30)
    
    if np.sum(tar_filter) >0:
        M = cv2.moments(tar_filter)
        cY = int(M["m10"] / M["m00"])
        cX = int(M["m01"] / M["m00"])

        
        if abs(cY - (columns/2)) > 0:
                steering_angle = driver.setSteeringAngle((cY - (columns/2))/(columns/4))
        elif np.sum(tar_filter) == 0:
            driver.setSteeringAngle(steering_angle)
        else:
            driver.setSteeringAngle(0)
    
    cv2.imshow("orginal with line", red_filter)
    # cv2.imshow("orginal with lin", final_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    
    pass







