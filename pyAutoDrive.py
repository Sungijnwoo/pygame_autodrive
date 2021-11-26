import pygame
from pygame.locals import * 
from time import sleep
import random
import numpy as np
import math
import cv2 as cv
import sys
import statistics
import copy
# Import a library of functions called 'pygame'
 
# Initialize the game engine
pygame.init()
  
# Define the colors we will use in RGB format
BLACK= ( 0,  0,  0)
WHITE= (255,255,255)
BLUE = ( 0,  0, 255)
GREEN= ( 0 ,255,  0)
RED  = (255,  0,  0)
YELLOW = (250, 250, 20)
GRAY = (128, 128, 128)
offset = 20 
# Set the height and width of the screen
size  = [1500,600]
screen= pygame.display.set_mode(size)
pygame.display.set_caption("pyAutodrive")
# pygame.mixer.music.load('bgm.mp3') 
# pygame.set_volume(0.1)
# Loop until the user clicks the close button.
done= False
clock= pygame.time.Clock()
length = 30
gap = 70
c = 1
v = 20
color = None
#성열이 코딩
color1 = 2
color2 = 10.67
color3 = 50

saturation_th1 = 74
saturation_th2 = 196
saturation_th3 = 116
value_th1 = 65
value_th2 = 77
value_th3 = 59

green_check = []
j = 0
contour_center = False

ranges = 10

font = cv.FONT_HERSHEY_SIMPLEX
fontScale = 2

direction = 0
LEFT = pygame.image.load('./imgs/arrow.png')
LEFT_rect = LEFT.get_rect()
LEFT_rect.x, LEFT_rect.y = (size[0]-140, 20)

def nothing(x):
    global color1, color2, color3
    global lower_red1, lower_red2, lower_red3
    global upper_red1, upper_red2, upper_red3
    global lower_yellow1, lower_yellow2, lower_yellow3
    global upper_yellow1, upper_yellow2, upper_yellow3
    global lower_green1, lower_green2, lower_green3
    global upper_green1, upper_green2, upper_green3

    color1 = int(color1)
    color2 = int(color2)
    color3 = int(color3)

    if color1 in range(0, 7):
        lower_red1 = np.array([color1 - ranges + 180, saturation_th1, value_th1])
        upper_red1 = np.array([180, 255, 255])
        lower_red2 = np.array([0, saturation_th1, value_th1])
        upper_red2 = np.array([color1, 255, 255])
        lower_red3 = np.array([color1, saturation_th1, value_th1])
        upper_red3 = np.array([color1 + ranges, 255, 255])
    elif color1 in range(165, 180):
        lower_red1 = np.array([color1, saturation_th1, value_th1])
        upper_red1 = np.array([180, 255, 255])
        lower_red2 = np.array([0, saturation_th1, value_th1])
        upper_red2 = np.array([color1 + ranges - 180, 255, 255])
        lower_red3 = np.array([color1 - 180 + (2*ranges), saturation_th1, value_th1])
        upper_red3 = np.array([color1 - 180 + (3*ranges), 255, 255])


    lower_yellow1 = np.array([color2 - (2*ranges), saturation_th2, value_th2])
    upper_yellow1 = np.array([color2 - ranges, 255, 255])
    lower_yellow2 = np.array([color2 - ranges, saturation_th2, value_th2])
    upper_yellow2 = np.array([color2, 255, 255])
    lower_yellow3 = np.array([color2, saturation_th2, value_th2])
    upper_yellow3 = np.array([color2 + ranges, 255, 255])

    lower_green1 = np.array([color3 - (2*ranges), saturation_th3, value_th3])
    upper_green1 = np.array([color3 - ranges, 255, 255])
    lower_green2 = np.array([color3 - ranges, saturation_th3, value_th3])
    upper_green2 = np.array([color3, 255, 255])
    lower_green3 = np.array([color3, saturation_th3, value_th3])
    upper_green3 = np.array([color3 + ranges, 255, 255])


cv.namedWindow('img_color')

cap = cv.VideoCapture(0)

class Car(pygame.sprite.Sprite):
    def __init__(self):
        super(Car, self).__init__()
        self.image = pygame.image.load('./imgs/car4.png')
        self.o_img = pygame.image.load('./imgs/car4.png')
        self.rect = self.image.get_rect()
        self.width = self.rect.size[0]
        self.height = self.rect.size[1]
        self.rect.x = 100
        self.rect.y = size[1] - self.rect.size[1]
        self.rotate_memory = 0
        self.offset = 0
        self.rotate_angle = 0

    def move(self, x, y):
        self.rect.center = (x, y)

    def collide(self, sprites):
        for sprite in sprites:
            if pygame.sprite.collide_circle(self, sprite):
                return sprite

    def rotate(self, x):
        try:
            if (x[1][1]-x[0][1])/(x[1][0]-x[0][0]) > 0 and x[1][1]-x[0][1] < 0:
                self.offset = 180
            elif (x[1][1]-x[0][1])/(x[1][0]-x[0][0]) > 0 and x[1][1]-x[0][1] > 0:
                self.offset = 0
            elif (x[1][1]-x[0][1])/(x[1][0]-x[0][0]) < 0 and x[1][1]-x[0][1] > 0:
                self.offset = 180
            elif (x[1][1]-x[0][1])/(x[1][0]-x[0][0]) < 0 and x[1][1]-x[0][1] < 0:
                self.offset = 0
            self.rotate_angle = -90 -np.arctan2((x[1][1]-x[0][1])/(x[1][0]-x[0][0]), 1)*180/np.pi -self.offset
            self.image = pygame.transform.rotate(self.o_img, self.rotate_angle%360)
            self.temp = [self.rect.x , self.rect.y]
            self.rect = self.image.get_rect()
            self.rect.center = (self.temp[0] + self.width/2, self.temp[1] + self.height/2)
            self.rotate_memory = self.rotate_angle

        except ZeroDivisionError:
            pass

class Traffic_Light():
    def __init__(self):
        self.image = pygame.image.load('./imgs/traffic.png')
        self.rect = self.image.get_rect()
        self.width = self.rect.size[0]
        self.height = self.rect.size[1]
        self.rect.x, self.rect.y = size[0]-self.width, 0


class Dot:
    def __init__(self):
        self.image = pygame.image.load('./imgs/dot.png')
        self.rect = self.image.get_rect()
        self.width = self.rect.size[0]
        self.height = self.rect.size[1]
        self.rect.x = 115
        self.rect.y = size[1] - 69
        self.diff = '무한'
    
    def move(self, x, y):
        self.nextx = (x[1][0]+y[1][0])/2
        self.nexty = (x[1][1]+y[1][1])/2 
        
        try:    
            self.diff = -(self.rect.y - self.nexty)/(self.rect.x - self.nextx)
            if self.diff > 8:
                self.diff = 7
            elif self.diff < -8:
                self.diff = -7
            self.angle = np.arctan2(self.diff, 1)
            self.dx = v * math.cos(self.angle)
            self.dy = v * math.sin(self.angle) 
            if self.dx < 1 and self.dx >0.7:
                self.dx = 1
                
            elif self.dy < 1 and self.dy > 0.7:
                self.dy = 1


        except ZeroDivisionError:
            self.dx = 0
            self.dy = v
        
        if self.rect.x - self.nextx !=0 and -(self.rect.y - self.nexty)/(self.rect.x - self.nextx) >0 and self.nextx - self.rect.x < 0:
            self.dx = -self.dx
            self.dy = -self.dy
        elif self.rect.x - self.nextx !=0 and -(self.rect.y - self.nexty)/(self.rect.x - self.nextx) < 0 and self.nextx - self.rect.x < 0:
            self.dx = -self.dx
            self.dy = -self.dy
        if color == 'RED' or color == 'YELLOW':
            self.dx = 0
            self.dy = 0


        self.rect.x += self.dx
        self.rect.y -= self.dy
                

class StopLine(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super(StopLine, self).__init__()
        self.image = pygame.image.load('./imgs/dot.png')
        self.rect = self.image.get_rect()
        self.rect.x = (x[1][0]+y[1][0])//2
        self.rect.y = (x[1][1]+y[1][1])//2 



class Line:
    def __init__(self, w, h, x, y):
        self.lpos = [[80, size[1]], [80, size[1]-h-offset]]
        self.rpos = [[80+(x - 80)*2 + w, size[1]], [80+(x - 80)*2 + w, size[1]-h-offset]]
        self.rot = 0
        self.end = [self.lpos[-1]]

    def right_rotate(self):
        self.diff = 5
        self.temp = self.rot
        self.r_angle = np.arctan2(-1/self.diff, 1)
        self.rd =  np.array([[gap*math.cos(self.r_angle)], [gap*math.sin(self.r_angle)]])
        
        while self.rot - self.temp > -np.pi/2:
            self.t = np.array([[math.cos(self.rot), -math.sin(self.rot)], [math.sin(self.rot), math.cos(self.rot)]])
            self.ld = np.array([[length/(1+abs(self.diff))], [length*self.diff/(1+abs(self.diff))]])
            self.dx, self.dy = np.dot(self.t, self.ld)[0][0], np.dot(self.t, self.ld)[1][0]
            self.lpos.append([self.lpos[-1][0]+self.dx, self.lpos[-1][1]-self.dy])

            self.dx, self.dy = np.dot(self.t, self.rd)[0][0], np.dot(self.t, self.rd)[1][0]
            self.rpos.append([self.lpos[-2][0]+self.dx, self.lpos[-2][1]-self.dy])

            self.rot += -np.pi/2 + np.arctan2(self.diff, 1)

        self.dx, self.dy = np.dot(self.t, self.rd)[0][0], np.dot(self.t, self.rd)[1][0]
        self.rpos.append([self.lpos[-1][0]+self.dx, self.lpos[-1][1]-self.dy])
        self.end.append(self.lpos[-4])

    def left_rotate(self):
        self.diff = -5
        self.temp = self.rot
        self.r_angle = np.arctan2(-1/self.diff, 1)
        self.rd =  np.array([[-gap*math.cos(self.r_angle)], [-gap*math.sin(self.r_angle)]])
        while self.rot - self.temp < np.pi/2:
            
            self.t = np.array([[math.cos(self.rot), -math.sin(self.rot)], [math.sin(self.rot), math.cos(self.rot)]])
            self.ld = np.array([[-length/(1+abs(self.diff))], [-length*self.diff/(1+abs(self.diff))]])
            self.dx, self.dy = np.dot(self.t, self.ld)[0][0], np.dot(self.t, self.ld)[1][0]
            self.rpos.append([self.rpos[-1][0]+self.dx, self.rpos[-1][1]-self.dy])


            self.dx, self.dy = np.dot(self.t, self.rd)[0][0], np.dot(self.t, self.rd)[1][0]
            self.lpos.append([self.rpos[-2][0]+self.dx, self.rpos[-2][1]-self.dy])

            self.rot += np.pi/2 + np.arctan2(self.diff, 1)

        self.dx, self.dy = np.dot(self.t, self.rd)[0][0], np.dot(self.t, self.rd)[1][0]
        self.lpos.append([self.rpos[-1][0]+self.dx, self.rpos[-1][1]-self.dy])
        self.end.append(self.lpos[-4])

    def straight(self, cnt):
        self.temp = self.lpos[-1]
        while (self.lpos[-1][0] - self.temp[0])**2 + (self.lpos[-1][1] - self.temp[1])**2 < 16:
            self.lpos.append([self.lpos[-1][0]+(self.lpos[-1][0] - self.lpos[-2][0]), self.lpos[-1][1]+(self.lpos[-1][1] - self.lpos[-2][1])])
            self.rpos.append([self.rpos[-1][0]+(self.rpos[-1][0] - self.rpos[-2][0]), self.rpos[-1][1]+(self.rpos[-1][1] - self.rpos[-2][1])])
        if cnt == 0:
            self.end.append(self.lpos[-1])
        else:
            self.end.append(self.lpos[-4])

    def Tline(self):
        self.temp_l = copy.deepcopy(self.lpos)
        self.temp_r = copy.deepcopy(self.rpos)
        self.temp_a = copy.deepcopy(self.rot)
        self.left_rotate()
        self.result_l = self.lpos[-10:]
        self.lpos = copy.deepcopy(self.temp_l)
        self.rpos = copy.deepcopy(self.temp_r)
        self.rot = copy.deepcopy(self.temp_a)
        self.right_rotate()
        self.lpos = self.temp_l + self.result_l
        self.rot = copy.deepcopy(self.temp_a)

        




ii=0
mode = 0
light = Traffic_Light()
car = Car()
dot = Dot()
line= Line(car.width, car.height, car.rect.x, car.rect.y)
done = True
stopline = pygame.sprite.Group()
stopline.add(StopLine(line.lpos, line.rpos))
pass_lpos = [[80, size[1]], [80, size[1]-car.height-offset]]
pass_rpos = [[80+(car.rect.x - 80)*2 + car.width, size[1]], [80+(car.rect.x - 80)*2 + car.width, size[1]-car.width-offset]]
lines = [3,2,1,1,1,1,1,1,1,1,1,3,1,1,1,1,3,3,1,1,1,1,2,2,1,1,1,2,1,1,1,1,3,3,2,1,1,1,1,3,2,1,1,1,4,1]

stop_car =0
cnt = 0
temp_angle = 0
while done and ii < len(lines):
    clock.tick(10)
    print('현재 위치 : {}, 기울기 : {}, 신호등 : {}'.format(car.rect.center, dot.diff, color))
    plus_x = 50
    plus_y = 100

    for event in pygame.event.get():
        if event.type == pygame.QUIT:   
            done=False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                color = 'RED'
            if event.key == pygame.K_s:
                mode = 1
            if event.key == pygame.K_y:
                color = 'YELLOW'
            if event.key == pygame.K_g:
                color = 'GREEN'
            if event.key == pygame.K_l:
                color = 'LEFT'
            
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_r or event.key == pygame.K_y or event.key == pygame.K_g or event.key == pygame.K_l:
                color = 'None'
            

#########################################################################################################

    ret, img_color = cap.read()

    if ret == False:
        continue

    img_color2 = img_color.copy()
    img_hsv = cv.cvtColor(img_color2, cv.COLOR_BGR2HSV)

    height, width = img_color.shape[:2]
    win_cx = int(width / 2)
    win_cy = int(height / 2)
    location = (win_cx - 250, win_cy - 150)

    nothing(0)

    # 범위 값으로 HSV 이미지에서 마스크를 생성합니다.
    img_mask_r1 = cv.inRange(img_hsv, lower_red1, upper_red1)
    img_mask_r2 = cv.inRange(img_hsv, lower_red2, upper_red2)
    img_mask_r3 = cv.inRange(img_hsv, lower_red3, upper_red3)
    temp = cv.bitwise_or(img_mask_r1, img_mask_r2)
    img_mask_RED = cv.bitwise_or(img_mask_r2, temp)

    img_mask_y1 = cv.inRange(img_hsv, lower_yellow1, upper_yellow1)
    img_mask_y2 = cv.inRange(img_hsv, lower_yellow2, upper_yellow2)
    img_mask_y3 = cv.inRange(img_hsv, lower_yellow3, upper_yellow3)
    temp = cv.bitwise_or(img_mask_y1, img_mask_y2)
    img_mask_YELLOW = cv.bitwise_or(temp, img_mask_y3)

    img_mask_g1 = cv.inRange(img_hsv, lower_green1, upper_green1)
    img_mask_g2 = cv.inRange(img_hsv, lower_green2, upper_green2)
    img_mask_g3 = cv.inRange(img_hsv, lower_green3, upper_green3)
    temp = cv.bitwise_or(img_mask_g1, img_mask_g2)
    img_mask_GREEN = cv.bitwise_or(temp, img_mask_g3)

        # 모폴로지 연산
    kernel = np.ones((11, 11), np.uint8)
    img_mask_RED = cv.morphologyEx(img_mask_RED, cv.MORPH_OPEN, kernel)
    img_mask_RED = cv.morphologyEx(img_mask_RED, cv.MORPH_CLOSE, kernel)

    kernel = np.ones((11, 11), np.uint8)
    img_mask_YELLOW = cv.morphologyEx(img_mask_YELLOW, cv.MORPH_OPEN, kernel)
    img_mask_YELLOW = cv.morphologyEx(img_mask_YELLOW, cv.MORPH_CLOSE, kernel)

    kernel = np.ones((11, 11), np.uint8)
    img_mask_GREEN = cv.morphologyEx(img_mask_GREEN, cv.MORPH_OPEN, kernel)
    img_mask_GREEN = cv.morphologyEx(img_mask_GREEN, cv.MORPH_CLOSE, kernel)

    # 마스크 이미지로 원본 이미지에서 범위값에 해당되는 영상 부분을 획득합니다.
    img_mask_to = cv.bitwise_or(img_mask_RED, img_mask_YELLOW)
    img_mask_total = cv.bitwise_or(img_mask_to, img_mask_GREEN)
#        img_result = cv.bitwise_and(img_color, img_color, mask=img_mask_total)

    contours, hierarchy = cv.findContours(img_mask_total, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    times = 0
    for ssung_cnt in contours:
        contour_center = False
        cv.drawContours(img_color, [ssung_cnt], 0, (255, 0, 0), 2)
        area = cv.contourArea(ssung_cnt)
        if area > 120:
            contour_center = True
            M = cv.moments(ssung_cnt)

            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            color_hsv = img_color[cy, cx]

            one_pixel = np.uint8([[color_hsv]])
            hsv = cv.cvtColor(one_pixel, cv.COLOR_BGR2HSV)
            color_h = hsv[0][0][0]
            times += 1
            # cv.circle(img_color, (cx, cy), 3, (255, 255, 255), -1)
    color = 'None'
    if contour_center and times == 1:
        if color_h < 6 or color_h > 165:             #red
            green_check = []
            color = 'RED'
            cv.putText(img_color, 'RED', location, font, fontScale, (0, 0, 255), 2)
        elif color_h in range(6, 35):           #orange
            green_check = []
            color = 'YELLOW'
            cv.putText(img_color, 'YELLOW', location, font, fontScale, (15, 255, 255), 2)
        elif color_h in range(35, 70):        #green
            while j < 100000:
                j += 1
            x, y, w, h = cv.boundingRect(ssung_cnt)  # 경계 사각형
            cv.rectangle(img_color, (x, y), (x + w, y + h), (15, 255, 255), 1)
            green_check.append(w)
            if len(green_check) == 5:
                minimum = np.min(green_check)
                maximum = np.max(green_check)
                if minimum > 30:
                    color = 'GREEN'
                    cv.putText(img_color, 'RIGHT_GREEN', location, font, fontScale, (0, 255, 0), 2)
                    green_check = []
                elif maximum < 27:
                    color = 'LEFT'
                    cv.putText(img_color, 'LEFT_GREEN', location, font, fontScale, (0, 255, 0), 2)
                    green_check = []
                else:
                    green_check = []

    elif times > 1:
        cv.putText(img_color, 'wait', location, font, fontScale, (255, 255, 255), 2)


        # cv.imshow('img_result', img_mask_total)

    cv.imshow('img_color', img_color)

################################################################################################


    if mode == 1:
        color_image = pygame.surfarray.array3d(screen)
        color_image = cv.transpose(color_image)
        color_image = cv.cvtColor(color_image, cv.COLOR_RGB2BGR)
        
        angle = car.rotate_angle*np.pi/180

        rt_angle = abs(angle-temp_angle)
        
        if rt_angle > 20/180*np.pi:
            car.rotate_angle = -temp_angle

        rot_matrix = [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
        plus_1 = [plus_x,plus_y]
        plus_2 = [-plus_x,plus_y]
        plus_3 = [-plus_x, -plus_y]
        plus_4 = [plus_x, -plus_y]
        
        #print(car.rect.center, car_2, -car.rotate_angle)
        plus_x1 = np.dot(rot_matrix[0],plus_1)
        plus_y1 = np.dot(rot_matrix[1],plus_1)
        plus_x2 = np.dot(rot_matrix[0],plus_2)
        plus_y2 = np.dot(rot_matrix[1],plus_2)
        plus_x3 = np.dot(rot_matrix[0],plus_3)
        plus_y3 = np.dot(rot_matrix[1],plus_3)
        plus_x4 = np.dot(rot_matrix[0],plus_4)
        plus_y4 = np.dot(rot_matrix[1],plus_4)


        car_1 = [(car.rect.center[0]+plus_x1), (car.rect.center[1]-plus_y1)]
        car_2 = [(car.rect.center[0]+plus_x2), (car.rect.center[1]-plus_y2)]
        car_3 = [(car.rect.center[0]+plus_x3), (car.rect.center[1]-plus_y3)]
        car_4 = [(car.rect.center[0]+plus_x4), (car.rect.center[1]-plus_y4)]


        pts1 = np.float32([car_2,car_3,car_1,car_4])
        # 좌표의 이동점
        pts2 = np.float32([[0, 0], [0, 200], [100, 0], [100, 200]])
        M = cv.getPerspectiveTransform(pts1, pts2)
        color_image = cv.warpPerspective(color_image, M, (100,200))
        temp_angle = angle
        
        blur = cv.GaussianBlur(color_image, (5, 5), 0)
        imgh = blur.shape[0]  # 720
        imgw = blur.shape[1]  # 1280
        hsv_2 = cv.cvtColor(blur, cv.COLOR_BGR2HSV)
        mask_m = cv.inRange(hsv_2, np.array([20,30,30]), np.array([40,255,255])) #마스크 색 변경
        roi_m = cv.bitwise_and(color_image, color_image, mask=mask_m)
        edges = cv.Canny(roi_m, 0, 250)

        div = 8
        midpoint = np.array([imgw//2,imgw//2,imgw//2,imgw//2,imgw//2,imgw//2,imgw//2,imgw//2])
        for i in range(div):
            div_l = edges[i*imgh//div:(i+1)*imgh//div,0:midpoint[i]]
            div_r = edges[i*imgh//div:(i+1)*imgh//div,midpoint[i]:imgw]
            indices_l = np.where(div_l != [0])
            indices_r = np.where(div_r != [0])
            avgl = 0 if len(indices_l[1])==0 else np.sum(indices_l[1])//len(indices_l[1])
            avgr = imgw if len(indices_r[1])==0 else np.sum(indices_r[1])//len(indices_r[1])+midpoint[i]
            midpoint[i] = (avgl+avgr)//2
            midpoint_y = (2*i+1)*imgh//2//div
        for i in range(len(midpoint)-1):
            cv.line(color_image,(midpoint[i],(2*i+1)*imgh//2//div),(midpoint[i+1],(2*i+3)*imgh//2//div), (0,0,255), 3)
            midpoint[i] = midpoint[i+1]
        
        diff_654 = int((-midpoint[1]+midpoint[0]))
        dif_angle = np.arctan2(3*imgh//8,diff_654*8//3//imgh)-np.pi//2
        cv.arrowedLine(color_image, (imgw//2-2*diff_654,8*imgh//16),(imgw//2+2*diff_654,4*imgh//16),(255,255,0),3)
        print(dif_angle)


        cv.imshow('Color', color_image)
        



        screen.fill(BLACK)
        screen.blit(light.image, light.rect)  
        pygame.draw.lines(screen, YELLOW, False, line.lpos, 3)
        pygame.draw.lines(screen, YELLOW, False, line.rpos, 3)
        pygame.draw.lines(screen, GRAY, False, pass_lpos, 3)
        pygame.draw.lines(screen, GRAY, False, pass_rpos, 3)
        if stop_car == 0:
            dot.move(line.lpos, line.rpos)
            
        elif stop_car == 1:
            if color == 'LEFT':
                line.lpos = line.lpos[:-9]
                line.rpos = line.rpos[:-9]
                stop_car = 0
                line.left_rotate()
                stopline.add(StopLine(line.lpos, line.rpos))
                
            elif color == 'GREEN':
                line.lpos = line.lpos[:-9]
                line.rpos = line.rpos[:-9]
                stop_car = 0
                line.right_rotate()
                stopline.add(StopLine(line.lpos, line.rpos))
                

        if color == 'GREEN':
            pygame.draw.circle(screen, GREEN, [size[0]-120, 38], 16, 0)
        elif color == 'YELLOW':
            pygame.draw.circle(screen, YELLOW, [size[0]-75, 38], 16, 0)
        elif color == 'RED':
            pygame.draw.circle(screen, RED, [size[0]-30, 38], 16, 0)
        elif color == 'LEFT':
            screen.blit(LEFT, LEFT_rect)

        car.move(dot.rect.x, dot.rect.y)
        
        screen.blit(car.image, car.rect)
        screen.blit(dot.image, dot.rect)
        stopline.draw(screen)
        stop = car.collide(stopline)
        if stop:
            stop.kill()
            if line.lpos[1] == line.end[-1]:
                if lines[ii] == 1:
                    print(cnt)
                    line.straight(cnt)
                elif lines[ii] == 2:
                    line.left_rotate()
                elif lines[ii] == 3:
                    line.right_rotate()
                elif lines[ii] == 4:
                    line.Tline()
                    stop_car = 1
                ii+=1
            pass_lpos.append(line.lpos[0])
            pass_rpos.append(line.rpos[0])
            del(line.lpos[0])
            del(line.rpos[0])
            if stop_car == 0:
                car.rotate(line.lpos)
                stopline.add(StopLine(line.lpos, line.rpos))
        cnt+=1

    else:
        screen.fill(WHITE) 
        pygame.draw.lines(screen, YELLOW, False, line.lpos, 3)
        pygame.draw.lines(screen, YELLOW, False, line.rpos, 3)
        screen.blit(car.image, car.rect)
 
    pygame.display.flip()

    
cap.release()
pygame.quit()
