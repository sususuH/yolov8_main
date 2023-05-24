from ultralytics import YOLO
import time

model = YOLO("/home/su/su/yolo/yolov8_tracking-master_main/yolov8/cfg/yolov8l.yaml")
# model.predict(source="/home/su/su/yolo/yolov8_tracking-master_main/runs/detect/bad_video_frame/cut1.mp4")

# for n in range(179)
#     r = n+0
#     str1 = f"/home/su/su/yolo/yolov8_tracking-master_main/runs/detect/bad_video_frame/cut{r}.mp4"
#     if r not in [13,21,22]:
#         model.predict(source=str1)
    
#     # print("10s前")
#     # time.sleep(10)
#     # print("10s后")
# print('done!')

# model.predict(source="/home/su/su/yolo/yolov8_tracking-master_main/runs/detect/a_yuan/cut19.mp4")
# model.predict(source="/home/su/su/20230403_frame/frame/cut0_1.jpg")

import os
files =  os.listdir("/home/su/su/20230403_frame/frame")
print(len(files))
for file in files:
    model.predict(source=f"/home/su/su/20230403_frame/frame/{file}")
print("done!")

