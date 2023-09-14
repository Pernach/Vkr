import argparse
import numpy as np
import os
import pandas as pd
import cv2
import ultralytics
from ultralytics import YOLO

parser = argparse.ArgumentParser(description="Parser", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("src", help="Source 0 - camera, 1 - video (labels have to be in folder /vkr)")
args = parser.parse_args()
config = vars(args)

color_yellow = (0,255,255)
color_red = (0,0,255)
color_green = (0,255,0)
color_black = (0,0,0)
text_place = (505,170)

# Работа либо с видеостримом, либо с уже полученным файлом

if(config['src'] == '0'):
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        try:          
            model = YOLO('yolov8n.pt')  # pretrained YOLOv8n model
            # Run batched inference on a list of images
            results = model.predict(frame, save=True, imgsz=640, conf=0.6)

            cv2.rectangle(frame, (500,140), (700,180), color_green, thickness=2, lineType=8, shift=0)
            cv2.putText(frame, "Weak traffic", text_place, cv2.FONT_HERSHEY_SIMPLEX, 1, color_black, 2)

            cv2.imshow('result', frame)
        except:
            cap.release()
            raise
    
        ch = cv2.waitKey(5)
        if ch == 27:
            break
else:
    cap = cv2.VideoCapture(config['src'])
    if (cap.isOpened() == False):
        print("Ошибка при открытии видеофайла")
    else:
        df_f = pd.DataFrame()
        columns = ['label', 'center_x', 'center_y', 'width', 'height', 'confidence']
        for file in os.listdir('vkr'):
            if file.endswith('txt'):
                filename = 'vkr/' + file
                df = pd.read_table(filename, sep = ' ', header = None)
                df.columns = columns
                numberframe = file.split(' ')[-1:][0].split('_')[-1].split('.')[0]
                df['frame'] = int(numberframe)
                df_f = pd.concat([df_f, df])
                df_f = df_f.sort_values(by = ['frame'], ascending = True)
                true_labels = [2, 5, 7]
                df_f_1 = df_f.loc[df_f['label'].isin(true_labels)]
                df_f_gb = df_f_1.groupby(['frame']).count()['label']
        
        # Получить информацию о частоте кадров

        fps = int(cap.get(5))
        print("Частота кадров: ", fps, "кадров в секунду")  

        # Получить количество кадров
        frame_count = cap.get(7)
        print("Количество кадров: ", frame_count)
        
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        frame_size = (frame_width,frame_height)
        print("Размер изображения: ", frame_size)
        # Инициализировать объект записи видео
        output = cv2.VideoWriter('output '+ config['src'], cv2.VideoWriter_fourcc('M','J','P','G'), fps, frame_size)

        frame_count_label = 0

        while True:
            ret, frame = cap.read()

            frame_count_label += 1
            # рисуем прямоугольник
            if (frame_count_label in df_f_gb.index):
                if (df_f_gb[frame_count_label] < 3):
                    cv2.rectangle(frame, (500,140), (700,180), color_green, thickness=2, lineType=8, shift=0)
                    cv2.putText(frame, "Weak traffic", text_place, cv2.FONT_HERSHEY_SIMPLEX, 1, color_black, 2)
                elif (df_f_gb[frame_count_label] < 5):
                    cv2.rectangle(frame, (465,140), (735,180), color_yellow, thickness=2, lineType=8, shift=0)
                    cv2.putText(frame, "Moderate traffic", (470,170), cv2.FONT_HERSHEY_SIMPLEX, 1, color_black, 2)
                else:
                    cv2.rectangle(frame, (500,140), (700,180), color_red, thickness=2, lineType=8, shift=0)
                    cv2.putText(frame, "Traffic jams", text_place, cv2.FONT_HERSHEY_SIMPLEX, 1, color_black, 2)
            else:
                cv2.rectangle(frame, (500,140), (700,180), color_green, thickness=2, lineType=8, shift=0)
                cv2.putText(frame, "Weak traffic", text_place, cv2.FONT_HERSHEY_SIMPLEX, 1, color_black, 2)
            
            
            if ret == True:
                # Записываем фрейм в выходные файлы
                output.write(frame)
            else:
                print("Поток отключен")
                break

    
cap.release()
output.release()
cv2.destroyAllWindows()    