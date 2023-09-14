import torchvision
import cv2
import torch
import argparse
import time
import detect_utils
import makedir

color_yellow = (0,255,255)
color_red = (0,0,255)
color_green = (0,255,0)
color_black = (0,0,0)
text_place = (505,170)

true_labels = [3, 6, 8]

# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='path to input video')
args = vars(parser.parse_args())

# define the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Загрузка модели
model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
# Загрузка модели на рабочую машину (cpu, или gpu)
model = model.eval().to(device)
# Открываем видео
cap = cv2.VideoCapture(args['input'])
# Проверка на открытие файла
if (cap.isOpened() == False):
    print('Error while trying to read video. Please check path again')

# Данные о высоте и ширине изображения
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

save_name = f"{args['input'].split('/')[-1].split('.')[0]}"

# Создаем папку для выхода
output_directory = "outputs"
makedir.makeoutputdir(output_directory)

# define codec and create VideoWriter object
out = cv2.VideoWriter(f"{output_directory}/{save_name}.mp4",
                      cv2.VideoWriter_fourcc(*'mp4v'), 30,
                      (frame_width, frame_height))

frame_count = 0 # to count total frames
total_fps = 0 # to get the final frames per second
labels_q_f = 0
# read until end of video
while(cap.isOpened()):
    # capture each frame of the video
    ret, frame = cap.read()
    if ret == True:
        # get the start time
        start_time = time.time()
        with torch.no_grad():
            # get predictions for the current frame
            boxes, classes, labels, scrores = detect_utils.predict(frame, model, device, 0.7)

        # draw boxes and show current frame on screen
        image = detect_utils.draw_boxes(boxes, classes, labels, frame, scrores)

        # get the end time
        end_time = time.time()
        # get the fps
        fps = 1 / (end_time - start_time)
        # add fps to total fps
        total_fps += fps
        # increment frame count
        frame_count += 1
        # Фильтр первого порядка
        labels_q = len([x for x in labels if x in true_labels])
        labels_q = labels_q / 7
        Td = end_time - start_time
        Tf = 6
        b0 = Td / Tf
        a1 = 1 - b0
        labels_q_f = b0 * labels_q + a1 * labels_q_f
        print(f"Frame counter: {frame_count}, FPS: {fps:.0f}, labels: {labels_q_f:.0f}")
        # write the FPS on the current frame
        cv2.putText(image, f"{fps:.3f} FPS", (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)


        #Put text about traffic
        if (labels_q_f < 3):
          cv2.rectangle(image, (500,140), (700,180), color_green, thickness=2, lineType=8, shift=0)
          cv2.putText(image, "Weak traffic", text_place, cv2.FONT_HERSHEY_SIMPLEX, 1, color_black, 2)
        elif (labels_q_f < 5):
          cv2.rectangle(image, (465,140), (735,180), color_yellow, thickness=2, lineType=8, shift=0)
          cv2.putText(image, "Moderate traffic", (470,170), cv2.FONT_HERSHEY_SIMPLEX, 1, color_black, 2)
        else:
          cv2.rectangle(image, (500,140), (700,180), color_red, thickness=2, lineType=8, shift=0)
          cv2.putText(image, "Traffic jams", text_place, cv2.FONT_HERSHEY_SIMPLEX, 1, color_black, 2)

        # press `q` to exit
        wait_time = max(1, int(fps/4))
        # convert from BGR to RGB color format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # cv2.imshow('image', image)
        out.write(image)
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break

    else:
        break

# release VideoCapture()
cap.release()
# close all frames and video windows
cv2.destroyAllWindows()

# calculate and print the average FPS
avg_fps = total_fps / frame_count
print(f"Average FPS: {avg_fps:.3f}")