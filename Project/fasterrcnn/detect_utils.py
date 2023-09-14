import torchvision.transforms as transforms
import cv2
import numpy as np

from coco_names import COCO_INSTANCE_CATEGORY_NAMES as coco_names

# функция случайно меняет цвета на выходе
COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))

# Преобразование изображения в необходимый для библиотек вид
transform = transforms.Compose([
    transforms.ToTensor(),
])

def predict(image, model, device, detection_threshold):
    # Преобразование изображения в тензор
    image = transform(image).to(device)
    image = image.unsqueeze(0) # размер батча
    outputs = model(image) # получение прогноза по картинке

    # получение значений классов
    pred_classes = [coco_names[i] for i in outputs[0]['labels'].cpu().numpy()]

    # получение значений уверенностей
    pred_scores = outputs[0]['scores'].detach().cpu().numpy()

    # получение значений рамок
    pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
    # срез рамок по уровню уверенностей по умолучанию
    boxes = pred_bboxes[pred_scores >= detection_threshold].astype(np.int32)
    # срез уверенностей по уровеню уверенностей по умолчанию
    scoresd = pred_scores[pred_scores >= detection_threshold]

    return boxes, pred_classes, outputs[0]['labels'], scoresd

def draw_boxes(boxes, classes, labels, image, scoresd):
    # Отрисовка картинки
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)
    for i, box in enumerate(boxes):
        color = COLORS[labels[i]]
        # Отрисовка рамки
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color, 2
        )
        # Отрисковка текста класса и уверенности прогноза
        cv2.putText(image, classes[i] + f" {scoresd[i]:.2f}", (int(box[0]), int(box[1]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
                    lineType=cv2.LINE_AA)
    return image
