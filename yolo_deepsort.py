import cv2
import torch
from deep_sort import build_tracker
from deep_sort import DeepSort
from deep_sort.tracker import Tracker
from deep_sort.detection import Detection
from deep_sort.utils.parser import get_config
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression
from yolov5.utils.torch_utils import select_device

# YOLOv5 + DeepSort の設定ファイルを読み込む
yolov5_deepsort_cfg = "path/to/your/config.yaml"
deep_sort_cfg = get_config()

# YOLOv5モデルの読み込み
def load_yolov5_model():
    model = attempt_load(yolov5_deepsort_cfg, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return model

# DeepSortのトラッカーのビルド
def build_deepsort_tracker():
    device = select_device("")
    tracker = build_tracker(deep_sort_cfg, use_cuda=torch.cuda.is_available())
    return tracker

# カウントする対象のクラスのインデックス
target_class_index = 0  # 例えば、0は人を表すクラスのインデックス

def count_and_track_objects(frame):
    # YOLOv5による物体検出
    img = torch.from_numpy(frame).cuda().float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    pred = model(img)[0]

    # 物体のクラスインデックスを取得
    pred_class = pred[:, -1].cpu().numpy().astype(int)
    target_indices = pred_class == target_class_index

    # DeepSortの入力用に物体の検出結果を整形
    detections = non_max_suppression(pred[target_indices], conf_thres=0.3, iou_thres=0.45)

    if detections[0] is not None:
        tracked_objects = []
        for x1, y1, x2, y2, conf, _ in detections[0]:
            box = [x1, y1, x2, y2]
            confidence = conf
            cls_id = target_class_index
            detection = Detection(box, confidence, cls_id)
            tracked_objects.append(detection)

        # DeepSortによる物体追跡
        tracker.predict()
        tracker.update(tracked_objects)

        # 物体のカウント
        count = len(tracker.tracks)

        # トラッキング結果を描画
        tracked_image = frame.copy()
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr().astype(int)
            cv2.rectangle(tracked_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

        return count, tracked_image

    else:
        return 0, frame

# YOLOv5モデルとDeepSortトラッカーの読み込み
model = load_yolov5_model()
tracker = build_deepsort_tracker()
