import cv2
import torch
from streamlit import upload_file, image, write
from yolo_deepsort import count_and_track_objects

def main():
    # StreamlitのUI
    uploaded_file = image("Upload an image", use_column_width=True)

    if uploaded_file is not None:
        # 画像を読み込んでカウントとトラッキング
        image_array = cv2.imread(uploaded_file.name)
        count, tracked_image = count_and_track_objects(image_array)

        # カウント結果の表示
        write("Number of objects:", count)

        # トラッキング結果の表示
        image(tracked_image, use_column_width=True)

if __name__ == "__main__":
    main()
