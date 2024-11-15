import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import io

from streamlit_echarts import st_echarts

from utils import plot_landmarks


def process_video_frame(frame, detector, predictor, emotion_estimator):
    """
    处理视频帧中的人脸检测和关键点绘制，生成裁剪的人脸图像和Frontal Landmarks图。

    Parameters:
    - frame: np.ndarray, 当前视频帧
    - detector: 面部推理模型
    - predictor: 关键点预测模型

    Returns:
    - processed_Frontal_stream: BytesIO, 处理后的Frontal Landmarks图像
    """

    # 验证 frame 是否有效
    if frame is None or frame.size == 0:
        raise ValueError("Frame is empty or invalid, check video source.")

    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)

    # 转换为灰度图像
    try:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        raise RuntimeError(f"Error converting frame to grayscale: {e}")

    # 验证 image 数据类型
    if image.dtype != np.uint8:
        raise RuntimeError("Image must be 8bit gray or RGB.")

    # 人脸检测
    faces = detector(image)
    if len(faces) == 0:
        return None, None

    # 选择最大人脸
    if len(faces) > 1:
        face_sizes = [(face.bottom() - face.top()) * (face.right() - face.left()) for face in faces]
        idx_largest_face = np.argmax(face_sizes)
    else:
        idx_largest_face = 0

    face = faces[idx_largest_face]

    # 人脸关键点检测
    try:
        landmarks_object = predictor(frame, face)
        dict_emotions = emotion_estimator.get_emotions(  # Get emotions
            landmarks_object)
        landmarks = dict_emotions['landmarks']['raw']
        landmarks_frontal = dict_emotions['landmarks']['frontal']
        arousal = dict_emotions['emotions']['arousal']
        valence = dict_emotions['emotions']['valence']
        intensity = dict_emotions['emotions']['intensity']
        emotion_name = dict_emotions['emotions']['name']
    except Exception as e:
        raise RuntimeError(f"Error in landmark detection: {e}")

    # 提取关键点坐标
    landmarks = np.array([(landmarks_object.part(i).x, landmarks_object.part(i).y)
                          for i in range(landmarks_object.num_parts)])

    # 生成 Frontal Landmarks 图像
    with plt.rc_context({'interactive': False}):
        fig_frontal, ax_frontal = plt.subplots(dpi=50)
        plot_landmarks(landmarks, axis=ax_frontal, title='Frontal Landmarks')
        fig_frontal.canvas.draw()

        # 转换为 JPEG 格式并保存为 BytesIO
        img_data = np.frombuffer(fig_frontal.canvas.tostring_rgb(), dtype=np.uint8)
        img_data = img_data.reshape(fig_frontal.canvas.get_width_height()[::-1] + (3,))
        is_success, buffer = cv2.imencode(".jpg", img_data, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not is_success:
            raise RuntimeError("Failed to encode frontal landmarks image.")

        processed_Frontal_stream = io.BytesIO(buffer)
        plt.close(fig_frontal)

    emotions = {
        "emotion_name": emotion_name,
        "arousal": arousal,
        "valence": valence,
        "intensity": intensity
    }

    return processed_Frontal_stream, emotions


def plot_emotion_history_echarts(emotions_history):
    if not emotions_history:
        return None

    arousal = [e["arousal"] for e in emotions_history]
    valence = [e["valence"] for e in emotions_history]
    intensity = [e["intensity"] for e in emotions_history]
    x_data = list(range(len(emotions_history)))

    options = {
        "title": {"text": "Dynamic Emotion History"},
        "tooltip": {"trigger": "axis"},
        "xAxis": {"type": "category", "data": x_data},
        "yAxis": {"type": "value"},
        "series": [
            {"name": "Arousal", "type": "line", "data": arousal},
            {"name": "Valence", "type": "line", "data": valence},
            {"name": "Intensity", "type": "line", "data": intensity},
        ],
    }
    return options
