import threading
import queue
import time
from collections import deque
import audonnx
import streamlit as st
import numpy as np
import sounddevice as sd
import cv2
import dlib
import os
import logging
from logging.handlers import RotatingFileHandler

from matplotlib import pyplot as plt
from matplotlib import rcParams
from streamlit_echarts import st_echarts

from components import process_video_frame, process_audio_chunk
from components.process_video_frame import plot_emotion_history_echarts
from utils import EmotionsDlib, plot_landmarks


# 设置日志
def setup_logger(log_dir="logs", log_level=logging.INFO):
    """
    设置日志记录器
    :param log_dir: 日志文件目录
    :param log_level: 日志级别
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger()
    logger.setLevel(log_level)

    # 日志格式
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(threadName)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 控制台日志处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件日志处理器
    file_handler = logging.FileHandler(os.path.join(log_dir, "application.log"), encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info("日志系统初始化完成")


class GlobalState:
    def __init__(self):
        self.image_queue = None
        self.processed_image_queue = None
        self.cap = None
        self.detector = None
        self.predictor = None
        self.emotion_estimator = None
        self.audio_model = None
        self.video_running = threading.Event()
        self.max_history = 20
        self.emotions_history = deque(maxlen=self.max_history)

    def initialize(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        shape_predictor_path = os.path.join(base_dir,  "models/shape_predictor_68_face_landmarks.dat")
        emotion_model_path = os.path.join(base_dir, "models/model_emotion_pls=31_fullfeatures=False.joblib")
        frontalization_model_path = os.path.join(base_dir, "models/model_frontalization.npy")
        audio_model_path = os.path.join(base_dir, "models/audio_model")

        self.image_queue = queue.Queue(maxsize=3)
        self.processed_image_queue = queue.Queue(maxsize=3)
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(shape_predictor_path)
        self.emotion_estimator = EmotionsDlib(
            file_emotion_model=emotion_model_path,
            file_frontalization_model=frontalization_model_path
        )
        self.audio_model = audonnx.load(audio_model_path)
        self.video_running.clear()
        logging.info("GlobalState 初始化完成")

    def update_emotions(self, new_emotion):
        """
        更新情绪历史记录，超出 maxlen 时删除最早的数据。
        :param new_emotion: dict, 包含 emotion_name, arousal, valence, intensity 的情绪数据
        """
        self.emotions_history.append(new_emotion)

    def get_emotions_history(self):
        """
        获取所有的情绪历史记录。
        :return: list, 历史记录
        """
        return list(self.emotions_history)


def audio_waveform():
    global_state = st.session_state.global_state
    duration = 3  # seconds
    samplerate = 16000  # Hz
    st.write("### 录制音频并分析情绪")

    if st.button("开始录音"):
        logging.info("开始录音...")

        with st.spinner("正在录音和处理音频，请稍候..."):
            recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
            sd.wait()  # Wait until recording is finished
        st.write("录音完成！")
        logging.info("录音完成")

        audio_results = process_audio_chunk(recording, sample_rate=samplerate, audio_model=global_state.audio_model)

        st.write("### 情感分析结果")
        st.json(audio_results)

        rcParams['font.sans-serif'] = ['SimSun']  # 或 ['Microsoft YaHei']
        rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

        # 绘制波形图
        fig, ax = plt.subplots()
        ax.plot(recording)
        ax.set_xlabel('样本')
        ax.set_ylabel('幅度')
        ax.set_title('音频波形')
        st.pyplot(fig)


def video_capture_thread(global_state):
    logging.info("视频捕获线程启动")
    while global_state.video_running.is_set():
        ret, frame = global_state.cap.read()
        if ret:
            if global_state.image_queue.full():
                global_state.image_queue.get_nowait()
            global_state.image_queue.put_nowait(frame)
        else:
            logging.error("无法捕获摄像头帧")
            break
    global_state.cap.release()
    logging.info("视频捕获线程停止")


def video_processing_thread(global_state):
    logging.info("视频处理线程启动")
    while global_state.video_running.is_set():
        if not global_state.image_queue.empty():
            frame = global_state.image_queue.get()
            processed_frame, emotions = process_video_frame(
                frame, global_state.detector, global_state.predictor, global_state.emotion_estimator
            )
            if emotions:
                global_state.update_emotions(emotions)
            if global_state.processed_image_queue.full():
                global_state.processed_image_queue.get_nowait()
            if processed_frame:
                global_state.processed_image_queue.put_nowait(processed_frame)
    logging.info("视频处理线程停止")


def video_frame_processor():
    global_state = st.session_state.global_state
    st.write("### 实时面部情绪分析")

    # 创建两列，分别显示原始视频流和处理后的视频流
    col1, col2 = st.columns([1, 1])  # 1:1 比例，可以根据需要调整

    with col1:
        st.write("### 原始视频流")
        original_frame_window = st.image([])

    with col2:
        st.write("### 面部关键点图像")
        processed_frame_window = st.image([])

        # 动态显示情绪和图表
    emotion_placeholder = st.empty()
    plot_placeholder = st.empty()

    def toggle_run():
        if global_state.video_running.is_set():
            logging.info("停止视频捕获和处理")
            global_state.video_running.clear()
        else:
            logging.info("启动视频捕获和处理")
            global_state.video_running.set()
            threading.Thread(target=video_capture_thread, args=(global_state,), daemon=True).start()
            threading.Thread(target=video_processing_thread, args=(global_state,), daemon=True).start()
            st.rerun()

    st.button("开始分析 / 停止分析", on_click=toggle_run)

    if global_state.video_running.is_set():
        # 更新情绪
        if global_state.emotions_history:
            emotion_placeholder.markdown(f"### 当前预测的情绪为：{global_state.emotions_history[-1]['emotion_name']}")

        # 更新视频流
        if global_state.image_queue:
            frame = global_state.image_queue.get()
            original_frame_window.image(frame, channels="BGR")

        if global_state.processed_image_queue:
            processed_frame = global_state.processed_image_queue.get()
            processed_frame_window.image(processed_frame, channels="BGR")

        # 更新图表
        if global_state.emotions_history:
            options = plot_emotion_history_echarts(global_state.emotions_history)
            plot_placeholder = st_echarts(options=options, height="400px", key="dynamic_emotion_chart")

        time.sleep(0.1)
        st.rerun()

def initialize():
    if "global_state" not in st.session_state:
        setup_logger()  # 初始化日志
        start_time = time.time()
        st.session_state.global_state = GlobalState()
        st.session_state.global_state.initialize()
        init_time = time.time() - start_time
        logging.info(f"初始化时间为：{init_time:.3f}秒")
    else:
        logging.info("GlobalState 已经初始化，无需重复初始化")


def main():
    initialize()

    st.title("多模态情绪分析系统")

    st.sidebar.title("选择功能")
    app_mode = st.sidebar.selectbox("请选择模式", ["音频情绪分析", "面部情绪分析"])

    if app_mode == "音频情绪分析":
        audio_waveform()
    elif app_mode == "面部情绪分析":
        video_frame_processor()


if __name__ == "__main__":
    main()
