import io
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

def process_eeg():
    return get_example_image()

def get_example_image():
    # 生成一个示例图像流
    buffer = io.BytesIO()
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.set_title("示例 EEG 雷达图", color='#3333FF', fontsize=16)
    plt.savefig(buffer, format='png', transparent=True)
    buffer.seek(0)
    plt.close(fig)
    return buffer
