import random
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import io

from matplotlib import rcParams


def process_eeg():
    delight = generate_value()
    dominance = generate_value()
    alert = generate_value()
    clear = generate_value()
    nervous = generate_value()
    attention = generate_value()
    meditation = generate_value()

    return radar(delight, nervous, alert, dominance, clear, attention, meditation)

def generate_value():
    if random.random() < 0.8:  # 80%的概率生成平稳数据
        return random.randint(40, 60)
    else:  # 20%的概率生成大起伏数据
        return random.randint(0, 100)

def radar(delight, nervous, alert, dominance, clear, attention, meditation):
    values = [delight, nervous, alert, dominance, clear, attention, meditation]
    feature = ["快感", "紧张", "警觉", "统御", "清醒", "注意力", "冥想", "快感"]
    angles = np.linspace(0, 2 * np.pi, len(values), endpoint=False)

    values = np.concatenate((values, [values[0]]))
    angles = np.concatenate((angles, [angles[0]]))

    rcParams['font.sans-serif'] = ['SimSun']  # 使用宋体
    rcParams['axes.unicode_minus'] = False  # 防止负号显示异常

    # 创建雷达图
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, 'o-', linewidth=2)
    ax.fill(angles, values, alpha=0.25)

    ax.set_thetagrids(angles * 180 / np.pi, feature, color='#00008B', weight='bold', fontsize=14)
    ax.set_ylim(0, 100)
    ax.tick_params(axis='y', colors='#6666FF')
    plt.title('EEG', color='#3333FF', fontsize=16)

    ax.grid(True)

    # 将图像保存到字节流
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', transparent=True)
    buffer.seek(0)
    plt.close(fig)

    return buffer