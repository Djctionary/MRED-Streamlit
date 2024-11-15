import time
import audresample
import numpy as np


def process_audio_chunk(chunk, sample_rate, audio_model):

    # 如果是双声道数据，取平均值变成单声道
    if len(chunk.shape) == 2 and chunk.shape[1] > 1:
        chunk = np.mean(chunk, axis=1)

    # 重采样到16000Hz
    if sample_rate != 16000:
        chunk = audresample.resample(chunk, sample_rate, 16000)

    # 调用模型进行预测
    pred = audio_model(chunk, 16000)

    # 映射到 -3 到 3
    arousal = 6 * pred['logits'][0][0] - 3
    dominance = 6 * pred['logits'][0][1] - 3
    valence = 6 * pred['logits'][0][2] - 3

    emotion_scores = {
        "arousal": arousal,
        "dominance": dominance,
        "valence": valence
    }

    # 返回字典
    return emotion_scores
