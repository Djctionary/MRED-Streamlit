# Streamlit日志

Streamlit 本质是一个同步单线程框架，适合机器学习模型展示、数据分析仪表盘等。

**特点：**

* 每次用户交互都会重新从头到尾运行
* 支持多种数据处理库，如Pandas；并支持读取和可视化CSV、Excel、JSON 等多种数据格式
* 高效交互和可视化

**多模态迁移问题：**

* Streamlit本身不支持多线程，只允许在主线程中渲染UI
* 对实时处理的可操作性不足（while, st.rerun）
* 超过100mb的模型无法部署在Streamlit云中
* Streamlit - webrtc存在bug