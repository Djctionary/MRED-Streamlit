# MRED-Streamlit: Multimodal Real-Time Emotion Detection

MRED-Streamlit is an advanced multimodal online emotion detection system, implemented using **Streamlit**. The project integrates multiple modalities to analyze and identify emotions in real-time. Currently, the system supports **audio** and **image** modalities, with **brainwave analysis** planned for future updates.

## Features and Modalities

### 1. **Audio Modality**
- **Objective**: Perform **PAD (Pleasure-Arousal-Dominance)** emotion recognition.
- **Model**: Utilizes the [Wav2Vec2 (W2V2) framework](https://github.com/audeering/w2v2-how-to.git) for speech feature extraction and analysis.
- **Pipeline**: Processes audio data to predict emotional states based on voice characteristics.
- 
![屏幕截图 2024-11-30 115756](https://github.com/user-attachments/assets/de9c61a0-8cf4-44b6-99f8-92f55b56ec77)

### 2. **Image Modality**
- **Objective**: Real-time facial emotion recognition through expression analysis.
- **Key Tools**:
  - **Facial landmark detection**: Leverages **dlib** for precise face and feature point detection.
  - **Emotion analysis**: Integrates methods from [facial-expression-analysis](https://github.com/bbonik/facial-expression-analysis.git).
- **Pipeline**: Captures live facial data via webcam, identifies facial expressions, and maps them to emotions.
- 
![屏幕截图 2024-11-30 115820](https://github.com/user-attachments/assets/a639c176-d576-47ef-a4e9-c4a385c86bae)

### 3. **Brainwave Modality** (Future Development)
- **Objective**: Enhance multimodal emotion detection with EEG data.
- **Challenges**: Requires specialized equipment for brainwave data collection.
- **Current Status**: Not open-sourced due to hardware dependencies.

---

## Installation and Setup

### Prerequisites
- Python 3.8+
- Clone the repository and ensure the required dependencies are installed.

### Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/Djctionary/MRED-Streamlit.git
   cd MRED-Streamlit
   ```
2. Install the environment using the provided `environment.yml` file:
   ```bash
   conda env create -f environment.yml
   conda activate streamlit
   ```
3. Download and Configure Models
  1). Download the models from the following link:  
     [Google Drive - MRED Models](https://drive.google.com/file/d/1iMXDkwCtMvlMREYVAWWOxqK3xiSOVbUr/view?usp=drive_link)
  
  2). Extract the downloaded file and move the **`models`** folder to the root directory of the project.
  
  3). Ensure the directory structure is as follows:
     ```
     MRED-Streamlit/
     ├── app.py
     ├── models/
     ├── environment.yml
     ├── ...
     ```
3. Launch the Streamlit application:
   ```bash
   streamlit run app.py
   ```

---

## Future Plans
- **Brainwave Modality**: Integrate real-time EEG data for deeper emotion analysis.
- **Multimodal Fusion**: Combine audio, image, and EEG modalities for comprehensive emotion detection.

---

## References
- [Wav2Vec2 Emotion Analysis](https://github.com/audeering/w2v2-how-to.git)
- [Facial Expression Analysis](https://github.com/bbonik/facial-expression-analysis.git)

Feel free to contribute or open issues for suggestions!
