# 🩺 OSA Diagnosis Web Application using ECG Signals

Welcome to a transformative medical AI solution designed for **early detection and severity prediction** of **Obstructive Sleep Apnea (OSA)** using ECG signals. Built with a robust **1D Convolutional Neural Network (CNN)** and wrapped in a user-friendly **Django web application**, this project empowers **doctors** to diagnose OSA with speed, precision, and ease.

---

## 🚀 Features

- 👨‍⚕️ **Doctor Authentication**: Secure signup and login system.
- 📝 **Patient Management**: Store and retrieve patient records efficiently.
- ⚖️ **BMI Calculation**: Automatically computes **Body Mass Index** from patient height and weight.
- 📈 **Upload ECG CSV**: Doctors upload ECG data files sampled at **100 Hz**.
- 🧠 **AI-Powered Diagnosis**: A trained 1D CNN predicts:
  - Presence of OSA
  - **Severity**: Mild | Moderate | Severe
- 📊 **Health Insights**: Explore the relationship between OSA severity and factors like **BMI, height, and weight**.
- 🔁 **Revisit Support**: For returning patients, just upload a new ECG file — no need to re-enter patient details.
- 📹 **Demo Included**: See how it works in a short demo video.

---

## 🧠 Model Architecture

Our 1D CNN was designed and trained specifically for ECG signals with a sampling frequency of **100 Hz**.

<p align="center">
  <img src="https://drive.google.com/file/d/1Ocs2iU5z7m6eHNE9uBF8CYlyuiGDCVVk/view?usp=drive_link" alt="Model Architecture" width="700"/>
</p>

---

## 📊 Training Performance

<p align="center">
  <img src="https://drive.google.com/file/d/1B-U95HLJdz9WSsGtVEDiV8jqN0CFQIui/view?usp=drive_link" alt="Training Plot" width="700"/>
</p>

> **Note**: The model was trained on labeled ECG datasets and validated for high precision and reliability in OSA classification.

---
## 📋 Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.92      | 0.93   | 0.92     | 2801    |
| 1     | 0.89      | 0.88   | 0.88     | 1913    |
| **Accuracy** |       |        | **0.91** | **4714** |

---

## 💾 Model & Demo Resources

- 🔗 **Trained Model**: [Download from Google Drive](https://drive.google.com/file/d/1j1wLkALEAPzME3Mai_PAcUVM-fTtBhAZ/view?usp=drive_link)
- 📽️ **Full Demo Video (Django Web App)**: [Watch on Google Drive](https://drive.google.com/file/d/1pQ66heM4wFee2g9uJoUKD5EKA6uIgEZ6/view?usp=drive_link)

---

