# **Smart-Safe: CCTV Camera Monitoring System**

Smart-Safe is a CCTV monitoring system equipped with **human activity detection** to recognize basic actions like **walking, running, and standing**. It includes a **facial recognition database** for tracking and storing faces, making it ideal for **schools and factories** to help administrators monitor attendance, ensure security, and improve workflow.

---

## 🚀 Steps to Run:

1. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
2. Run the application:
   ```sh
   python main.py
   ```

---

## 🔥 Features:

✅ **Face Verification** - Identifies admins and individuals, tracking their activity and displaying results on the dashboard.
✅ **Activity Detection** - Detects human actions using pose estimation and deep learning models.

---

## 🛠 Workflow:

1. **Face Verification:**
   - A **CNN-based model** encodes faces and compares them against a stored database.
   - Matches are determined using similarity scores.

2. **Activity Detection:**
   - Uses **pose detection** to extract key body points.
   - An **LSTM network** predicts actions.
   - Identifies the individual performing the action through facial recognition.

---

## 📌 To-Do List:

- [ ] **Enhance Object Tracking:** Implement **DeepSORT** or **StrongSORT** for efficient tracking. The current CNN-based tracking is **CPU-heavy**.
- [ ] **Dashboard Development:** Build a dashboard to group activities by individuals along with timestamps.
- [ ] **Advanced Activity Detection:** Implement a **CNN-RNN** or **Vision Transformer-based** model to improve spatial and temporal feature extraction for **better accuracy**.

---

🎯 *Smart-Safe aims to revolutionize surveillance by integrating AI-driven face recognition and activity monitoring for real-time analysis and security!* 🚀
