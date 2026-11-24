

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-Web_App-lightgrey?logo=flask)](https://flask.palletsprojects.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Image_Processing-green?logo=opencv)](https://opencv.org/)
[![TailwindCSS](https://img.shields.io/badge/Tailwind_CSS-Styling-38B2AC?logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)



# 🧠 Brain Tumor MRI Segmentation using Deep Learning & Local LLM Reports

This project presents a full pipeline for **automatic segmentation of brain tumors** from MRI images. It combines **deep learning**, **medical image processing**, and a **local LLM-based report generation system** – all integrated into a user-friendly web interface.

## 📌 Objectives

- Automatically detect and segment tumor regions in brain MRIs
- Offer an intuitive web interface for image upload and mask visualization
- Generate **automated medical reports** using a local large language model (LLM)

## ⚙️ Technologies Used

- **Main language:** Python
- **Deep Learning:** TensorFlow / Keras (U-Net model)
- **Image processing:** NumPy, OpenCV, Pillow
- **Frontend:** HTML + Tailwind CSS
- **Backend:** Flask
- **Local LLM:** Ollama + LLaMA Phi

## 📚 Datasets

- [LGG MRI Segmentation Dataset (Kaggle)](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) MRI Images + masks
- [Brain Tumor MRI Dataset ](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (MRI Images with no masks)

## 🧪 Project Workflow

1. **MRI Preprocessing**
   - All images resized to 256×256
   - Denoising using Gaussian blur
   - Contrast enhancement (CLAHE & histogram equalization)

2. **Binary Mask Generation for 2nd DataSet**
- Manual thresholding (after experimenting with Otsu and adaptive)


![Test filters for masks](https://i.imgur.com/JDztdGl.png)


- Post-cleaning of masks 
   

![Test filters for masks](https://i.imgur.com/syGRsvY.png)

3. **U-Net Model Training**
   - Classic U-Net architecture (encoder + decoder + skip connections)

![U-Net Architecture](https://i.imgur.com/EeZ9bIl.jpeg)

   - Loss: Dice Loss
   - Metrics: IoU, Dice Coefficient
<p align="center">
<img  src="https://i.imgur.com/rmFnmfz.png" alt="U-Net Architecture" width="200" /></p>


   - Trained using Kaggle Notebook with GPU (Tesla T4)

4. **Web App Integration with Flask**
   - Upload MRI → Predict mask → Display overlay
   - Generate a clinical report using a local LLM via Ollama

5. **Edge Case Handling**
   - Case of no tumor → report indicates normal brain image
   - Full visualization of original + overlay + generated report

## 🖥️ Getting Started Locally

### 1. Clone the repo

```bash
git clone https://github.com/ana3ss7z/Brain-Tumor-Segmentation-and-LLM-based-report-generation-system.git
cd brain-segmentation
````

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### 3. Install the dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Flask app

```bash
python app.py
```

The app will be available at `http://127.0.0.1:5000/`

### 5. Start the local LLM with Ollama

#### 🔧 Setup Instructions

1. **Install Ollama**

   Follow the official guide:  
   👉 https://ollama.com/download



2. **Pull the Phi model**

   Once installed, download the `phi` model (or another like `mistral`, `llama2`, etc.):

   ```bash
   ollama pull phi
   ```

3. **Run the LLM in background**

   ```bash
   ollama run phi
   ```

   This will start a local HTTP API that your Flask backend can call to generate structured medical reports.

---

### 🧪 How It’s Used in This Project

- After brain MRI segmentation, extracted features (e.g., tumor size, shape, location) are passed to the LLM in JSON format.
- The LLM then generates a radiology-style report with:
  - ✅ Tumor presence and confidence
  - 📏 Approximate area
  - 🧬 Morphological description
  - 🧠 Anatomical location

All processing happens **locally** — no sensitive medical data leaves your machine.

---

### 🔐 Why Ollama?

- ✔️ **Privacy-first**: No cloud dependency
- ⚙️ **Customizable**: Use any local model (`phi`, `mistral`, `llama2`, etc.)
- 🖥️ **Lightweight**: Efficient on CPU with small models like `phi`
- 🔌 **API Ready**: Seamlessly integrates with Python/Flask backends

---

```bash
ollama run phi  # or another local model (e.g., mistral, llama2)
```

## 📸 Sample Output

| Original MRI            | Predicted Mask         | Overlay               |
| ----------------------- | ---------------------- | -------------------------------- |
| ![](https://i.imgur.com/M9lLWTS.png) | ![](https://i.imgur.com/RhxxDpT.png) | ![](https://i.imgur.com/wmMnCkl.png) |


---


## 📷 Execution Screenshots

Here are some screenshots showing the web app in action during typical usage:

### 1. Web App Homepage  
![Homepage](https://i.imgur.com/b0bvAX3.png)  
*Simple upload interface with drag-and-drop and file selection.*

### 2. Uploading an MRI Image  
![Uploading Image](https://i.imgur.com/sDmLpDe.png)  
*Uploading a brain MRI for tumor segmentation.*

### 3. Prediction Results  
![Prediction Result](https://i.imgur.com/CaGcS50.png)  
*Displaying the original MRI, predicted mask, overlay, and generated clinical report.*

### 4. Generated Medical Report  
![Medical Report](https://i.imgur.com/ecFu9bp.png)  
*Example of an automatically generated local LLM report.*

---

## ✅ Key Advantages

* **Speed:** From hours to seconds
* **Privacy:** 100% local processing (no cloud transfer)
* **Clarity:** Automatically generated, human-readable reports
* **Usability:** Lightweight and easy-to-use web interface


## 🚀 Future Improvements

* Upgrade to **U-Net++** or 3D segmentation
* Add **multi-class annotations** (glioma, meningioma, etc.)
* Export reports as PDF
* Add user authentication and admin dashboard


## 📝 License

This project is open-source and intended for research and educational purposes.
Feel free to fork or contact me for collaborations.



---

