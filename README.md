# 🧠 Fashion MNIST Image Classifier (.exe GUI App)

This desktop application lets users upload an image (`.jpg`, `.png`, etc.) of a clothing item and predicts what item it is, using a neural network trained on the Fashion MNIST dataset.

Built in Python with **TensorFlow**, **Pillow**, and a **Tkinter GUI**, it can be bundled into a Windows `.exe` for easy distribution.

## ✨ Features
- 🧠 Trains a neural network on Fashion MNIST  
- 🖼 User uploads an image from disk  
- 🔍 Automatically cleans/handles `.jfif`, `.jpg`, `.jpeg` (even truncated files)  
- 📊 Predicts one of 10 fashion categories (T-shirt, boot, dress, …)  
- ✅ Ships as a standalone `.exe` via PyInstaller  
- 🧪 Shows a human-readable prediction popup  

## 📁 Project Structure
```
📦 HumanBrain_tensorflow/
├── train_model.py          # Train and save the Fashion MNIST model
├── predict_gui.py          # GUI: load model, pick image, show prediction
├── fashion_model.keras     # Saved model (generated by training)
├── requirements.txt        # Exact package versions (frozen from venv)
├── README.md               # You’re here!
└── dist/
    └── predict_gui.exe     # Compiled standalone app
```

## 🛠 Setup (for developers)

1. **Create & activate** a virtual environment:
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

2. **Install exact dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

> All package versions are pinned in `requirements.txt`, so every developer gets identical libraries.

## 🚀 How to Use

### 1 – Train the model (first time only)
```bash
python train_model.py
```
Creates `fashion_model.h5`.

### 2 – Run the GUI
```bash
python predict_gui.py
```
—or run the compiled version—
```bash
./dist/predict_gui.exe
```
Steps:
1. Click **“Choose Image and Predict”**  
2. Select a `.jpg`, `.jpeg`, or `.png`  
3. Read the prediction popup!

## 📦 Build the `.exe`
```bash
pyinstaller --onefile --windowed predict_gui.py
```
Copy `fashion_model.keras` into `dist/` next to `predict_gui.exe`.

> The `.exe` can be shared or used on any Windows machine—no Python required.

## 🧪 Known Limitations
- **Real-world image mismatch**: The model is trained on 28×28 grayscale MNIST-style images (white on black), so real images with white backgrounds or rotated items (e.g. boots) may be misclassified even at high confidence.
- **Visual ambiguity in small resolution**: Certain classes (e.g. bags vs ankle boots, shirts vs pullovers) look similar when resized to 28×28 and lose key shape details.
- **Softmax always returns a class**: Even when unsure, the model must assign one label — which can result in high-confidence wrong predictions.
- **Dataset bottleneck reached**: Fashion-MNIST has limited diversity (no side views, no lighting variation, all centered), which restricts the model’s ability to generalize to real-world scenarios.
- **No rotation or context awareness**: Without additional training data, the model can’t reliably identify tilted or partially visible items.
- **GPU usage not automatic**: TensorFlow requires a correctly installed GPU build with matching CUDA/cuDNN. Most pip installs are CPU-only unless handled via Conda or Docker.
- **Model size and bundling**: Bundling the `.keras` model inside the `.exe` requires careful file handling since `.keras` is a directory, not a flat file.
---

## ✅ To-Do / Next Steps

### ✅ Completed
- [x] Switched to Conv2D + MaxPool CNN architecture  
- [x] Trained for 15 epochs  
- [x] Preprocessed images to match shape `(28, 28, 1)` and normalized to `[0, 1]`  
- [x] Prediction function now correctly interprets softmax output  
- [x] Image preview supported (optional for debugging)  
- [x] Accuracy increased significantly with CNN and correct preprocessing 
- [x] Add `Dropout` layers to reduce overfitting (improved realworld accuracy by large margin)
- [x] Use `EarlyStopping` to avoid unnecessary training epochs 
- [x] Add `ImageDataGenerator` for rotation/zoom/shift augmentation  
- [x] Add `BatchNormalization` for more stable and faster training   


## 📚 Credits
Created by **Jakub Janca** using:
- [TensorFlow](https://www.tensorflow.org/)  
- [Pillow](https://python-pillow.org/)  
- [Tkinter (stdlib)](https://docs.python.org/3/library/tkinter.html)  
- Fashion MNIST dataset (Zalando Research)  

## 📄 License
MIT License — free to use, modify, and share.