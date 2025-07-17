import tensorflow as tf
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageOps
import numpy as np
from PIL import ImageEnhance

model = tf.keras.models.load_model("fashion_model.keras")

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

def predict_image(path):
    img = Image.open(path).convert("RGB")

    img = img.convert("L")
    img = ImageOps.invert(img)
    img = ImageEnhance.Contrast(img).enhance(2.0)
    img = ImageOps.fit(img, (28, 28), method=Image.Resampling.LANCZOS)
    img = np.array(img, dtype="float32") / 255
    img = img.reshape(1, 28, 28, 1)

    probs = model.predict(img, verbose=0)[0]
    pred = np.argmax(probs)
    confidence = float(probs[pred])
    return f"{class_names[pred]} ({confidence:.2%} confidence)"

def upload_and_predict():
    filepath = filedialog.askopenfilename(filetypes=[("Image Files", ("*.png", ".*jpeg", "*.jpg"))])
    if filepath:
        try:
            result = predict_image(filepath)
            messagebox.showinfo("Prediction", f"This looks like a: {result}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

root = tk.Tk()
root.title("Fashion MINST Classifier")

btn = tk.Button(root, text="Choose Image and Predict", command=upload_and_predict)
btn.pack(padx=20, pady=20)

root.mainloop()