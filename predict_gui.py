import tensorflow as tf
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageOps
import numpy as np

model = tf.keras.models.load_model("fashion_model.h5")

class_names = [    
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

def predict_image(path):
    img = Image.open(path).convert("RGB")

    img = img.convert("L")
    img = ImageOps.invert(img)
    img = img.resize((28, 28))
    img = np.array(img).astype("float32") / 255.0
    img = img.reshape(1, 28, 28)

    logits = model.predict(img)
    pred = np.argmax(logits)
    probs = tf.nn.softmax(logits[0])
    confidence = probs[pred].numpy()
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