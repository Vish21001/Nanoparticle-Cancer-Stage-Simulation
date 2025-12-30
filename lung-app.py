import tkinter as tk
from tkinter import filedialog, Label
from tensorflow.keras.models import load_model
import cv2
import numpy as np
from grad_cam import make_gradcam

model = load_model("lung_cancer_multiclass_model.h5")
IMG_SIZE = 128

def predict_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img_input = img_resized.reshape(1,IMG_SIZE,IMG_SIZE,1)/255.0
        pred = model.predict(img_input)[0]
        class_index = np.argmax(pred)
        class_names = ["Healthy","Benign","Malignant"]
        result = f"Prediction: {class_names[class_index]}"
        label.config(text=result)
        make_gradcam(file_path, class_index)

root = tk.Tk()
root.title("Lung Cancer Multiclass Detection")
label = Label(root, text="Upload a CT Scan", font=("Arial", 16))
label.pack(pady=20)
btn = tk.Button(root, text="Select Image", command=predict_image)
btn.pack(pady=10)
root.mainloop()
