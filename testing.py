import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('animal_classification_model2.h5')
class_labels = ['dogs', 'cats']
# Function to preprocess an image for prediction
def preprocess_image(img_path):
    img = Image.open(img_path).resize((128, 128))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize the pixel values
    return img_array

# Function to make predictions and update the label
def predict_image():
    file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        preprocessed_image = preprocess_image(file_path)
        predictions = model.predict(preprocessed_image)
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = class_labels[predicted_class_index]
        result_label.config(text=f'The model predicts that the image belongs to the class: {predicted_class}')

# Create the main window
root = tk.Tk()
root.title("Animal Classification")

# Create and set up the widgets
browse_button = tk.Button(root, text="Browse Image", command=predict_image)
browse_button.pack(pady=20)

result_label = tk.Label(root, text="")
result_label.pack()

# Start the Tkinter event loop
root.mainloop()
