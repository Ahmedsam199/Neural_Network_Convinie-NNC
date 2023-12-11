from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = load_model('animal_classification_model.h5')

# Function to preprocess an image for prediction
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the pixel values
    return img_array

# Example: Make predictions on a new image
new_image_path = 'DogTest.jpg'
preprocessed_image = preprocess_image(new_image_path)

# Make predictions
predictions = model.predict(preprocessed_image)

# Get the predicted class
predicted_class_index = np.argmax(predictions[0])
class_labels = ['dogs', 'cats']  # Make sure this list matches your original class labels
predicted_class = class_labels[predicted_class_index]

# Display the prediction
print(f'The model predicts that the image belongs to the class: {predicted_class}')
