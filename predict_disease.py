# Loads the model from disk using load_model().

# Preprocessed image is passed through the model using model.predict().

# Softmax output gives probability for each class (disease).

# The class with the highest probability is selected.




import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from image_enhancer import enhance_image
from suggestions import get_suggestion

# Load your trained model
model = load_model("models/plant_disease_model.h5")

# Define class labels (update based on label_map.json)
class_names = ["Bacterial Spot", "Early Blight", "Healthy", "Late Blight"]

def predict_disease(image_path):
    # Step 1: Enhance the image
    enhanced_image_path = enhance_image(image_path)

    # Step 2: Preprocess image
    image = cv2.imread(enhanced_image_path)
    image = cv2.resize(image, (128, 128))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # Step 3: Predict
    prediction = model.predict(image)
    predicted_class = class_names[np.argmax(prediction)]

    # Step 4: Get suggestion
    suggestion = get_suggestion(predicted_class)

    return predicted_class, suggestion
