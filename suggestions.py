# Based on predicted disease name (e.g., "Potato___Early_blight"), a predefined suggestion is retrieved from a Python dictionary.

# Suggestion may include pesticide, care tips, etc.


def get_suggestion(disease_name):
    suggestions = {
        "Bacterial Spot": "Remove infected leaves and avoid overhead watering. Use copper-based fungicide.",
        "Early Blight": "Apply fungicides like chlorothalonil. Rotate crops and avoid water on leaves.",
        "Late Blight": "Destroy infected plants. Avoid working in garden when foliage is wet.",
        "Healthy": "Your plant is healthy! Keep it that way with regular checks and balanced nutrition."
    }
    return suggestions.get(disease_name, "No suggestion available.")
