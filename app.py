import gradio as gr
import joblib

# Load the trained model
model = joblib.load("linear_regression_model.joblib")

# Define a prediction function
def predict_house_price(size):
    prediction = model.predict([[size]])
    return f"Predicted house price: {prediction[0]}"

# Create Gradio interface
interface = gr.Interface(
    fn=predict_house_price, 
    inputs=gr.Number(label="House Size (e.g., 3)"),  # Use gr.Number instead of gr.inputs.Number
    outputs="text",
    title="House Price Prediction",
    description="Enter the size of the house to predict its price using a simple linear regression model."
)

# Launch the interface
if __name__ == "__main__":
    interface.launch()
