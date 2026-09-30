import gradio as gr
import joblib
import pandas as pd

model = joblib.load("best_model.pkl")

def predict_weight(height):
    if height is None:
        return "Please enter your height."

    sample_height = pd.DataFrame({"Height": [height]})
    prediction = model.predict(sample_height)

    return f"Predicted Weight: {prediction[0]:.2f} pounds"

with gr.Blocks(title="Height to Weight Predictor") as interface:

    gr.Markdown("""
    # Height to Weight Prediction
    ### Machine Learning Regression Model

    Enter your height to estimate your weight.
    """)

    with gr.Row():
        with gr.Column():
            height_input = gr.Number(
                label="Enter Height (inches)",
                value=68
            )

            predict_button = gr.Button(
                "Predict Weight",
                variant="primary"
            )

        with gr.Column():
            output = gr.Textbox(
                label="Prediction Result",
                interactive=False,
                placeholder="Your prediction will appear here."
            )

    predict_button.click(
        fn=predict_weight,
        inputs=height_input,
        outputs=output
    )

    gr.Markdown("""
    ---
    **Note:** Enter height in inches. The predicted weight is in pounds.
    """)

interface.launch(theme=gr.themes.Soft())