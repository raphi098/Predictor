import gradio as gr
import os
import json
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Function to handle the video processing and prediction
def predict(video_file):
    print('Predicting from video...')

    # Path to YOLO model
    model_path = os.path.join("prediction_model", "yolo_n_v8.pt")
    model = YOLO(model_path)

    # Process the video file with YOLO
    results = model.predict(source=video_file.name)
    result_json_list = []

    for index, result in enumerate(results):
        try:
            result_dict = json.loads(result.to_json(normalize=True))[0]
        except IndexError:
            result_dict = {"class": "no detection"}
        result_json_list.append(result_dict)

    # Dictionary to hold class counts
    class_counts = {
        "ulnua_krank": 0, "ulnoa_krank": 0, "medua_krank": 0, "medoa_krank": 0,
        "ulnua": 0, "ulnoa": 0, "medua": 0, "medoa": 0
    }
    print(result_json_list)

    # Loop through results and count detected classes
    for result in result_json_list:
        print(result)
        if result["class"] != "no detection":
            if result["name"] == "ulnua":
                class_counts["ulnua"] += 1
            elif result["name"] == "ulnoa":
                class_counts["ulnoa"] += 1
            elif result["name"] == "medua":
                class_counts["medua"] += 1
            elif result["name"] == "medoa":
                class_counts["medoa"] += 1
            elif result["name"] == "ulnua_krank":
                class_counts["ulnua_krank"] += 1
            elif result["name"] == "ulnoa_krank":
                class_counts["ulnoa_krank"] += 1
            elif result["name"] == "medua_krank":
                class_counts["medua_krank"] += 1
            elif result["name"] == "medoa_krank":
                class_counts["medoa_krank"] += 1
        
    # Return the class counts for generating the pie chart
    print(class_counts)
    return class_counts

def generate_pie_chart(class_counts):
    labels = list(class_counts.keys())
    sizes = list(class_counts.values())

    # Custom color palette
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0', '#ffb3e6', '#c4e17f', '#c5c3c6']
    
    # Create explode to highlight one segment
    explode = (0.1, 0, 0, 0, 0, 0, 0, 0)  # Explode only the first slice

    fig, ax = plt.subplots()

    # Function to show percentage only for non-zero slices and handle small slice labels
    def autopct_format(pct):
        return ('%1.1f%%' % pct) if pct > 0 else ''

    wedges, texts, autotexts = ax.pie(
        sizes,
        explode=explode,
        colors=colors,
        autopct=autopct_format,  # Use custom format for autopct
        shadow=True,
        startangle=90,
        wedgeprops={'edgecolor': 'black'}
    )

    # Adjust the text labels to avoid overlaps
    for autotext in autotexts:
        text_value = autotext.get_text().replace('%', '')  # Extract the percentage value
        try:
            if text_value and float(text_value) < 1.0:  # For small slices (<1%), move the text outwards
                autotext.set_position((1.4, autotext.get_position()[1]))  # Move further from the pie
        except ValueError:
            # In case of empty or invalid text, do nothing
            pass
        autotext.set_fontsize(12)
        autotext.set_color('white')
        autotext.set_weight('bold')

    # Add a title and equal aspect ratio
    ax.set_title('Class Distribution in Predictions', fontsize=14, weight='bold')
    ax.axis('equal')

    # Add a legend with class names and percentages outside of the pie chart
    ax.legend(wedges, labels, title="Classes", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

    plt.tight_layout()

    # Return the pie chart figure for display in Gradio
    return fig

# Gradio Interface
def build_interface():
    with gr.Blocks() as demo:
        # Upload Section
        with gr.Row():
            video_input = gr.File(label="Upload Video", type="filepath")  # Use 'filepath' here

        # Prediction Section
        predict_button = gr.Button("Start Prediction")
        pie_chart_output = gr.Plot(label="Class Distribution")

        # Predict button action
        predict_button.click(
            fn=lambda video_file: generate_pie_chart(predict(video_file)),  # Send video to predict and generate pie chart
            inputs=video_input,
            outputs=pie_chart_output
        )

    return demo

# Run Gradio App
if __name__ == "__main__":
    app = build_interface()
    app.launch(debug=True)
