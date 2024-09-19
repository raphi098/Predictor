from flask import Blueprint, request, jsonify, render_template
import os
from werkzeug.utils import secure_filename
import subprocess
from ultralytics import YOLO
import json
from PIL import Image

ai_classification_blueprint = Blueprint('Ki Klassifizierung', __name__)

@ai_classification_blueprint.route('/ai_classification')
def ai_classification():
    return render_template('ai_classification.html')

def extract_frames(video_path, output_dir):
    print('Extracting frames...')
    every_n_frames = 30
    command = [
        'ffmpeg',
        '-i', video_path,
        '-vf', f"fps={every_n_frames}",  # Select every every_n_frames frame
        os.path.join(output_dir, 'frame_%04d.png')  # use the output file path here
    ]
    subprocess.run(command)

@ai_classification_blueprint.route('/upload', methods=['POST'])
def upload_file():
    print('Uploading file...')
    file = request.files['file']
    print('File:', file.filename)
    if file.filename == '':
        return 'No selected file'
    if file:
        filename = secure_filename(file.filename)
        try:
            video_path = os.path.join('Videos', filename)
            file.save(video_path)

            # Specify the directory
            dir_path = os.path.join(os.getcwd(), 'extracted_frames')

            # Delete all image files in the directory
            image_files = os.listdir(dir_path)
            for file in image_files:
                os.remove(os.path.join(dir_path, file))
            extract_frames(video_path, 'extracted_frames')
        except Exception as e:
            print("Error saving file:", e)
            return 'Error saving file'

        return f'Video: {filename}, Upload erfolgreich'

@ai_classification_blueprint.route('/predict', methods=['GET'])
def predict():
    print('Predicting...')
    try:
        # Get the list of pictures in the extracted_frames folder
        files = os.listdir("extracted_frames")
        pictures = list(filter(lambda x: x.endswith('.png'), files))
        picture_paths = [os.path.join("extracted_frames", picture) for picture in pictures]
        # Path to trained model and prediction for pictures in extracted_frames saved in results
        model_path = os.path.join("prediction_model", "yolo_n_v8.pt")
        model = YOLO(model_path)
        results = model.predict(picture_paths, max_det=1)
        
        # Create a JSON file with the results + Picture size and width + save it in the runs folder
        with Image.open(picture_paths[0]) as img:
            width, height = img.size
    except Exception as e:
        print("Error predicting:", e)
        return jsonify({"error": str(e)}), 500

    result_json_list = []
    filenames = []
    for index, result in enumerate(results):
        try:
            result_dict = json.loads(result.tojson(normalize=True))[0]
        except IndexError:
            result_dict = {"class": "no detection"}
        result_dict["picture"] = pictures[index]
        result_dict["width"] = width
        result_dict["height"] = height
        result_json_list.append(result_dict)
        filenames.append(pictures[index])
        result.save(f"static/detected_nerves/{pictures[index]}")

    count_classes = [
        {"name": "ulnua_krank", "count": 0},
        {"name": "ulnoa_krank", "count": 0},
        {"name": "medua_krank", "count": 0},
        {"name": "medoa_krank", "count": 0},
        {"name": "ulnua", "count": 0},
        {"name": "ulnoa", "count": 0},
        {"name": "medua", "count": 0},
        {"name": "medoa", "count": 0}
    ]

    for result in result_json_list:
        print(result)
        if result["class"] != "no detection":
            if result["name"] == "ulnua":
                count_classes[4]["count"] += 1
            elif result["name"] == "ulnoa":
                count_classes[5]["count"] += 1
            elif result["name"] == "medua":
                count_classes[6]["count"] += 1
            elif result["name"] == "medoa":
                count_classes[7]["count"] += 1
            elif result["name"] == "ulnua_krank":
                count_classes[0]["count"] += 1
            elif result["name"] == "ulnoa_krank":
                count_classes[1]["count"] += 1
            elif result["name"] == "medua_krank":
                count_classes[2]["count"] += 1
            elif result["name"] == "medoa_krank":
                count_classes[3]["count"] += 1

    print(count_classes)
    
    with open('runs/result_json_list.json', 'w') as f:
        json.dump(result_json_list, f)

    return jsonify(count_classes)

