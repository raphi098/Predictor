import os
import sys
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from Blueprints.ai_predictor import ai_classification_blueprint

app = Flask(__name__)
app.register_blueprint(ai_classification_blueprint)
CORS(app)  # Erlaubt Anfragen von allen Domänen


if __name__ == '__main__':
    app.run(debug=True)

@app.route('/')
def home():
    return render_template('ai_classification.html')