# User Manual/Setup Guide
- Overview: Web-based fraud detection system that combines machine learning models (Random Forest, Isolation Forest, Autoencoders) with a user-friendly frontend. Allows users to upload credit card datasets, train models, evaluate them, and visualize fraud detection results interactively.

# Software Requirements
- Python 3.8 or newer
- pip (Python package installer)
- Node.js (only if modifying JavaScript assets with build tools, otherwise optional)
- Modern browser (such as Microsoft Edge, Google Chrome, Mozilla Firefox, etc. the first of which is the one used for testing)

Python Libraries: Install all the dependencies using pip install -r requirements.txt on the terminal of your chosen Code Editor.

# Folder Structure
📁 fraud_detection_project/
├── app.py                    # Main Flask application
├── preprocessing.py          # Data preprocessing logic
├── randomforest.py           # Random Forest and combined model logic
├── isolationforest.py        # Isolation Forest training script
├── autoencoder_backend.py    # Autoencoder model training and utilities
│
├── templates/
│   └── index.html            # Frontend HTML UI
│
├── static/
│   ├── scripts.js            # JavaScript for training & prediction
│   └── styles.css            # Optional CSS styling
│
├── models/                   # Folder to store trained models
│
├── data/
│   ├── processed/            # Preprocessed and intermediate data
│   └── uploads/              # User-uploaded CSV files
│
└── current_dataset.txt       # Tracks the latest uploaded dataset


# Setup for Running
1. Clone the Repository by typing the following into your chosen Code Editor’s Terminal:

  *git clone https://github.com/yourusername/creditcard-fraud-detection.git*

  *cd creditcard-fraud-detection*

2. Start the Flask app by typing the following into your chosen Code Editor’s Terminal: 

  *python app.py*

3. Afterwards, the app will run locally at http://127.0.0.1:5005. 

4. Open a browser of your choice and go to http://127.0.0.1:5005.

5. Freely navigate the dashboard, upload and preprocess any chosen singular dataset, and check for fraud detections as you wish!

6. If any resulting training model were to be corrupted or unidentified by the system, do not fret, and just retrain the respective model alone.

# Further Reports
https://docs.google.com/document/d/1HmWWhmZWT2dRR9gsvY1PLjA76K11B8ZPTiensDoEDig/edit?usp=sharing
