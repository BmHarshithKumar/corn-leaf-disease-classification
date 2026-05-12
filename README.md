
# Corn Leaf Disease Classification 🌽

This project is a Machine Learning-based image classification system used to classify corn plant leaves as **Healthy** or **Unhealthy**.

## Project Overview
The system uses image processing and Machine Learning techniques to analyze corn leaf images and classify them based on health condition.

The project:
- Loads corn leaf images from a dataset
- Extracts image color features
- Uses a **K-Nearest Neighbors (KNN)** classifier
- Classifies leaves into:
  - Healthy
  - Unhealthy
- Automatically sorts images into separate folders

## Technologies Used
- Python
- OpenCV
- NumPy
- Pandas
- Scikit-learn

## Project Structure

PROJECT_02/
│── DATASET/
│── HEALTHY/
│── UNHEALTHY/
│── Project_02.py
│── README.md

## How to Run
1. Install required libraries:

pip install opencv-contrib-python pandas scikit-learn numpy

2. Run the project:

python Project_02.py

3. Enter your name when prompted.

## Output
The model classifies leaf images and stores them in:
- `HEALTHY/`
- `UNHEALTHY/`

## Author
**Harshith Kumar**  
B.Tech Student | Learning Python, Machine Learning & Android Development 🚀
