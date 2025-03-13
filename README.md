# Fruit Ripeness Classification Game

A machine learning-based game that helps users learn about fruit ripeness classification through an interactive interface. The game uses a Random Forest Classifier to predict whether fruits are ripe or not ripe based on user-provided training data.

## Features

- Interactive image classification interface
- Real-time training and testing phases
- Visual feedback on model predictions
- Detailed statistics and visualizations
- Keyboard shortcuts for quick classification
- User-friendly interface
- Progress tracking and status updates

## Requirements

- Python 3.6 or higher
- Required Python packages:
  ```
  tkinter
  PIL (Pillow)
  numpy
  scikit-learn
  matplotlib
  ```

## Installation

1. Clone this repository or download the source code
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install required packages:
   ```bash
   pip install pillow numpy scikit-learn matplotlib
   ```

## Usage

1. Run the game:
   ```bash
   python "import tkinter as tk.py"
   ```

2. Training Phase:
   - Classify images as "Ripe" or "Not Ripe" using the buttons or keyboard shortcuts
   - Classify at least 15 images before training the model
   - Use keyboard shortcuts:
     - 'r' key for Ripe
     - 'u' key for Unripe

3. Testing Phase:
   - After training, click "Test Model" to start the testing phase
   - Classify images and compare your choices with the model's predictions
   - View detailed statistics and visualizations

4. Results:
   - View prediction distribution
   - Compare your classifications with the model's predictions
   - See accuracy metrics and visualizations
   - Reset the game to start over

## How It Works

The game uses machine learning to classify fruit ripeness based on:
- Color features (RGB values and ratios)
- Texture features (edge patterns)
- User-provided training data

The model:
1. Extracts features from images
2. Learns patterns from user classifications
3. Makes predictions on new images
4. Compares predictions with user classifications

## Technical Details

- Uses Random Forest Classifier with 100 estimators
- Processes images at 50x50 resolution for feature extraction
- Splits dataset into 80% training and 20% testing
- Requires minimum 15 training samples
- Supports continuous training mode

## Contributing

Feel free to submit issues and enhancement requests!

## Acknowledgments

- Built with Python and Tkinter
- Uses scikit-learn for machine learning
- Matplotlib for visualizations
- PIL for image processing
