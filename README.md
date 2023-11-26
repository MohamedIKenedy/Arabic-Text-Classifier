# Arabic Text Classifier using TF-IDF

## Overview

This project implements an Arabic text classifier using the TF-IDF (Term Frequency-Inverse Document Frequency) technique. The classifier, trained on a corpus from Hespress, a major Arabic news outlet, categorizes Arabic text into predefined categories.

## Usage

1. **Training**:

   - Train the TF-IDF classifier using the Hespress news corpus.
   - Preprocess data (e.g., tokenization, stop words removal).
   - Save the trained model for later use.

2. **Web Application**:

   - Start the Flask server with `app.py`.
   - Access the web app via the provided URL.
   - Input Arabic text for classification.

## File Structure

- `Dataset`: CSV files for training the SGD classifier.
- `templates/`: Directory for HTML templates.
- `static/`: Contains the style.css file and the images folder
- `run.py`: contains the flask backend code for the web app

## Training Process

1. **Data Collection**: Obtain the Hespress Arabic news corpus.
2. **Data Preprocessing**: Clean text data (stop words removal, tokenization).
3. **Feature Extraction**: Apply TF-IDF vectorization.
4. **Model Training**: Fit the TF-IDF vectors to a classifier model.
5. **Evaluation**: Assess model performance.

## Support

For inquiries or issues, contact [Your Contact Information].

## License

