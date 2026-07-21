# 📚 BookBot — AI-Powered Book Recommendation Chatbot

## Overview

BookBot is a multimodal AI-powered book recommendation system built with TensorFlow and Streamlit. It recommends books based on either a text description or a book cover image.

This project was developed collaboratively as part of a Deep Learning project.

---

## Features

- Text-based book recommendations
- Image-based book recommendations
- Multiple CNN and RNN models
- Book cover image classification
- Natural language query classification
- Streamlit web application
- Top-rated book recommendations

---

## Technologies Used

- Python
- TensorFlow
- Keras
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Open Library API
- Pillow

---

## Project Structure

```text
bookbot/
│
├── app4.py
├── Wab_scrapering.ipynb
├── project_deep_fixed.ipynb
├── books_dataset.csv
├── images/
├── outputs/
└── README.md
```

---

## How It Works

The application supports two input methods:

### Text Recommendation

1. User enters a book description.
2. The RNN model predicts the book category.
3. The system recommends the highest-rated books from that category.

### Image Recommendation

1. User uploads a book cover.
2. The CNN model classifies the image.
3. The system recommends similar books from the predicted category.

---

## Models

### CNN Models

- ResNet50
- MobileNetV2
- EfficientNetB0

### RNN Models

- Bidirectional LSTM
- Bidirectional GRU
- Bidirectional LSTM with Attention

---

## Requirements

- Python 3.9+
- TensorFlow
- Streamlit

---

## Running the Project

Clone the repository

```bash
git clone https://github.com/<repository>.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app4.py
```

---

## Educational Concepts

This project demonstrates:

- Deep Learning
- Computer Vision
- Natural Language Processing (NLP)
- Transfer Learning
- Convolutional Neural Networks (CNN)
- Recurrent Neural Networks (RNN)
- Recommendation Systems
- Streamlit Application Development

---

## Author

- Abdelrahman Khaled

Computer Science Student
AI & Robotics Enthusiast

This project was developed collaboratively as a deep learning project. My contributions included data preprocessing, model development, testing, and documentation.

---

## License

This project is intended for educational purposes.
