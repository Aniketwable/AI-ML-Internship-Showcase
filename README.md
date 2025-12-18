# AI-ML-Internship-Showcase
Let’s organize all four of my projects into a polished GitHub README so your portfolio looks professional and recruiter-ready.
This repository contains projects demonstrating my skills in Artificial Intelligence and Machine Learning.  
It is designed as a portfolio for internship applications at CHANGE Networks Pvt. Ltd.

## Projects
1. **Student Performance Prediction** – Regression model for score prediction.  
2. **FAQ Chatbot** – NLP-based chatbot for college queries.  
3. **Face Mask Detection** – Computer vision project using CNNs.  
4. **Disease Prediction** – Classification model for healthcare applications.

## Tech Stack
- Python
- Scikit-learn
- TensorFlow / Keras
- OpenCV
- HuggingFace Transformers

# Student Performance Prediction
# Author: Aniket [Your Last Name]
# Navi Mumbai, India

# 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 2. Create sample dataset
data = {
    'StudyHours': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    'Attendance': [60, 65, 70, 72, 75, 80, 85, 88, 90, 95],
    'PastScores': [50, 55, 58, 60, 62, 65, 68, 70, 72, 75],
    'ExamScore': [55, 60, 65, 68, 70, 75, 80, 85, 88, 92]
}
df = pd.DataFrame(data)

# 3. Explore dataset
print(df.head())
sns.pairplot(df)
plt.show()

# 4. Split data
X = df[['StudyHours', 'Attendance', 'PastScores']]
y = df['ExamScore']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train model
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Predictions
y_pred = model.predict(X_test)

# 7. Evaluate model
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# 8. Visualize predictions
plt.scatter(y_test, y_pred, color='blue')
plt.xlabel("Actual Exam Scores")
plt.ylabel("Predicted Exam Scores")
plt.title("Actual vs Predicted Exam Scores")
plt.show()

# FAQ Chatbot using NLP
# Author: Aniket [Your Last Name]
# Navi Mumbai, India

# 1. Import libraries
import random
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Download NLTK resources
nltk.download('punkt')

# 2. Training data (FAQs)
faq_data = {
    "library": "The library is open from 9 AM to 8 PM.",
    "internship": "You can apply for internships via the placement cell or online portals.",
    "canteen": "The canteen serves food from 10 AM to 6 PM.",
    "sports": "Sports facilities are available from 7 AM to 7 PM."
}

# 3. Prepare dataset
questions = list(faq_data.keys())
answers = list(faq_data.values())

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(questions)
y = questions

# 4. Train model
model = MultinomialNB()
model.fit(X, y)

# 5. Chatbot function
def chatbot_response(user_input):
    user_tokens = word_tokenize(user_input.lower())
    user_text = " ".join(user_tokens)
    user_vector = vectorizer.transform([user_text])
    prediction = model.predict(user_vector)[0]
    return faq_data[prediction]

# 6. Demo
print("Chatbot ready! Type 'exit' to quit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Chatbot: Goodbye!")
        break
    response = chatbot_response(user_input)
    print("Chatbot:", response)

    # Face Mask Detection using CNN
# Author: Aniket [Your Last Name]
# Navi Mumbai, India

# 1. Import libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# 2. Data preparation
# Assume dataset has two folders: 'mask' and 'no_mask'
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    'dataset/',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    'dataset/',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# 3. Build CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 4. Train model
history = model.fit(train_generator, validation_data=val_generator, epochs=10)

# 5. Evaluate model
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.title("Training vs Validation Accuracy")
plt.show()

# 6. Save model
model.save("face_mask_detector.h5")

# Disease Prediction using ML
# Author: Aniket [Your Last Name]
# Navi Mumbai, India

# 1. Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 2. Create sample dataset
data = {
    'Fever': [1,0,1,0,1,0,1,0],
    'Cough': [1,1,0,0,1,0,1,0],
    'Fatigue': [1,0,1,1,0,0,1,0],
    'Headache': [0,1,0,1,1,0,0,1],
    'Disease': ['Flu','Cold','Flu','Migraine','Flu','Healthy','Flu','Migraine']
}
df = pd.DataFrame(data)

# 3. Split data
X = df[['Fever','Cough','Fatigue','Headache']]
y = df['Disease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# 4. Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Predictions
y_pred = model.predict(X_test)

# 6. Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# 7. Demo prediction
sample = [[1,1,1,0]]  # Fever, Cough, Fatigue, no Headache
print("Predicted Disease:", model.predict(sample)[0])
