# Heart Disease Detection using Machine Learning

A machine learning project that predicts heart disease based on various medical parameters using Logistic Regression.

## 🏥 Project Overview

This project implements a machine learning model to detect heart disease using patient data including age, sex, chest pain type, blood pressure, cholesterol levels, and other medical indicators. The model achieves **83.19% accuracy** in predicting heart disease presence.

## 📊 Dataset

The project uses a heart disease dataset with the following features:
- **Age**: Patient's age in years
- **Sex**: Patient's gender (0 = Female, 1 = Male)
- **Chest Pain Type**: Type of chest pain experienced
- **Resting Blood Pressure**: Systolic blood pressure in mm Hg
- **Cholesterol**: Serum cholesterol in mg/dl
- **Fasting Blood Sugar**: Blood sugar level (0 = ≤120 mg/dl, 1 = >120 mg/dl)
- **Resting ECG**: Resting electrocardiographic results
- **Max Heart Rate**: Maximum heart rate achieved
- **Exercise Angina**: Exercise induced angina (0 = No, 1 = Yes)
- **Oldpeak**: ST depression induced by exercise relative to rest
- **ST Slope**: Slope of the peak exercise ST segment

**Target Variable**: Heart disease presence (0 = No disease, 1 = Disease present)

## 🚀 Features

- **Machine Learning Model**: Logistic Regression classifier
- **Data Preprocessing**: Proper feature engineering and data splitting
- **Model Evaluation**: Comprehensive metrics including accuracy, precision, recall, and F1-score
- **Prediction Interface**: Easy-to-use prediction function for new patient data
- **Probability Scores**: Confidence levels for predictions

## 📈 Model Performance

- **Accuracy**: 83.19%
- **Precision**: 0.84 (Heart Disease), 0.82 (No Heart Disease)
- **Recall**: 0.85 (Heart Disease), 0.80 (No Heart Disease)
- **F1-Score**: 0.85 (Heart Disease), 0.81 (No Heart Disease)

## 🛠️ Technologies Used

- **Python 3.x**
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning algorithms and evaluation metrics
- **Jupyter Notebook**: Interactive development and analysis

## 📁 Project Structure

```
Heart Disease/
├── Heart_disease__detection.ipynb    # Main Jupyter notebook
├── dataset.csv                       # Heart disease dataset
├── Detect Heart Disease.pdf          # Project documentation
└── README.md                         # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- Jupyter Notebook or JupyterLab
- Required Python packages (install via pip):

```bash
pip install pandas numpy scikit-learn jupyter
```

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/heart-disease-detection.git
cd heart-disease-detection
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the notebook**:
```bash
jupyter notebook Heart_disease__detection.ipynb
```

## 💻 Usage

### Training the Model

1. Open the Jupyter notebook
2. Run all cells to train the model
3. The model will be trained on 80% of the data and tested on 20%

### Making Predictions

```python
# Example: Predict for a new patient
input_data = (62, 0, 0, 140, 268, 0, 0, 160, 0, 3.6, 0)

# Create DataFrame with proper feature names
new_data = pd.DataFrame([input_data], columns=X.columns)

# Make prediction
prediction = model.predict(new_data)
prediction_proba = model.predict_proba(new_data)

print(f"Prediction: {prediction[0]}")
print(f"Probability of heart disease: {prediction_proba[0][1]:.3f}")
```

## 📊 Results Interpretation

- **Prediction = 0**: No heart disease detected
- **Prediction = 1**: Heart disease detected
- **Probability scores**: Confidence level in the prediction (0.0 to 1.0)

## 🔧 Customization

You can easily modify the model by:
- Changing the train-test split ratio
- Using different machine learning algorithms
- Adding feature scaling or normalization
- Implementing cross-validation
- Adding more evaluation metrics

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Your Name** - [Your GitHub Profile](https://github.com/rathodakashnayak)

## 🙏 Acknowledgments

- Dataset providers
- Scikit-learn community
- Open source contributors

## 📞 Contact

If you have any questions or suggestions, please open an issue on GitHub or contact me at [rathodakashnayak45@gmail.com]

---

⭐ **Star this repository if you find it helpful!**
