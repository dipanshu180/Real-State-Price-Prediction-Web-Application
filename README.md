````markdown
# 🏠 Bengaluru House Price Prediction

This is a machine learning web application that predicts house prices in Bengaluru based on input features like location, square footage, number of bedrooms (BHK), and number of bathrooms. It uses a trained regression model and provides a REST API built with Flask.

---

## 🚀 Features

- Predict house prices in Bengaluru using ML
- REST API using Flask
- Web frontend/backend separation (ideal for deployment)
- Location auto-suggestion
- Model trained using Linear Regression, Lasso, Decision Tree (best selected via GridSearchCV)
- Preprocessing and outlier removal logic included

---

## 🧠 Tech Stack

- Python
- Pandas, NumPy, scikit-learn
- Flask (Backend API)
- HTML/CSS/JS (Frontend - optional)
- Jupyter Notebook (for training)
- Pickle (model serialization)

---

## 📁 Project Structure

```
.
├── app.py
├── util.py
├── requirements.txt
├── artifacts/
│   ├── bangaluru_price_model_pickel
│   └── columns.json
├── excel/
│   └── bengaluru_house_prices.csv
├── model/
│   └── training_notebook.ipynb (optional)
```
---

## 🧪 Installation

### 1. Clone the repo

```bash
git clone https://github.com/your-username/house-price-predictor.git
cd house-price-predictor
````

### 2. Create a virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🛠️ How to Run

### 1. Train the model (if not already trained)

```bash
# Run the training script or Jupyter notebook to generate the model file and columns.json
```

### 2. Start the Flask server

```bash
python app.py
```

You’ll see:

```
Starting Python Flask Server For Home Price Prediction...
```

Then visit: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 📡 API Endpoints

### ➤ `GET /get_location_names`

Returns a list of available locations.

**Example Response:**

```json
{
  "locations": ["1st Phase JP Nagar", "Ejipura", "Whitefield", ...]
}
```

### ➤ `POST /predict_home_price`

**Form Data:**

* `total_sqft`: float
* `location`: string
* `bhk`: int
* `bath`: int

**Example Request (POST form-data):**

```json
{
  "total_sqft": 1200,
  "location": "Whitefield",
  "bhk": 3,
  "bath": 2
}
```

**Example Response:**

```json
{
  "estimated_price": 75.5
}
```

---




