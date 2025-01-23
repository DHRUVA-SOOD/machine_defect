# **Manufacturing Defect Prediction API**

This project provides an API to predict manufacturing defects based on production data. It allows users to upload data, train a machine learning model, and make predictions on new data.

---

## **Features**

- Upload production data via API.
- Train a machine learning model.
- Predict defect status with probabilities.
- View interactive API documentation using Swagger UI.

---

## **Setup Instructions**

### **1. Clone the Repository**

bash
git clone https://github.com/dhruva-sood/machine_defect.git
cd your-repo-name

### 2. Install Dependencies
Ensure you have Python (>=3.8) installed. Then, install the required packages using:

bash
Copy
Edit
pip install -r requirements.txt
### 3. Run the API Locally
Start the FastAPI application by running the following command:

bash
Copy
Edit
uvicorn main:app --reload
