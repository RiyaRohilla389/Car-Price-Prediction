# Car Price Prediction using Linear Regression 🚗💰

A comparative machine learning study evaluating how different categorical data encoding methodologies affect the performance of a Linear Regression model when predicting the pricing of used Ford vehicles.

---

## 📌 Project Overview
This project processes a comprehensive dataset containing structural, mechanical, and usage metrics of over 17,000 vehicles. Categorical traits like vehicle model, transmission mechanics, and fuel metrics present a predictive constraint for regression algorithms. To determine the mathematically optimal approach, two standalone preprocessing methodologies were engineered and evaluated:

1. **One-Hot Encoding (Dummies):** Translating categorical entries into dedicated binary columns.
2. **Label Encoding:** Mapping text categories directly into integer markers via mapping functions.

---

## 📊 Dataset Insights
The workflow utilizes data sourced from a structural used car repository (`ford.csv`). Initial Exploratory Data Analysis (EDA) mapped the feature boundaries:
* **Total Instances:** 17,966 records.
* **Features Extracted:** `model`, `year`, `transmission`, `mileage`, `fuelType`, `tax`, `mpg`, and `engineSize`.
* **Target Vector:** `price`.
* **Data Integrity:** Checked for missing variables ($0$ structural nulls found across features).

---

## ⚙️ Preprocessing & Feature Engineering
Numerical features (`year`, `mileage`, `tax`, `mpg`, `engineSize`) were normalized using `StandardScaler` to bring variance magnitudes to a unified baseline scale. 

The categorical variations were transformed as follows:
* **One-Hot Encoding Execution:** Evaluated using `pd.get_dummies()` dropping the first dimension to eliminate multicollinearity and dummy variable traps.
* **Label Encoding Execution:** Evaluated via `sklearn.preprocessing.LabelEncoder()` to handle low-dimensionality representations.

---

## 📈 Experimental Metrics & Results

A `LinearRegression` model was trained on a $67/33$ cross-validation split for both configurations with the following baseline results:

| Preprocessing Technique | Baseline $R^2$ Score | Adjusted $R^2$ Score |
| :--- | :---: | :---: |
| **One-Hot Encoded (Dummy Vectors)** | **`0.8396`** | **`0.8387`** |
| **Label Encoded (Integer Mapped)** | `0.7310` | `0.7306` |

### 🔍 Key Findings
* **One-Hot Encoding outperformed Label Encoding by ~10.8%** on the baseline metric ($0.8396$ vs $0.7310$).
* **Why it worked:** Label Encoding maps an artificial numerical order (e.g., `Automatic = 0`, `Manual = 1`) which a Linear Regression algorithm misinterprets as a quantitative scale. One-Hot Encoding isolates categorical features cleanly into orthogonal dimensions, enabling proper mathematical coefficient scaling.

---

## 🛠️ Technology Stack
* **Language:** Python 3.11
* **Libraries Used:** 
  * Data Extraction: `Pandas`, `NumPy`
  * Visual Grounding: `Matplotlib`, `Seaborn`
  * Machine Learning Pipeline: `Scikit-Learn`
