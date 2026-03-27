╔══════════════════════════════════════════════════════════════╗
║        UNIVERSAL DATA ANALYZER — Streamlit App              ║
║        Project for Exam / Viva Submission                   ║
╚══════════════════════════════════════════════════════════════╝

📁 FILES IN THIS FOLDER:
   ├── universal_data_analyzer_app.py   ← Main Streamlit app
   ├── requirements.txt                 ← Python dependencies
   └── README.txt                       ← This file

─────────────────────────────────────────────────────────────
🚀 HOW TO RUN (Step-by-Step)
─────────────────────────────────────────────────────────────

STEP 1 — Install Python (if not installed)
   Download from: https://www.python.org/downloads/

STEP 2 — Install required libraries
   Open terminal / command prompt and run:

   pip install streamlit pandas numpy matplotlib seaborn scikit-learn

   OR use the requirements file:
   pip install -r requirements.txt

STEP 3 — Run the app
   streamlit run universal_data_analyzer_app.py

STEP 4 — App opens in browser automatically at:
   http://localhost:8501

─────────────────────────────────────────────────────────────
📋 WHAT THE APP DOES (9 Steps)
─────────────────────────────────────────────────────────────

Step 1 → Upload any CSV dataset (or use auto-generated sample)
Step 2 → Data Understanding (shape, types, nulls, stats)
Step 3 → Data Cleaning (remove duplicates, fill missing values)
Step 4 → Data Filtering (filter by column + condition)
Step 5 → EDA — statistics, correlation, value counts
Step 6 → Visualization — Heatmap, Histogram, Scatter, Line,
                          Bar Chart, Box Plot
Step 7 → ML Prediction — Linear Regression + Logistic Regression
Step 8 → Result Summary + Insights + Downloads + Conclusion

─────────────────────────────────────────────────────────────
🎤 VIVA ONE-LINER
─────────────────────────────────────────────────────────────

"My project is a Universal Data Analyzer built using Python
and Streamlit. Any CSV dataset can be uploaded and the system
automatically performs data cleaning, filtering, EDA,
visualization, and machine learning prediction."

─────────────────────────────────────────────────────────────
📦 LIBRARIES USED
─────────────────────────────────────────────────────────────

Library         Purpose
─────────────── ───────────────────────────────────────────
streamlit       Interactive web UI
pandas          Data loading & manipulation
numpy           Numerical operations
matplotlib      Charts and plots
seaborn         Statistical visualizations
scikit-learn    Linear & Logistic Regression ML models