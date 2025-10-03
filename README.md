# 🏠 House Price Predictor

**End-to-end machine learning demo that predicts California house prices using polynomial regression and Ridge regularization, wrapped in a Streamlit app for interactive training and CSV-based predictions.**

This project demonstrates an end-to-end machine learning workflow for predicting median California house values. It uses regression models with polynomial feature expansion and Ridge regularization. A [Streamlit](https://streamlit.io/) app provides an interactive interface where developers can train models, save them, and upload CSVs with house features to get predictions.

---

## Techniques Used

- **[scikit-learn Pipelines](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html):** Chains preprocessing steps and regression into a reusable sequence.
- **[StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html):** Normalizes input features to improve training stability.
- **[PolynomialFeatures](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html):** Expands features to capture non-linear relationships.
- **[Ridge Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html):** Adds L2 regularization to reduce overfitting.
- **[Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html):** Evaluates model consistency across folds.
- **[Joblib](https://joblib.readthedocs.io/):** Saves trained models for reuse.
- **[Streamlit File Uploader](https://docs.streamlit.io/library/api-reference/widgets/st.file_uploader):** Enables CSV upload and batch prediction.

---

## Libraries and Tools of Interest

- [scikit-learn](https://scikit-learn.org/) — Machine learning framework for preprocessing and regression.  
- [Streamlit](https://streamlit.io/) — Framework for interactive data apps.  
- [Pandas](https://pandas.pydata.org/) — Data wrangling and CSV handling.  
- [Matplotlib](https://matplotlib.org/) — Visualization for metrics and tuning curves.  
- [Joblib](https://joblib.readthedocs.io/) — Model serialization.  
- Dataset: [California Housing dataset](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset).  

---

## Project Structure

```text
.
├── app.py
├── artifacts/
│   ├── house_price_ridge_degX_alphaY.joblib
│   ├── summary.json
│   └── test_predictions.csv
├── requirements.txt
├── sample_houses.csv
└── images/
    └── house_price_flowchart.png

```

---

## 📝 License

MIT — use it freely, modify it, and share it.

---

> 🔧 Built with ❤️ by [Sushyam Nagallapati](www.linkedin.com/in/sushyamnagallapati)

> MEng System Design Engineering, University of Waterloo
