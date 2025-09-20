Forest Cover Type Prediction
```
This repository contains a machine learning project for predicting forest cover types using cartographic variables. The project focuses on a comparative analysis of two powerful ensemble learning algorithms: Random Forest Classifier (RFC) and XGBoost Classifier. The objective was to build a model that is not only highly accurate but also computationally efficient for a large-scale classification task.
```
Comparative Analysis & Key Findings
```
The core of this project was to determine the best model based on a trade-off between predictive accuracy and training time.
```
Initial Results
```
Random Forest Classifier: Achieved a high accuracy of 0.96 but had a significant training time of 12 minutes.

XGBoost Model: Delivered a very similar accuracy of 0.95, but trained in just 18 seconds.

This initial comparison revealed that XGBoost was a far more efficient choice, with a negligible drop in accuracy.
```
Hyperparameter Tuning
```
To optimize performance, I performed hyperparameter tuning on both models.

Random Forest Classifier: Tuning was found to be computationally infeasible on a standard machine, estimated to take around 19.5 hours.

XGBoost Model: After tuning, the model's accuracy improved to 0.97 with the optimal parameters: colsample_bytree: 1.0, learning_rate: 0.2, max_depth: 9, n_estimators: 648, subsample: 0.6.
```
Conclusion:
```
The final decision was to move forward with the XGBoost model. Its superior computational speed and ability to achieve the highest accuracy after tuning made it the optimal solution for this large dataset. This project serves as a clear demonstration that for real-world applications, efficiency is a critical factor in model selection, and sacrificing a small amount of initial accuracy can lead to significant gains in development time and resource management.

Repository Contents
notebooks/: Jupyter notebooks showing the full analysis pipeline from data exploration to model evaluation.

src/: Python scripts for data preprocessing and model inference.

models/: The final, trained and optimized XGBoost model.

data/: The original dataset.
```
