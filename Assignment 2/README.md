Here you will find the data and scripts for Assignment 2

The main script has both Regression and Classification models in order.
Pre-processing of the dataset was necessary to increase the model's fitness.

You can adjust the hyperparameters for each model if needed by modifying the required variables right after the comment "Can adjust ____ values" (e.g. alpha, l1_ratio, penalty, etc.)

Regression Models:
The script will output the RMSE and R^2 scores for the ElasticNet model, as well as a plot showing the predicted vs actual cholesterol levels. It will also include a heatmap of R^2 and RMSE across a range of 0.0 to 1.0 in increasing intervals of 0.10.

Note: The output might look disorganized as warnings pop out when running the script.

Classification Models:
The script will output the following evaluation metrics for each model:
-Accuracy
-F1 score
-ROC AUC
-Average Precision

Hyperparameter tuning will take place for both models and the script will indicate the best scores and parameters for each model. AUROC and AUPRC curves will be ploted based on the best configuration.
Note: The hyperparameters were manually changed to the ones obtained after hyperparameter tuning.