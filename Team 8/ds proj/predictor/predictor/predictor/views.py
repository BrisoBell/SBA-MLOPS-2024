from django.shortcuts import render
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Home view
def home(request):
    return render(request, "home.html")

# Predict view
def predict(request):
    return render(request, "predict.html")

# Result view using Voting Classifier
def result(request):
    # Load the diabetes dataset
    df = pd.read_csv("C:/Users/shine rijo/Downloads/diabetes.csv")

    # Handle outliers for 'Insulin' and 'DiabetesPedigreeFunction'
    numeric_columns = ['Insulin', 'DiabetesPedigreeFunction']
    for column_name in numeric_columns:
        Q1 = np.percentile(df[column_name], 25, interpolation='midpoint')
        Q3 = np.percentile(df[column_name], 75, interpolation='midpoint')
        IQR = Q3 - Q1
        low_lim = Q1 - 1.5 * IQR
        up_lim = Q3 + 1.5 * IQR

        df[column_name] = np.where(df[column_name] < low_lim, low_lim, df[column_name])
        df[column_name] = np.where(df[column_name] > up_lim, up_lim, df[column_name])

    # Split dataset into features (X) and labels (Y)
    X = df.drop('Outcome', axis=1)
    Y = df['Outcome']

    # Split the dataset into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

    # Normalize the features (especially needed for SVM and Logistic Regression)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize the individual classifiers
    rand_clf = RandomForestClassifier(n_jobs=-1, random_state=42)
    svm_clf = SVC(probability=True, random_state=42)  # probability=True for soft voting
    log_reg = LogisticRegression(random_state=42)

    # Create a Voting Classifier combining all three models
    voting_clf = VotingClassifier(
        estimators=[
            ('rf', rand_clf),
            ('svm', svm_clf),
            ('lr', log_reg)
        ],
        voting='hard'  # 'hard' for majority voting, 'soft' for probability voting
    )

    # Train the Voting Classifier
    voting_clf.fit(X_train, Y_train)

    # Get user input from the request (Expecting 8 feature inputs from the user)
    user_input = [float(request.GET[f'n{i+1}']) for i in range(8)]

    # Normalize the user input in the same way as the training data
    user_input = scaler.transform([user_input])

    # Predict using the Voting Classifier
    pred = voting_clf.predict(user_input)

    # Interpret the prediction result (0 = negative, 1 = positive for diabetes)
    result1 = "positive" if pred[0] == 1 else "negative"

    # Return the result to the template
    return render(request, "predict.html", {"result2": result1})