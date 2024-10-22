from django.shortcuts import render
import pandas as pd
import matplotlib.pyplot as pl
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_auc_score, roc_curve, classification_report
from sklearn.preprocessing import scale, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import KFold
def home(request):
    return render(request, "home.html")
def predict(request):
    return render(request, "predict.html")
def result(request):
    df = pd.read_csv("C:/Users/shine rijo/Downloads/diabetes.csv")
    X = df.drop("Outcome", axis=1)
    Y = df['Outcome']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)

    st_x = StandardScaler()
    X_train = st_x.fit_transform(X_train)
    X_test = st_x.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train, Y_train)

    knn = KNeighborsClassifier()
    knn.fit(X_train, Y_train)

    rand_clf = RandomForestClassifier(criterion='entropy', max_depth=15, max_features=0.75, min_samples_leaf=2,
                                      min_samples_split=3, n_estimators=130)
    rand_clf.fit(X_train, Y_train)

    val1 = float(request.GET['n1'])
    val2 = float(request.GET['n2'])
    val3 = float(request.GET['n3'])
    val4 = float(request.GET['n4'])
    val5 = float(request.GET['n5'])
    val6 = float(request.GET['n6'])
    val7 = float(request.GET['n7'])
    val8 = float(request.GET['n8'])

    pred=model.predict([val1, val2, val3, val4,val5,val6,val7,val8])

    result1= ""
    if pred==[1]:
        result1 ='positive'
    else:
        result1 ='negative'

    return render(request, "predict.html",{"result2":result1})
