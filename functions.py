import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import plotly.offline as py 



def distribute_data_equally(df,target_column):
    df = df.copy()
    employee_attrition = df[df[target_column] == "Yes"]
    employee_non_attrition = df[df[target_column] != "Yes"] \
                        .sample(n = len(employee_attrition))
    df =  pd.concat([employee_attrition, employee_non_attrition])
    df = df.sample(frac=1).reset_index(drop=True)
    df.reset_index(drop=True, inplace=True)
    return df


def show_confusion_matrix(test_labels,predict_labels):
    y_predict = predict_labels
    cm = confusion_matrix(test_labels, y_predict)
    sns.heatmap(cm, annot=True)
    print(classification_report(test_labels, y_predict))
    print(f"Total accuracy: {accuracy_score(test_labels, y_predict)}")

def split_dataset(df, target_column, test_ratio, validation_ratio=0):
    df = df.copy()
    Y = df[target_column]
    df.drop(columns=[target_column],inplace=True, axis=1)
    X = df
    # Split the data into training and testing sets
    train_data, test_data, train_labels, test_labels = train_test_split(
        X, Y, test_size=test_ratio, random_state=42)
    
    if validation_ratio != 0:
        train_data, val_data, train_labels, val_labels = train_test_split(
                                    train_data, train_labels, test_size=validation_ratio, random_state=42)
        return train_data, train_labels, val_data, val_labels, test_data, test_labels
    return train_data, train_labels, test_data, test_labels