import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from tqdm import tqdm


def openFile(fn1):
    # Using the pd.read_sas function to read in the .XPT files. The files are
    # converted to a pandas dataframe format and 'SEQN' is used as the index
    df = pd.read_sas(fn1, format=None, index="SEQN")

    # Returning the dataframe created from each .XPT file that we open
    return df


# A list of all the .XPT files we need to read into dataframes
FN = [
    "P_DEMO.XPT",
    "P_DIQ.XPT",
    "P_BMX.XPT",
    "P_BPXO.XPT",
    "P_HDL.XPT",
    "P_TRIGLY.XPT",
    "P_TCHOL.XPT",
    "P_INS.XPT",
    "P_GLU.XPT",
    "P_HSCRP.XPT",
    "P_FERTIN.XPT",
    "P_KIQ_U.XPT",
    "P_GHB.XPT",
]

# Declaring an empty list to hold the dataframes created from each .xpt file
D = []

# For loop to iterate through all of the .XPT files
for fn in FN:
    # This just gives the correct path for the files
    fn1 = os.path.join("../../Data/", fn)

    # Calling the openFile function to return the file's contents as a dataframe
    X = openFile(fn1)

    # Appending the list D with the new dataframe
    D.append(X)

## Merge all the data files
Z = pd.concat(D, axis=1)

# A list of all the features we want to keep in the dataframes
selectedData = Z[
    [
        "RIDAGEYR",
        "RIDEXPRG",
        "RIAGENDR",
        "BPXOSY1",
        "BPXODI1",
        "BPXOSY2",
        "BPXODI2",
        "BPXOSY3",
        "BPXODI3",
        "BMXWT",
        "BMXBMI",
        "BMXWAIST",
        "LBDLDMSI",
        "WTSAFPRP",
        "LBDHDDSI",
        "LBDTCSI",
        "LBDGLUSI",
        "LBXHSCRP",
        "LBDHRPLC",
        "LBDFERSI",
        "DIQ010",
        "DIQ050",
        "DIQ160",
        "KIQ026",
        "WTINTPRP",
        "WTMECPRP",
        "LBDINLC",
        "LBXGH",
    ]
]

# This just prints to the terminal each column and how many missing values there are
# print(selectedData.isna().sum())

# This makes a .csv file with information about each of the features
selectedData.describe().to_csv("dataInfo.csv")

# Turning our data into a .csv file. This will be the raw data before any processing
selectedData.to_csv("rawData.csv", index=True)

# This fills in all the NaN values with 0
# inplace = False tells the function to make a copy instead of modifying the dataframe
# directly. This is the default value. If it was set to true, we would get a
# warning
processedData = selectedData.fillna(0, inplace=False)


# Using Boolean indexing to choose certain columns that we want to exclude from
# the dataset. Here we are removing the responses from individuals who are reported
# as pregnant RIDEXPRG == 1 or could not be determined RIDEXPRG == 3
mk1 = (processedData["RIDEXPRG"] == 1) | (processedData["RIDEXPRG"] == 3)
processedData = processedData.drop(processedData[mk1].index, axis=0)

# Here we are doing the same thing as above except we are dropping all individuals
# who are less than the age of 12
mk2 = processedData["RIDAGEYR"] < 12
mk2Sum = int(mk2.sum())
print("The sum of people less than 12 years old is " + str(mk2Sum))
processedData = processedData.drop(processedData[mk2].index, axis=0)

# Averaging the three BP columns and making two new columns with the averages
syoMean = processedData[["BPXOSY1", "BPXOSY2", "BPXOSY3"]].mean(axis=1)
processedData = processedData.assign(BPXOSYAVG=syoMean)
diaMean = processedData[["BPXODI1", "BPXODI2", "BPXODI3"]].mean(axis=1)
processedData = processedData.assign(BPXOSDIVG=diaMean)

# Dropping the three bp measurement columns and the pregnancy column because
# we will not include those in our model
processedData = processedData.drop(
    ["RIDEXPRG", "BPXOSY1", "BPXOSY2", "BPXOSY3", "BPXODI1", "BPXODI2", "BPXODI3"],
    axis=1,
)

# Here we are removing the responses from individuals who are reported
# as borderline, refusing to answer, or don't know if they have diabetes
# We are also doing this for individual's responses to if they have prediabetes
values_to_remove = [3, 7, 9]
processedData = processedData[~processedData["DIQ010"].isin(values_to_remove)]
values_to_remove = [3, 7, 9]
processedData = processedData[~processedData["DIQ160"].isin(values_to_remove)]

extraInfo = processedData[
    ["WTINTPRP", "WTMECPRP", "WTSAFPRP", "LBDINLC", "LBDHRPLC", "DIQ050"]
]
processedData = processedData.drop(
    ["WTINTPRP", "WTMECPRP", "WTSAFPRP", "LBDINLC", "LBDHRPLC", "DIQ050"], axis=1
)


# print info abot the dataset
print(processedData.info())


# Turning our data into a .csv file. This will be the processed data
processedData.to_csv("processedData.csv", index=True)
extraInfo.to_csv("extraInfo.csv", index=True)

# This makes a .csv file with information about each of the features
processedData.describe().to_csv("dataInfo.csv")

# Seperating the data based on feature columns and target columns. We actually
# have two seperate target columns here. One is diabetes and another is prediabetes
X = processedData.drop(["DIQ010", "DIQ160"], axis=1)
Y1 = processedData["DIQ010"]
Y2 = processedData["DIQ160"]


# Split the dataset into random training and testing sets. This uses
# random_state = 1 which ensures that the output will be reproducible across
# multiple calls
X_train, X_test, y_train, y_test = train_test_split(
    X, Y1, test_size=0.3, random_state=1
)

# Create the SVM classifier
# clf = SVC(kernel='linear', C=1, gamma='auto')


# Fit the classifier to the training data
# for i in tqdm(range(10)):
# clf.fit(X_train, y_train)


# Predict the target values for the testing data
# y_pred = clf.predict(X_test)


# Evaluate the model performance
# print("Accuracy:", accuracy_score(y_test, y_pred))
