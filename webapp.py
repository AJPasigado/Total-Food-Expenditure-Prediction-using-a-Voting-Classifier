import numpy as np
from sklearn.externals import joblib
from flask import Flask, render_template, request
import pandas as pd
import re

# initialize our Flask application and the Keras model
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def predict():
    string_data, bin_data = load_data()
    params = []

    if request.method == 'POST':
        for item in request.form:
            if item == 'alc_exp':
                params.append(clean_data(request.form[item], bin_data["Alcoholic Beverages Expenditure"]))
            elif item == 'water_exp':
                params.append(clean_data(request.form[item], bin_data["Housing and water Expenditure"]))
            elif item == 'med_exp':
                params.append(clean_data(request.form[item], bin_data["Medical Care Expenditure"]))
            elif item == 'trans_exp':
                params.append(clean_data(request.form[item], bin_data["Transportation Expenditure"]))
            elif item == 'com_exp':
                params.append(clean_data(request.form[item], bin_data["Communication Expenditure"]))
            elif item == 'head_age':
                params.append(clean_data(request.form[item], bin_data["Household Head Age"])) 
            else:
                params.append(request.form[item])
            print(item)
        return render_template('home.html', 
            regions=string_data["Region"], 
            main_source=string_data["Main Source of Income"],
            household_type=string_data["Type of Household"],
            predict = predictor(params, bin_data["Total Food Expenditure"]))
    else:

        return render_template('home.html', 
            regions=string_data["Region"], 
            main_source=string_data["Main Source of Income"],
            household_type=string_data["Type of Household"]
            )

def clean_data(data, bin):
    temp = np.append(bin, [data])
    temp.sort()
    index = np.where(temp==data)[0][0]

    if index >= len(bin):
        index = len(bin)-1
    return index

def predictor(data, bins):
    model = joblib.load('voting_model.pkl')
    index = int(model.predict([data])[0])
    if (index == 0):
        predict = "0 - {0:.2f}".format(bins[0])
    else: 
        predict = "{0:.2f}- {1:.2f}".format(bins[index-1], bins[index])
    return predict

def load_data():
    dataset ='dataset.csv'
    dataset = pd.read_csv(dataset)
    dataset.columns

    dataset = dataset.dropna()

    dataset = dataset.drop(['Bread and Cereals Expenditure',
       'Total Rice Expenditure', 'Meat Expenditure',
       'Total Fish and  marine products Expenditure', 'Fruit Expenditure',
       'Vegetables Expenditure', ], axis = 1)
    
    columns = ['Region', 'Main Source of Income', 'Household Head Sex', 'Household Head Marital Status', 'Household Head Highest Grade Completed', 
           'Household Head Job or Business Indicator', 'Household Head Occupation', 'Household Head Class of Worker', 
           'Type of Household']
    
    global columns_string
    columns_string = {}

    for column in columns:
        columns_string[column] = dataset[column].unique()
        for index, occ in enumerate(dataset[column].unique()):
            dataset[column] = dataset[column].replace(occ, index)
        dataset[column] = pd.to_numeric(dataset[column])

    for column in dataset.columns:
        dataset[column] = dataset[column].astype('float64')
    
    columns = ['Total Household Income', 
       'Imputed House Rental Value',
      'Crop Farming and Gardening expenses',
       'Total Income from Entrepreneurial Acitivites', 'Restaurant and hotels Expenditure',
       'Alcoholic Beverages Expenditure', 'Tobacco Expenditure', 'Clothing, Footwear and Other Wear Expenditure',
       'Housing and water Expenditure',   'Medical Care Expenditure', 'Transportation Expenditure',
       'Communication Expenditure', 'Education Expenditure',
       'Miscellaneous Goods and Services Expenditure',
       'Special Occasions Expenditure']

    global columns_bins
    columns_bins = {}

    for column in columns: 
        counts, bin_edges = np.histogram(dataset[column], bins=300)
        dataset[column] = dataset[column].astype('float')
        dataset[column] = pd.cut(dataset[column], bin_edges, right=False, labels = False)
        dataset[column] = pd.to_numeric(dataset[column])
        columns_bins[column] = bin_edges
        
    column = 'Household Head Age'
    counts, bin_edges = np.histogram(dataset[column], bins=10)
    dataset[column] = dataset[column].astype('float')
    dataset[column] = pd.cut(dataset[column], bin_edges, right=False, labels = False)
    dataset[column] = pd.to_numeric(dataset[column])
    columns_bins[column] = bin_edges
        
    column = 'Total Food Expenditure'
    counts, bin_edges = np.histogram(dataset[column], bins=5)
    dataset[column] = dataset[column].astype('float')
    dataset[column] = pd.cut(dataset[column], bin_edges, right=False, labels = False)
    dataset[column] = pd.to_numeric(dataset[column])
    columns_bins[column] = bin_edges
        
    dataset = dataset.dropna()
    return columns_string, columns_bins


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading SKLearn model and Flask starting server..."
           "please wait until server has fully started"))
    load_models()
    app.run(host='0.0.0.0')
