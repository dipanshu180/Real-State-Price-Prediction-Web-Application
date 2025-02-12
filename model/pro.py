import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load the dataset
df = pd.read_csv("excel/bengaluru_house_prices.csv")
# print(df)
# print(df.shape)
# print(df.groupby('area_type')['area_type'].agg('count'))

# Drop unnecessary columns
df1 = df.drop(['area_type', 'availability', 'society', 'balcony'], axis='columns')
# print(df1.head())

# Check for missing values
print(df1.isna().sum())

# Drop rows with missing values
df3 = df1.dropna()
# print(df3.head())

# Extract number of bedrooms (bhk) from the size column
df3['size'].unique()
df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))
# print(df3.head())

# Function to check if value is float or not
def is_float(x):
    try:
        float(x)
    except:
        return False
    return True

# Check rows where total_sqft is not numeric
print(df3[~df3['total_sqft'].apply(is_float)])

# Function to handle ranges in total_sqft column
def convert_to_sqrf(x):
    token = x.split('-')
    if len(token) == 2:
        return (float(token[0]) + float(token[1])) / 2
    try:
        return float(x)
    except:
        return None

# Apply the conversion function
df4 = df3.copy()
df4['total_sqft'] = df4['total_sqft'].apply(convert_to_sqrf)
# print(df4.head())

# Calculate price per square foot
df5 = df4.copy()
df5["price_per_sqft"] = df5['price'] * 100000 / df5['total_sqft']
# print(df5.head())       

# print(len(df5.location.unique()))  # finding the total unique value in location

df5.location = df5.location.apply(lambda x : x.strip())
location_stats = df5.groupby('location')['location'].agg('count').sort_values(ascending=False)
# print(location_stats)  

# print(len(location_stats[location_stats<=10]))

location_stats_less_10 = location_stats[location_stats<=10] 
# print(location_stats_less_10)

df5.location = df5.location.apply(lambda x : 'other' if x in location_stats_less_10 else x)
# print(len(df5.location.unique()))
# print(df5.head())

df6 = df5[~(df5.total_sqft/df5.bhk<300)]
print(df6.shape)

df6.price_per_sqft.describe()

def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
df7 = remove_pps_outliers(df6)
# print(df7.shape)


def plot_scatter_chart(df,location):
    bhk2 = df[(df.location==location) & (df.bhk==2)]
    bhk3 = df[(df.location==location) & (df.bhk==3)]
    plt.figure(figsize=(15,10))
    # matplotlib.rcParams['figure.figsize'] = (15,10)
    plt.scatter(bhk2.total_sqft,bhk2.price,color='blue',label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price,marker='+', color='green',label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price (Lakh Indian Rupees)")
    plt.title(location)
    plt.legend()
    plt.show()
    
# plot_scatter_chart(df7,"Rajaji Nagar")


def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location , location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk , bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] ={
                'mean':np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
df8 = remove_bhk_outliers(df7)

print(df8.shape)

# plot_scatter_chart(df8,"Rajaji Nagar")

plt.hist(df8.price_per_sqft,rwidth=0.8)
plt.xlabel("Price Per Square Feet")
plt.ylabel("Count")
# plt.show()

plt.hist(df8.bath,rwidth=0.8)
plt.xlabel("Number of bathrooms")
plt.ylabel("Count")
# plt.show()

df8[df8.bath>df8.bhk+2] # just for checking the 

df9 = df8[df8.bath<df8.bhk+2]
# print(df9.shape)
# print(df9.head(2))

df10 = df9.drop(['size','price_per_sqft'],axis='columns')
# print(df10.head())

# print(pd.get_dummies(df10.location).astype(int)).

dummies = pd.get_dummies(df10.location).astype(int)

df11 = pd.concat([df10,dummies.drop('other',axis='columns')],axis='columns')
# print(df11.head())

df12 = df11.drop('location',axis='columns')
# print(df12.head())


print(df12.shape)

X = df12.drop('price',axis='columns')
y= df12.price

X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=10)

lr_clf = LinearRegression()
lr_clf.fit(X_train,y_train)
print(lr_clf.score(X_test,y_test)) 

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

print(cross_val_score(LinearRegression(), X, y, cv=cv))


def find_best_model_using_gridsearchcv(X, y):
    algos = {
        'linear_regression': {
            'model': Pipeline([
                ('scaler', StandardScaler()),  # Add scaling as part of the pipeline
                ('regression', LinearRegression())
            ]),
            'params': {
                'regression__fit_intercept': [True, False],  # Note the 'regression__' prefix
                'regression__n_jobs': [None, -1],
                'regression__positive': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1, 2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion': ['squared_error', 'friedman_mse', 'absolute_error'],  # Corrected 'mse' to 'squared_error'
                'splitter': ['best', 'random']
            }
        }
    }

    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    
    for algo_name, config in algos.items():
        gs = GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X, y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])

# Assuming X and y are already defined
print(find_best_model_using_gridsearchcv(X, y))

def predict_price(location,sqft,bath,bhk):    
    loc_index = np.where(X.columns==location)[0][0]

# Test the model for few properties
    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return lr_clf.predict([x])[0]

  

import json
columns = {
    'data_columns' : [col.lower() for col in X.columns]
}
with open("columns.json","w") as f:
    f.write(json.dumps(columns))