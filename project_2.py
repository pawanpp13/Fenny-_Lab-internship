# House price prediction.

## Import all required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

## Load dataset

df = pd.read_csv('House_Dataset.csv')
df.head()

df.shape

df.isnull().mean()*100

df['area_type'].value_counts()

df.drop(columns=["availability","area_type","society","balcony"],axis=1,inplace=True)

df.head()

df.isnull().sum()

df.dropna(inplace=True)
df.isnull().sum()

df.shape

df.head()

df['size'].unique()

df['bhk'] = df['size'].apply(lambda x: int(x.split(' ')[0]))

df.head()

df.drop(columns=["size"],axis=1,inplace=True)

df.shape

df[df.bhk>22]

df.total_sqft.unique()

def is_float(x):
    try:
        float(x)
    except:
        return False
    return True

df[~df['total_sqft'].apply(is_float)].head(10)

def convert_sqft_into_number(x):
    token = x.split('-')
    if len(token) == 2:
        return (float(token[0]) + float(token[1])) / 2
    try:
        return float(x)
    except:
        return None

df1 = df.copy()

df1['total_sqft'] = df1['total_sqft'].apply(convert_sqft_into_number)

df1.loc[30]

df2 = df1.copy()

df2['price_per_sqft'] = df2['price']*100000 / df2['total_sqft']
df2.head()

df2['location'].value_counts()

df2['location'] = df2['location'].apply(lambda x: x.strip())
df2.location.value_counts()

loc_stats[loc_stats<=10]

loc_less_than_10 = loc_stats[loc_stats<=10]
loc_less_than_10

df2.location = df2.location.apply(lambda x: 'other' if x in loc_less_than_10 else x)
df2.head()

len(df2.location.unique())

df2[ (df2.total_sqft / df2.bhk < 300) ].head()

df3 = df2[ ~(df2.total_sqft / df2.bhk < 300) ]
df3.shape

df3.price_per_sqft.describe()

def remove_outlier_from_price_per_sqft(df):
    df_out = pd.DataFrame()
    for key,sub in df.groupby('location'):
        m = np.mean( sub.price_per_sqft )
        st = np.std( sub.price_per_sqft )
        reduce_df = sub[( sub.price_per_sqft>(m-st) ) & ( sub.price_per_sqft<=(m+st) ) ]
        df_out = pd.concat( [df_out, reduce_df],ignore_index=True )
    return df_out

df4 = remove_outlier_from_price_per_sqft(df3)
df4.shape

df4.describe()

def plot_scatter_chart(df,location):
    bhk2 = df[(df.location==location) & (df.bhk==2)]
    bhk3 = df[(df.location==location) & (df.bhk==3)]
    plt.rcParams['figure.figsize'] = (12,9)
    plt.scatter(bhk2.total_sqft,bhk2.price,color='blue',label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price,marker='+', color='green',label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price (Lakh Indian Rupees)")
    plt.title(location)
    plt.legend()
    
plot_scatter_chart(df4,"Rajaji Nagar")

###### We should also remove properties where for same location, the price of (for example) 3 bedroom apartment is less than 2 bedroom apartment (with same square ft area). What we will do is for a given location, we will build a dictionary of stats per bhk, i.e.

{
    '1' : {
    
        'mean': 4000,  
        'std: 2000, 
        'count': 34
    },
    
    '2' : {
        'mean': 4300,
        'std: 2300,
        'count': 22
    },    
}
###### Now we can remove those 2 BHK apartments whose price_per_sqft is less than mean price_per_sqft of 1 BHK apartment

def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
df5 = remove_bhk_outliers(df4)
df5.shape

plot_scatter_chart(df5,"Hebbal")

df5.bath.unique()

df5[df5.bath>10]

plt.hist(df5.bath,rwidth=0.8)
plt.xlabel("Number of bathrooms")
plt.ylabel("Count")

df5[(df5.bath > df5.bhk+2)]

df6 = df5[~(df5.bath > df5.bhk+2)]
df6.head()

df6.shape

df7 = df6.drop(['price_per_sqft'],axis='columns')
df7.head()

dummies = pd.get_dummies(df7.location)
dummies.head()

df8 = pd.concat([df7,dummies.drop('other',axis='columns')],axis='columns')
df8.head()

df8.drop('location',axis='columns',inplace=True)
df8.head()

df8.shape

x = df8.drop('price',axis=1)
y = df8['price']

x.shape

y.shape

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=101)

X_train.shape, X_test.shape, y_train.shape, y_test.shape

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)
lr.score(X_test,y_test)

pred = lr.predict(X_test)
pred

y_test

def predict_price(location,sqft,bath,bhk):    
    loc_index = np.where(x.columns==location)[0][0]

    X = np.zeros(len(x.columns))
    X[0] = sqft
    X[1] = bath
    X[2] = bhk
    if loc_index >= 0:
        X[loc_index] = 1

    return lr.predict([X])[0]

predict_price('1st Phase JP Nagar',1000, 2, 2)

predict_price('1st Phase JP Nagar',1000, 2, 3)

predict_price('Indira Nagar',1400, 2, 3)

import joblib
joblib.dump(lr, "banglore house price prediction model.pkl")



