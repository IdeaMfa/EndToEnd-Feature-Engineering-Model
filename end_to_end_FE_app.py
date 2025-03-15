import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
from datetime import date
from sklearn.metrics import  accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from statsmodels.stats.proportion import proportions_ztest

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500),

def load_application_train():
    data = pd.read_csv("datasets/application_train.csv")
    return data


def load():
    data = pd.read_csv("datasets/titanic.csv")
    return data


def outliersThresholds(df, colname, q1=0.25, q3=0.75):
    quartile1 = df[colname].quantile(q1)
    quartile3 = df[colname].quantile(q3)
    
    iqr_val = quartile3 - quartile1
    up_limit = quartile3 + 1.5*iqr_val
    low_limit = quartile1 - 1.5*iqr_val
    
    return low_limit, up_limit


def checkOutliers(df, columname):
    threshHolds = outliersThresholds(df, colname=columname)
    
    return df[(df[columname] < threshHolds[0]) | (df[columname] > threshHolds[1])].any(axis=None)

def grab_col_names(dataframe, cat_th=10, car_th=20, print_flag=True):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisinde numerik görünümlü kategorik değişkenler de dahildir.

    
    """
    
    """
    List comprehencion:
    l1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    l2 = [item for item in l1 if item > 4]
    print(l2)
    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
    dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
    dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    if print_flag:
        print(f"Obsevrations: {dataframe.shape[0]}")
        print(f"Variables: {dataframe.shape[1]}")
        print(f"cat_cals: {len(cat_cols)}")
        print(f"num_cols: {len(num_cols)}")
        print(f"cat_but_car: {len(cat_but_car)}")
        print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car


def grab_outliers(dataframe, colname, index=False):
    low, up = outliersThresholds(dataframe, colname)
    outliers = dataframe[(dataframe[colname] < low) | (dataframe[colname] > up)]

    if outliers.shape[0] > 10:
        print(outliers.head())
    else:
        print(outliers)

    if index:
        return outliers.index
    

def removeOutliers(df, columname):
    low, up = outliersThresholds(df, colname=columname)
    
    return df[~((df[columname] < low) | (df[columname] > up))]


def reassign_with_thresholds(dataframe, columname):
    # re-assignment with thresholds
    low , up = outliersThresholds(dataframe, columname)
    
    # For upper limits
    dataframe.loc[(dataframe[columname] > up), columname] = up
    # For lower limits
    dataframe.loc[(dataframe[columname] < low), columname] = low


def missing_values_table(dataframe, na_name=False):
    na_cols = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_cols].isnull().sum().sort_values(ascending=False)
    ratios = (dataframe[na_cols].isnull().sum() / dataframe.shape[0]).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratios, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_cols
    

def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + "_NA_FLAG"] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")
        

def label_encoder(dataframe, binary_column):
    le = LabelEncoder()
    dataframe[binary_column] = le.fit_transform(dataframe[binary_column])

    return dataframe


def find_bin_cols(dataframe):
    binCols = [col for col in dataframe.columns if dataframe[col].dtype not in [int, float]
               and dataframe[col].nunique() == 2]
    
    return binCols


def one_hot_encoder(dataframe, cat_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=cat_cols, drop_first=drop_first).astype(int)

    return dataframe


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")


def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df if temp_df[col].dtype == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]
    
    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df


def num_summary(dataframe, numerical_col, plot=False):
    quantuiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantuiles).T)

    if plot:
        sns.histplot(dataframe[numerical_col], bins=20, kde=True, color="red", alpha=0.5)
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()
    

def hypothesis_test(df: pd.DataFrame, variable: str, target: str, flag1, flag2):
    test_stat, pvalue = proportions_ztest(count=[df.loc[df[variable] == flag1, target].sum(),
                                             df.loc[df[variable] == flag2, target].sum()],
                                             
                                             nobs=[df.loc[df[variable] == flag1, target].shape[0],
                                                   df.loc[df[variable] == flag2, target].shape[0]])
    
    return test_stat, pvalue


# %%
"""
Titanic Data Set End-to-End Feature Engineering & Data Preprocessing
The main goal of this problem is to model whether humans can survive or not over this data set.
"""
df = load()
df.shape
df.head()

# Make uppercase names of the columns in the dataset
df.columns = [col.upper() for col in df.columns]

###########################################################################
# 1. Feature Extraction
###########################################################################
# Cabin bool 
df["NEW_CABIN_BOOL"] = df["CABIN"].notnull().astype('int')
# Number of characters in names
df["NEW_NAME_LETTER_COUNT"] = df["NAME"].str.len()
# Name of words in names
df["NEW_NAME_WORD_COUNT"] = df["NAME"].apply(lambda x: len(str(x).split(" ")))
# Name dr
df["NEW_NAME_DR"] = df["NAME"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr.")]))
# Name title
df["NEW_NAME_TITLE"] = df.NAME.str.extract(' ([A-Za-z]+)\.', expand=False)
# Family size
df["NEW_FAMILY_SIZE"] = df["SIBSP"] + df["PARCH"] + 1
# Age - Pclas factor
df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]
# Is alone flag
df.loc[((df["SIBSP"] + df["PARCH"]) > 0), "NEW_ISALONE"] = "NO"
df.loc[((df["SIBSP"] + df["PARCH"]) == 0), "NEW_ISALONE"] = "YES"
# Age level
df.loc[(df["AGE"] < 18), "NEW_CAT_AGE"] = "young"
df.loc[((df["AGE"] >= 18) & df["AGE"] < 56), "NEW_CAT_AGE"] = "mature" 
df.loc[(df["AGE"] >= 56), "NEW_CAT_AGE"] = "senior"
# Age x Sex
df.loc[((df["AGE"] < 18) & (df["SEX"]) == "male"), "NEW_CAT_AGE_SEX"] = "youngmale"
df.loc[(((df["AGE"] >= 18) & (df["AGE"] < 56)) & (df["SEX"] == "male")), "NEW_CAT_AGE_SEX"] = "maturemale" 
df.loc[((df["AGE"] >= 56) & (df["SEX"] == "male")), "NEW_CAT_AGE_SEX"] = "seniormale"

df.loc[((df["AGE"] < 18) & (df["SEX"]) == "female"), "NEW_CAT_AGE_SEX"] = "younfemale"
df.loc[(((df["AGE"] >= 18) & (df["AGE"] < 56)) & (df["SEX"] == "female")), "NEW_CAT_AGE_SEX"] = "maturefemale" 
df.loc[((df["AGE"] >= 56) & (df["SEX"] == "female")), "NEW_CAT_AGE_SEX"] = "seniorfemale"

cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if "PASSENGERID" not in col]

#############
# 2. Outliers
#############
# Check outliers
for col in num_cols:
    print(f"{col}: {checkOutliers(df, col)}")

for col in num_cols:
    reassign_with_thresholds(df, col)

###################
# 3. Missing Values
###################
missing_values_table(df)

# Since we have created a new (CABIN_BOOL) variable instead of the CABIN variable, we can drop the Cabin variable
df.drop("CABIN", inplace=True, axis=1)

# We also delete the other variables we want to delete (we already created features from NAME)
removeCalls = ["TICKET", "NAME"]
df.drop(removeCalls, inplace=True, axis=1)

# Let's fill the NA values related to AGE according to the TITLE
df["AGE"] = df["AGE"].fillna(df.groupby("NEW_NAME_TITLE")["AGE"].transform("median"))

# Now we need to reassign the features values about the age
# Age - Pclas factor
df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]
# Age level
df.loc[(df["AGE"] < 18), "NEW_CAT_AGE"] = "young"
df.loc[((df["AGE"] >= 18) & (df["AGE"] < 56)), "NEW_CAT_AGE"] = "mature" 
df.loc[(df["AGE"] >= 56), "NEW_CAT_AGE"] = "senior"
# Age x Sex
df.loc[((df["AGE"] < 18) & (df["SEX"] == "male")), "NEW_CAT_AGE_SEX"] = "youngmale"
df.loc[(((df["AGE"] >= 18) & (df["AGE"] < 56)) & (df["SEX"] == "male")), "NEW_CAT_AGE_SEX"] = "maturemale" 
df.loc[((df["AGE"] >= 56) & (df["SEX"] == "male")), "NEW_CAT_AGE_SEX"] = "seniormale"

df.loc[((df["AGE"] < 18) & (df["SEX"] == "female")), "NEW_CAT_AGE_SEX"] = "younfemale"
df.loc[(((df["AGE"] >= 18) & (df["AGE"] < 56)) & (df["SEX"] == "female")), "NEW_CAT_AGE_SEX"] = "maturefemale" 
df.loc[((df["AGE"] >= 56) & (df["SEX"] == "female")), "NEW_CAT_AGE_SEX"] = "seniorfemale"

# I will also fill the missing values of the EMBARKED variable with the mode value
df = df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0)

###################
# 4. Label Encoding
###################
# Find binary columns
binary_cols = find_bin_cols(df)

for col in binary_cols:
    label_encoder(df, col)

##################
# 5. Rare Encoding
##################
# Make rareness analysis
rare_analyser(df, "SURVIVED", cat_cols)
# Make rare encoding programmatically, it's possibile to do a more deep analysis on this problem,
# but to make a general overview it's not that bat to make rare encoding in a programmatic way.
df = rare_encoder(df, 0.01)
# To see the results:
df["NEW_NAME_TITLE"].value_counts()

#####################
# 6. One-Hot Encoding
#####################
# Select the columns for one-hot encoding
ohe_cols = [col for col in df.columns if 2 < df[col].nunique() <= 10]
# Give them into one-hot encoder
df = one_hot_encoder(df, ohe_cols)
# We have provided new variables so do the updates
cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if "PASSENGERID" not in col]
df.shape

# A new problem is emerged now, is there two class variable that one of
# it's classes less than 0.01 ratio
rare_analyser(df, "SURVIVED", cat_cols)
# Determine useless columns
useless_cols = [col for col in df.columns if df[col].nunique() == 2 and
                (df[col].value_counts() / len(df) < 0.01).any(axis=None)]
# It's an option to drop the useless columns
#df.drop(useless_cols, axis=1, inplace=True)

#
# 7. Standard Scaling (We don't need in this problem)
# scaler = StandardScaler()
# df[num_cols] = scaler.fit_transform(df[num_cols])

# df[num_cols].head()

##########
# 8. Model
##########

# Determine the dependent and independent variables
# Dependent variable:
Y = df["SURVIVED"]
X = df.drop(["PASSENGERID", "SURVIVED"], axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=17)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=46).fit(X_train, Y_train)
Y_pred = rf_model.predict(X_test)
accuracy_score(Y_pred, Y_test)
# I've got 80% of accuracy score with this work


def plot_importance(model, feature_names, top_n=10):
    """
    Plots the top_n most important features from a trained model.

    Parameters:
    - model: A trained model with a `feature_importances_` attribute (e.g., XGBoost, LightGBM, RandomForest).
    - feature_names: List of feature names corresponding to the model's input features.
    - top_n: Number of top features to display.
    """
    if not hasattr(model, "feature_importances_"):
        raise ValueError("Model does not have feature_importances_ attribute.")

    # Get feature importances
    importances = model.feature_importances_
    
    # Create a DataFrame
    importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    
    # Sort by importance and select top_n
    importance_df = importance_df.sort_values(by="Importance", ascending=False).head(top_n)

    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=importance_df, palette="viridis")
    plt.title(f"Top {top_n} Feature Importances")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.show()


plot_importance(rf_model, X_train)

# %%