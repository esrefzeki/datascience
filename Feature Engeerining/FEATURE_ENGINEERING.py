#############################################
# FEATURE ENGINEERING & DATA PRE-PROCESSING
#############################################

# 1. OUTLIERS
#    - Aykırı Gözlemleri Yakalama: boxplot, outlier_thresholds, check_outlier, grab_outliers
#    - Aykırı Gözlem Problemini Çözme: Silme, re-assignment with thresholds, Local Outlier Factor

# 2. MISSING VALUES
#    - Eksik Değerleri Yakalama
#    - Eksik Değer Problemini Çözme: Silme, Lambda ve Apply ile doldurma, kategorik değişken kırılımında doldurma
#    - Gelişmiş Analizler: Yapı ve Rassallık İncelemesi, missing_vs_target

# 3. LABEL ENCODING

# 4. ONE-HOT ENCODING

# 5. RARE ENCODING

# 6. STANDARDIZATION: StandardScaler, RobustScaler, MinMaxScaler, Log, Numeric to Categorical

# 7. FEATURE EXTRACTION: NA_FLAG, BOOL, BINARY, LETTER COUNT, WORD COUNT, SPECIAL_CHAR

# 8. INTERACTIONS: Toplam, çarpım, kombinasyon, ağırlık

# 9. END TO END APPLICATION

# 10. PROJE


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
import os
from sklearn.metrics import accuracy_score
from sklearn.neighbors import LocalOutlierFactor

pd.pandas.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 170)


def load_application_train():
    data = pd.read_csv("datasets/application_train.csv")
    return data

df = load_application_train()
df.head()


def load():
    data = pd.read_csv("datasets/titanic.csv")
    return data

df = load()
df.head()

#############################################
# OUTLIERS
#############################################

#############################################
# AYKIRI GOZLEMLERI YAKALAMAK
#############################################

####################################################
# Grafik teknikle aykırılar nasıl gözlemlenir?
####################################################

sns.boxplot(x=df["Age"])
plt.show()

df["Age"].describe().T

####################################################
# Aykırı değerler nasıl yakalanır?
####################################################

q1 = df["Age"].quantile(0.25)
q3 = df["Age"].quantile(0.75)
iqr = q3 - q1
up = q3 + 1.5 * iqr
low = q1 - 1.5 * iqr

# alt sınırdan küçük ya da üst sınırdan büyük olanlar
df[(df["Age"] < low) | (df["Age"] > up)]
df[(df["Age"] < low) | (df["Age"] > up)]["Age"]
df[(df["Age"] < low) | (df["Age"] > up)].index

####################################################
# Aykırı değer var mı yok mu?
####################################################

df[(df["Age"] > up) | (df["Age"] < low)].any(axis=None)

# 1. eşik deger belirledik
# 2. aykırılara eriştik
# 3. aykırı var mı diye sorduk

def outlier_thresholds(dataframe, col_name):
    quartile1 = dataframe[col_name].quantile(0.25)
    quartile3 = dataframe[col_name].quantile(0.75)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

outlier_thresholds(df, "Fare")

low, up = outlier_thresholds(df, "Age")

df[(df["Age"] < low) | (df["Age"] > up)].head()
df[(df["Age"] < low) | (df["Age"] > up)]["Age"]
df[(df["Age"] < low) | (df["Age"] > up)].index


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


check_outlier(df, "Age")

from helpers.eda import grab_col_names
cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(df)

for col in num_cols:
    print(col, check_outlier(df, col))


dff = load_application_train()
cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(dff)

for col in num_cols:
    print(col, check_outlier(dff, col))

####################################################
# Aykırı Gözlemleri Yakalamak
####################################################

def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)
    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

grab_outliers(df, "Age")
grab_outliers(df, "Age", True)


# RECAP

outlier_thresholds(df, "Age")
check_outlier(df, "Age")
grab_outliers(df, "Age", True)


#############################################
# AYKIRI GOZLEM PROBLEMINI COZME
#############################################

#############################################
# SİLME
#############################################

low, up = outlier_thresholds(df, "Fare")
df.shape

df[~((df["Fare"] < low) | (df["Fare"] > up))].shape


def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers

df.shape
remove_outlier(df, "Fare").shape

for col in ["Age", "Fare"]:
    new_df = remove_outlier(df, col)

df.shape[0] - new_df.shape[0]


#############################################
# BASKILAMA YONTEMI (re-assignment with thresholds)
#############################################

low, up = outlier_thresholds(df, "Fare")

df.loc[(df["Fare"] > up), "Fare"] = up

# df.loc[(df["Fare"] < low), "Fare"] = low

def replace_with_thresholds(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if low_limit > 0:
        dataframe.loc[(dataframe[col_name] < low_limit), col_name] = low_limit
        dataframe.loc[(dataframe[col_name] > up_limit), col_name] = up_limit
    else:
        dataframe.loc[(dataframe[col_name] > up_limit), col_name] = up_limit


df = load()
df.shape

for col in ["Age", "Fare"]:
    print(col, check_outlier(df, col))

for col in ["Age", "Fare"]:
    replace_with_thresholds(df, col)

for col in ["Age", "Fare"]:
    print(col, check_outlier(df, col))


#########################
# RECAP
########################

df = load()

# AYKIRI DEGER SAPTAMA
outlier_thresholds(df, "Age")
check_outlier(df, "Age")
grab_outliers(df, "Age", index=True)

# AYKIRI DEGER TEDAVI
remove_outlier(df, "Age").shape
replace_with_thresholds(df, "Age")
check_outlier(df, "Age")


#############################
# Çok Değişkenli Aykırı Değer Analizi: Local Outlier Factor
#############################

# Gözlemleri bulundukları konumda yoğunluk tabanlı skorlayarak
# buna göre aykırı değer olabilecek değerleri tanımlayabilmemize imkan sağlıyor.

df = sns.load_dataset('diamonds')
df = df.select_dtypes(include=['float64', 'int64'])
df = df.dropna()
df.head()

for col in df.columns:
    print(col, check_outlier(df, col))

clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df)
df_scores = clf.negative_outlier_factor_
df_scores[0:5]

np.sort(df_scores)[0:5]

scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 20], style='.-')
plt.show()

esik_deger = np.sort(df_scores)[3]

df[df_scores < esik_deger]

df[df_scores < esik_deger].shape
df.describe().T

df[df_scores < esik_deger].index


df[df_scores < esik_deger].drop(axis=0, labels=41918)

df[df_scores < esik_deger].drop(axis=0, labels=df[df_scores < esik_deger].index)

# ÖDEV (Keyfi)
# LOF ile yakalanan index'lerdeki aykırı değerleri değişkenleri tek tek düşünerek baskılayınız.

#############################################
# MISSING VALUES
#############################################

#############################################
# EKSIK DEGERLERIN YAKALANMASI
#############################################

df = load()
df.head()

# eksik gozlem var mı yok mu sorgusu
df.isnull().values.any()

# degiskenlerdeki eksik deger sayisi
df.isnull().sum()

# degiskenlerdeki tam deger sayisi
df.notnull().sum()

# en az bir tane eksik degere sahip olan gözlem birimleri
df[df.isnull().any(axis=1)]

# tam olan gözlem birimleri
df[df.notnull().all(axis=1)]

df.isnull().sum().sort_values(ascending=False)

# Oransal olarak görmek icin
(df.isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)

na_cols = [col for col in df.columns if df[col].isnull().sum() > 0]

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)

    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)

    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])

    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df)
missing_values_table(df, True)


df = load_application_train()
missing_values_table(df)


#############################################
# EKSIK DEGER PROBLEMINI COZME
#############################################

#############################################
# COZUM 1: Hızlıca silmek
#############################################

df.dropna()

#############################################
# COZUM 2: Lambda ve apply ile doldurmak
#############################################
df = load()
df["Age"].fillna(0)
df["Age"].fillna(df["Age"].mean())
df["Age"].fillna(df["Age"].median())

df.apply(lambda x: x.fillna(x.mean()), axis=0)

df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0).head()

dff = df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)

dff.isnull().sum().sort_values(ascending=False)

df["Embarked"].fillna(df["Embarked"].mode()[0])
df["Embarked"].fillna(df["Embarked"].mode()[0]).isnull().sum()
df["Embarked"].fillna("missing")


df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0)

df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0).isnull().sum()


#############################################
# Scikit-learn ile eksik deger atama
#############################################

# pip install scikit-learn

V1 = np.array([1, 3, 6, np.NaN, 7, 1, np.NaN, 9, 15])
V2 = np.array([7, np.NaN, 5, 8, 12, np.NaN, np.NaN, 2, 3])
V3 = np.array([np.NaN, 12, 5, 6, 14, 7, np.NaN, 2, 31])

df = pd.DataFrame(
    {"V1": V1,
     "V2": V2,
     "V3": V3}
)

df

from sklearn.impute import SimpleImputer

imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')

imp_mean.fit(df)
imp_mean.transform(df)

#############################################
# Kategorik Değişken Kırılımında Değer Atama
#############################################

V1 = np.array([1, 3, 6, np.NaN, 7, 1, np.NaN, 9, 15])
V2 = np.array([7, np.NaN, 5, 8, 12, np.NaN, np.NaN, 2, 3])
V3 = np.array([np.NaN, 12, 5, 6, 14, 7, np.NaN, 2, 31])
V4 = np.array(["IT", "IT", "IK", "IK", "IK", "IK", "IT", "IT", "IT"])

df = pd.DataFrame(
    {"salary": V1,
     "V2": V2,
     "V3": V3,
     "departman": V4}
)

df


df.groupby("departman")["salary"].mean()

df["salary"].fillna(df.groupby("departman")["salary"].transform("mean"))

df.loc[(df["salary"].isnull()) & (df["departman"] == "IK"), "salary"] = df.groupby("departman")["salary"].mean()["IK"]

df = load()
df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()


#############################################
# RECAP
#############################################

df = load()
# missing table
missing_values_table(df)
# sayısal değişkenleri direk median ile oldurma
df.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0)
# kategorik değişkenleri mode ile doldurma
df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0)
# kategorik değişken kırılımında sayısal değişkenleri doldurmak
df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()


#############################################
# GELISMIS ANALIZLER
#############################################

#############################################
# EKSIK VERI YAPISININ INCELENMESI
#############################################

msno.bar(df)
plt.show()

msno.matrix(df)
plt.show()

msno.heatmap(df)
plt.show()

#############################################
# EKSIK DEGERLERIN BAGIMLI DEGISKEN ILE ILISKISININ INCELENMESI
#############################################


missing_values_table(df, True)
na_cols = missing_values_table(df, True)

def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")


missing_vs_target(df, "Survived", na_cols)


#############################################
# RECAP
#############################################

df = load()
na_cols = missing_values_table(df, True)
df.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0)
df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0)
df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean"))
missing_vs_target(df, "Survived", na_cols)


#############################################
# 3. LABEL ENCODING
#############################################

df = load()
df.head()
df["Sex"].head()
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit_transform(df["Sex"])[0:5]

le.inverse_transform([0, 1])


def label_encoder(dataframe, binary_col):
    labelencoder = preprocessing.LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


df = load()

binary_cols = [col for col in df.columns if df[col].dtypes == "O"
               and len(df[col].unique()) == 2]


for col in binary_cols:
    label_encoder(df, col)

df.head()


#############################################
# 4. ONE-HOT ENCODING
#############################################

# İkiden fazla sınıfa sahip olan kategorik değişkenlerin 1-0 olarak encode edilmesi.

df = load()
df["Embarked"].value_counts()

pd.get_dummies(df, columns=["Embarked"]).head()
pd.get_dummies(df, columns=["Embarked"], dummy_na=True).head()
pd.get_dummies(df, columns=["Embarked"], drop_first=True).head()


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = load()

ohe_cols = [col for col in df.columns if 10 >= len(df[col].unique()) > 2]

one_hot_encoder(df, ohe_cols).head()

one_hot_encoder(df, ohe_cols, drop_first=True).head()


#############################################
# 5. RARE ENCODING
#############################################

# 1. Kategorik değişkenlerin azlık çokluk durumunun analiz edilmesi.
# 2. Rare kategoriler ile bağımlı değişken arasındaki ilişkinin analiz edilmesi.
# 3. Rare encoder yazacağız.


#############################################
# 1. Kategorik değişkenlerin azlık çokluk durumunun analiz edilmesi.
#############################################

df = load_application_train()
df.head()


df["NAME_EDUCATION_TYPE"].value_counts()

cat_cols = [col for col in df.columns if df[col].dtypes == 'O']

from helpers.eda import cat_summary

for col in cat_cols:
    cat_summary(df, col)


#############################################
# 2. Rare kategoriler ile bağımlı değişken arasındaki ilişkinin analiz edilmesi.
#############################################

df["NAME_INCOME_TYPE"].value_counts()

df.groupby("NAME_INCOME_TYPE")["TARGET"].mean()

# 1. Sınıf Frekansı
# 2. Sınıf Oranı
# 3. Sınıfların target açısından değerlendirilmesi

[col for col in df.columns if df[col].dtypes == 'O' and (df[col].value_counts() / len(df) < 0.10).any(axis=None)]



def rare_analyser(dataframe, target, rare_perc):

    rare_columns = [col for col in dataframe.columns if dataframe[col].dtypes == 'O'
                    and (dataframe[col].value_counts() / len(dataframe) < rare_perc).any(axis=None)]

    for col in rare_columns:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")


rare_analyser(df, "TARGET", 0.01)


#############################################
# 3. Rare encoder'ın yazılması.
#############################################


temp_df = df.copy()
temp_df["ORGANIZATION_TYPE"].unique()
len(temp_df["ORGANIZATION_TYPE"].unique())
temp_df["ORGANIZATION_TYPE"].value_counts() / len(temp_df)
tmp = temp_df["ORGANIZATION_TYPE"].value_counts() / len(temp_df)
rare_labels = tmp[tmp < 0.01].index
len(rare_labels)

temp_df["ORGANIZATION_TYPE"] = np.where(temp_df["ORGANIZATION_TYPE"].isin(rare_labels), 'Rare',
                                        temp_df["ORGANIZATION_TYPE"])

len(temp_df["ORGANIZATION_TYPE"].unique())
temp_df["ORGANIZATION_TYPE"].value_counts() / len(temp_df)


def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df


new_df = rare_encoder(df, 0.01)

rare_analyser(new_df, "TARGET", 0.01)
rare_analyser(df, "TARGET", 0.01)


#############################################
# 6. FEATURE SCALING (STANDARTLASTIRMA & DEĞİŞKEN DÖNÜŞÜMLERİ)
#############################################

# StandardScaler: Klasik standartlaştırma. Ortalamayı çıkar, standart sapmaya böl. z = (x - u) / s
df = load()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(df[["Age"]])
df["Age"] = scaler.transform(df[["Age"]])
df["Age"].describe().T

scaler = StandardScaler().fit(df[["Fare"]])
df["Fare"] = scaler.transform(df[["Fare"]])
df["Fare"].describe().T


# RobustScaler: Medyanı çıkar iqr'a böl.
df = load()
from sklearn.preprocessing import RobustScaler

transformer = RobustScaler().fit(df[["Age"]])
df["Age"] = transformer.transform(df[["Age"]])
df["Age"].describe().T

# MinMaxScaler: Verilen 2 değer arasında değişken dönüşümü
# X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

df = load()
from sklearn.preprocessing import MinMaxScaler

transformer = MinMaxScaler().fit(df[["Age"]])
df["Age"] = transformer.transform(df[["Age"]])

df["Age"].describe().T


# Log: Logaritmik dönüşüm.
# değişkende - değerler varsa logaritma alınma.

df = load()
df["Age"] = np.log(df["Age"])
df["Age"].describe().T

# Numeric to Categorical

df = load()

df.loc[(df['Age'] < 18), 'NEW_AGE_CAT'] = 'young'
df.loc[(df['Age'] >= 18) & (df['Age'] < 56), 'NEW_AGE_CAT'] = 'mature'
df.loc[(df['Age'] >= 56), 'NEW_AGE_CAT'] = 'senior'

df.head()


#############################################
# 7. FEATURE EXTRACTION
#############################################

#############################################
# NA_FLAG, BOOL, BINARY
#############################################

df = load()
df["NEW_CABIN_BOOL"] = df["Cabin"].isnull().astype('int')
df["NEW_CABIN_BOOL"] = df["Cabin"].notnull().astype('int')

df.loc[((df['SibSp'] + df['Parch']) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df['SibSp'] + df['Parch']) == 0), "NEW_IS_ALONE"] = "YES"


#############################################
# LETTER COUNT
#############################################

df["NEW_NAME_COUNT"] = df["Name"].str.len()
df.head()

df.groupby("NEW_CABIN_BOOL").agg({"NEW_NAME_COUNT":"mean"})

#############################################
# WORD COUNT
#############################################

df["NEW_NAME_WORD_COUNT"] = df["Name"].apply(lambda x: len(str(x).split(" ")))


#############################################
# OZEL YAPILARI YAKALAMAK
#############################################

df["NEW_NAME_DR"] = df["Name"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
df.head()

df["NEW_NAME_DR"].value_counts()

df.groupby("NEW_NAME_DR").agg({"Survived": ["mean", "count"]})

#############################################
# TITLE'LARI CEKMEK
#############################################

df = load()
df.isnull().sum()
df["Age"].mean()

df['NEW_TITLE'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
df.head()

df[["NEW_TITLE", "Survived", "Age"]].groupby(["NEW_TITLE"]).agg({"Survived": "mean", "Age": ["count", "mean"]})

#############################################
# 8. INTERACTIONS
#############################################

df["NEW_FAMILY_SIZE"] = df["SibSp"] + df["Parch"] + 1
df.head()

df.loc[(df['Sex'] == 'male') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['Sex'] == 'male') & ((df['Age'] > 21) & (df['Age']) < 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['Sex'] == 'male') & (df['Age'] > 50), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['Sex'] == 'female') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['Sex'] == 'female') & ((df['Age'] > 21) & (df['Age']) < 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['Sex'] == 'female') & (df['Age'] > 50), 'NEW_SEX_CAT'] = 'seniorfemale'


df[["NEW_SEX_CAT", "Survived"]].groupby(["NEW_SEX_CAT"]).agg({"Survived": ["count", "mean"]})


df["NEW_AGExPCLASS"] = df["Age"] * df["Pclass"]


#############################################
# 9. END TO END APPLICATION
#############################################

#############################################
# TITANIC
#############################################

df = load()
df.head()
df.shape


#############################################
# 1. FEATURE ENGINEERING
#############################################

df["NEW_CABIN_BOOL"] = df["Cabin"].isnull().astype('int')
df["NEW_NAME_COUNT"] = df["Name"].str.len()
df["NEW_NAME_WORD_COUNT"] = df["Name"].apply(lambda x: len(str(x).split(" ")))
df["NEW_NAME_DR"] = df["Name"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
df['NEW_TITLE'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
df["NEW_FAMILY_SIZE"] = df["SibSp"] + df["Parch"] + 1
df["NEW_AGE_PCLASS"] = df["Age"] * df["Pclass"]

df.loc[((df['SibSp'] + df['Parch']) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df['SibSp'] + df['Parch']) == 0), "NEW_IS_ALONE"] = "YES"

df.loc[(df['Age'] < 18), 'NEW_AGE_CAT'] = 'young'
df.loc[(df['Age'] >= 18) & (df['Age'] < 56), 'NEW_AGE_CAT'] = 'mature'
df.loc[(df['Age'] >= 56), 'NEW_AGE_CAT'] = 'senior'

df.loc[(df['Sex'] == 'male') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['Sex'] == 'male') & ((df['Age'] > 21) & (df['Age']) < 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['Sex'] == 'male') & (df['Age'] > 50), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['Sex'] == 'female') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['Sex'] == 'female') & ((df['Age'] > 21) & (df['Age']) < 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['Sex'] == 'female') & (df['Age'] > 50), 'NEW_SEX_CAT'] = 'seniorfemale'

df.head()
df.shape

df.columns = [col.upper() for col in df.columns]

#############################################
# 2. AYKIRI GOZLEM
#############################################

num_cols = [col for col in df.columns if len(df[col].unique()) > 20
            and df[col].dtypes != 'O'
            and col not in "PASSENGERID"]


df.head()

from helpers.data_prep import check_outlier

for col in num_cols:
    print(col, check_outlier(df, col))


from helpers.data_prep import replace_with_thresholds

for col in num_cols:
    replace_with_thresholds(df, col)


for col in num_cols:
    print(col, check_outlier(df, col))

from helpers.eda import check_df
check_df(df)


#############################################
# 3. EKSIK GOZLEM
#############################################
check_df(df)

from helpers.data_prep import missing_values_table
missing_values_table(df)

df.drop("CABIN", inplace=True, axis=1)
missing_values_table(df)

remove_vars = ["TICKET", "NAME"]
df.drop(remove_vars, inplace=True, axis=1)
df.head()
missing_values_table(df)

df["AGE"] = df["AGE"].fillna(df.groupby("NEW_TITLE")["AGE"].transform("median"))

missing_values_table(df)

df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]
df.loc[(df['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
df.loc[(df['AGE'] >= 18) & (df['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
df.loc[(df['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'

df.loc[(df['SEX'] == 'male') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['SEX'] == 'male') & ((df['AGE'] > 21) & (df['AGE']) < 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['SEX'] == 'male') & (df['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['SEX'] == 'female') & ((df['AGE'] > 21) & (df['AGE']) < 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniorfemale'


df = df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0)


#############################################
# 4. LABEL ENCODING
#############################################

df.head()
df.shape

binary_cols = [col for col in df.columns if len(df[col].unique()) == 2 and df[col].dtypes == 'O']

from helpers.data_prep import label_encoder


for col in binary_cols:
    df = label_encoder(df, col)


#############################################
# 5. RARE ENCODING
#############################################

from helpers.data_prep import rare_analyser

rare_analyser(df, "SURVIVED", 0.05)

from helpers.data_prep import rare_encoder
df = rare_encoder(df, 0.01)
rare_analyser(df, "SURVIVED", 0.01)

df["NEW_NAME_WORD_COUNT"].value_counts()

#############################################
# 5. ONE-HOT ENCODING
#############################################

ohe_cols = [col for col in df.columns if 10 >= len(df[col].unique()) > 2]

from helpers.data_prep import one_hot_encoder
df = one_hot_encoder(df, ohe_cols)

df.head()
df.shape


#############################################
# 6. STANDART SCALER
#############################################

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(df[["AGE"]])
df["AGE"] = scaler.transform(df[["AGE"]])

check_df(df)

#############################################
# 7. MODEL
#############################################

y = df["SURVIVED"]
X = df.drop(["PASSENGERID", "SURVIVED"], axis=1)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier().fit(X, y)
y_pred = rf_model.predict(X)
accuracy_score(y_pred, y)


def plot_importance(model, X, num=len(X)):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': X.columns})
    plt.figure(figsize=(10, 15))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('importances-01.png')
    plt.show()

plot_importance(rf_model, X)