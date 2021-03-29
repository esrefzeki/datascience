# Product Rating *
# Sorting Products *
# Sorting Products with 5 Star Rated *
# Sorting Products with 10 Star Rated
# Sorting Comments with Thumbs_Up / Thumbs_Down (helpful or not) Interactions

# Social proof
# The  Wisdom of Crowds


###################################################
# Product Rating
###################################################

import pandas as pd
import math
import scipy.stats as st

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


###################################################
# (50+ Saat) Python A-Z™: Veri Bilimi ve Machine Learning
###################################################

# Kurs Linki: https://www.udemy.com/course/python-egitimi/
# Kurs puanı: 4.8
# Toplam Puan: 4611
# Puan Yüzdeleri: 75, 20, 4, 1, <1
# Yaklaşık Sayısal Karşılıkları: 3458, 922, 184, 46, 6

df = pd.read_csv("datasets/course_reviews.csv")
df.head()

# rating dagılımı
df["Rating"].value_counts()

# sorulan soru dağılımı
df["Questions Asked"].value_counts()

# sorulan soru kırılımında verilen puan
df.groupby("Questions Asked").agg({"Questions Asked": "count",
                                   "Rating": "mean"})



###################################################
# Ortalama Puan
###################################################

df["Rating"].mean()

###################################################
# Puan Zamanlarına Göre Ağırlıklı Ortalama
###################################################

df.head()
df.info()

df['Timestamp'] = pd.to_datetime(df['Timestamp'])
current_date = pd.to_datetime('2021-02-10 0:0:0')
df["days"] = (current_date - df['Timestamp']).dt.days

df.loc[df["days"] <= 30, "Rating"].mean() * 28 / 100 + \
    df.loc[(df["days"] > 30) & (df["days"] <= 90), "Rating"].mean() * 26 / 100 + \
    df.loc[(df["days"] > 90) & (df["days"] <= 180), "Rating"].mean() * 24 / 100 + \
    df.loc[(df["days"] > 180), "Rating"].mean() * 22 / 100


###################################################
# User Kalitesine Göre Ağırlıklı Ortalama
###################################################

df.loc[df["Progress"] <= 10, "Rating"].mean() * 22 / 100 + \
    df.loc[(df["Progress"] > 10) & (df["Progress"] <= 45), "Rating"].mean() * 24 / 100 + \
    df.loc[(df["Progress"] > 45) & (df["Progress"] <= 75), "Rating"].mean() * 26 / 100 + \
    df.loc[(df["Progress"] > 75), "Rating"].mean() * 28 / 100



###################################################
# Sorting Products
###################################################

df = pd.read_excel("datasets/product_sorting.xlsx")
df.head(8)

df.shape

df = df[["course_name", "rating", "purchase_count", "commment_count",
         "5_point", "4_point", "3_point", "2_point", "1_point"]]


###################################################
# Rating'lere göre sıralama
###################################################

df.sort_values("rating", ascending=False).head(20)


###################################################
# commment_count ve purchase_count'a göre sıralama
###################################################

df["com_purc"] = (df["commment_count"] * df["purchase_count"])
df.sort_values("com_purc", ascending=False).head(20)


###################################################
# rating, commment_count ve purchase_count'a göre sıralama
###################################################

df["com_purc"] = (df["commment_count"] * df["purchase_count"] * df["rating"])
df.sort_values("com_purc", ascending=False).head(20)


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(1, 5))
scaler.fit(df[["commment_count"]])
df[["commment_count"]] = scaler.transform(df[["commment_count"]])

scaler = MinMaxScaler(feature_range=(1, 5))
scaler.fit(df[["purchase_count"]])
df[["purchase_count"]] = scaler.transform(df[["purchase_count"]])

df.head()

df["com_purc"] = (df["commment_count"] * 32 / 100 +
                  df["purchase_count"] * 26 / 100 +
                  df["rating"] * 42 / 100)



df.sort_values("com_purc", ascending=False).head(20)

df[df["course_name"].str.contains("Veri Bilimi")].sort_values("com_purc", ascending=False).head(20)


###################################################
# Sorting Products with 5 Star Rated
###################################################

# wilson_lower_bound

def bayesian_rating_products(n, confidence=0.95):
    """
    N yıldızlı puan sisteminde wilson lower bound score'u hesaplamak için kullanılan fonksiyon.
    Parameters
    ----------
    n: list or df
        puanların frekanslarını tutar.
        Örnek: [2, 40, 56, 12, 90] 2 tane 1 puan, 40 tane 2 puan, ... , 90 tane 5 puan.
    confidence: float
        güven aralığı

    Returns
    -------
    BRP score: float
        BRP ya da WLB skorları

    """
    if sum(n) == 0:
        return 0
    K = len(n)
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    N = sum(n)
    first_part = 0.0
    second_part = 0.0
    for k, n_k in enumerate(n):
        first_part += (k + 1) * (n[k] + 1) / (N + K)
        second_part += (k + 1) * (k + 1) * (n[k] + 1) / (N + K)
    score = first_part - z * math.sqrt((second_part - first_part * first_part) / (N + K + 1))
    return score


for course in df.index:
    score = bayesian_rating_products(df.loc[course, ["1_point", "2_point", "3_point", "4_point", "5_point"]].tolist())
    df.loc[course, "brp_score"] = score

df.head(10)


df[df["course_name"].str.contains("Veri Bilimi")].sort_values("brp_score", ascending=False).head(20)


df.apply(lambda x: bayesian_rating_products(x[["1_point", "2_point", "3_point", "4_point", "5_point"]]), axis=1)

###################################################
# Bayesian Rating Products + Diğer Faktorler
###################################################

df["com_purc"] = (df["commment_count"] * 27 / 100 +
                  df["purchase_count"] * 20 / 100 +
                  df["brp_score"] * 28 / 100 +
                  df["rating"] * 25 / 100)


df[df["course_name"].str.contains("Veri Bilimi")].sort_values("com_purc", ascending=False).head(20)



## AFTER PARTY
###################################################
# İŞ KARAR NOKTASI: SORULARA YANIT VERMELİ Mİ?
###################################################

# soru sorup yanıt alamayan kişilerle soru sorup yanıt alan kişilerin verdiği
# puan ortalamaları arasında istatistiki olarak anlamlı bir farklılık var mıdır.

# sorularına yanıt alamayanlar
df[(df["Questions Asked"] > 0) & (df["Questions Answered"] < 1)]["Rating"].mean()

# sorularına yanıt alanlar
df[(df["Questions Asked"] > 0) & (df["Questions Answered"] >= 1)]["Rating"].mean()

# Bir fark var gibi ama sizce bu fark gerçekten var mı?

# H0: Normal dağılım varsayımı sağlanmaktadır.
# H1:..sağlanmamaktadır.
# p-value < ise 0.05'ten HO RED.
# p-value < değilse 0.05 H0 REDDEDILEMEZ.

from scipy.stats import shapiro
from scipy import stats

test_istatistigi, pvalue = shapiro(df[(df["Questions Asked"] > 0) & (df["Questions Answered"] < 1)]["Rating"])
print('Test İstatistiği = %.4f, p-değeri = %.4f' % (test_istatistigi, pvalue))
test_istatistigi, pvalue = shapiro(df[(df["Questions Asked"] > 0) & (df["Questions Answered"] >= 1)]["Rating"])
print('Test İstatistiği = %.4f, p-değeri = %.4f' % (test_istatistigi, pvalue))

# H0: M1 = M2 (... iki grup ortalamaları arasında ist ol.anl.fark yoktur.)
# H1: M1 != M2 (...vardır)

test_istatistigi, pvalue = stats.mannwhitneyu(
    df[(df["Questions Asked"] > 0) & (df["Questions Answered"] < 1)]["Rating"],
    df[(df["Questions Asked"] > 0) & (df["Questions Answered"] >= 1)]["Rating"])
print('Test İstatistiği = %.4f, p-değeri = %.4f' % (test_istatistigi, pvalue))



###################################################
# İŞ KARAR NOKTASI: KURSUN BÜYÜK ÇOĞUNLUĞUNU İZLEYENLER KURSU NASIL DEĞERLENDİRİYOR?
###################################################


# kursun büyük çoğunluğunu izleyenler
df[(df["Progress"] > 75)]["Rating"].mean()

# kursun büyük çoğunluğunu izlemeyenler
df[(df["Progress"] < 75)]["Rating"].mean()

# H0: M1 = M2 (... iki grup ortalamaları arasında ist ol.anl.fark yoktur.)
# H1: M1 != M2 (...vardır)

test_istatistigi, pvalue = stats.mannwhitneyu(df[(df["Progress"] > 75)]["Rating"],
                                              df[(df["Progress"] < 75)]["Rating"])
print('Test İstatistiği = %.4f, p-değeri = %.4f' % (test_istatistigi, pvalue))

###################################################
# İŞ KARAR NOKTASI: İLERLEMESİ DÜŞÜK OLUP SORULARINA YANIT ALAMAYANLAR İLE İLERLEMESİ
# YÜKSEK OLUP SORUSUNA YANIT ALAMAYANLAR
###################################################

df[(df["Questions Asked"] > 0) & (df["Questions Answered"] < 1) & (df["Progress"] < 25)]["Rating"].mean()
df[(df["Questions Asked"] > 0) & (df["Questions Answered"] < 1) & (df["Progress"] > 75)]["Rating"].mean()

test_istatistigi, pvalue = stats.mannwhitneyu(
    df[(df["Questions Asked"] > 0) & (df["Questions Answered"] < 1) & (df["Progress"] < 25)]["Rating"],
    df[(df["Questions Asked"] > 0) & (df["Questions Answered"] < 1) & (df["Progress"] > 75)]["Rating"])
print('Test İstatistiği = %.4f, p-değeri = %.4f' % (test_istatistigi, pvalue))
