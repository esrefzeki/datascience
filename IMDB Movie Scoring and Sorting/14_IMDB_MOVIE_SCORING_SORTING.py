# Product Rating
# Sorting Products
# Sorting Products with 5 Star Rated
# Sorting Products with 10 Star Rated *
# Sorting Comments with Thumbs_Up / Thumbs_Down (helpful or not) Interactions

###################################################
# IMDB Movie Scoring & Sorting
###################################################



import pandas as pd
import math
import scipy.stats as st

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df = pd.read_csv("datasets/the_movies_dataset/movies_metadata.csv",
                 low_memory=False)  # DtypeWarning kapamak icin
df.head()
df.shape
df.info()

df["imdb_id"].nunique()

from helpers.eda import cat_summary, grab_col_names, num_summary


cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(df)


for col in cat_cols:
    cat_summary(df, col)


for col in num_cols:
    num_summary(df, col)


df_sub = df[["title", "vote_average", "vote_count"]]

###################################################
# vote_average
###################################################

df_sub.sort_values("vote_average", ascending=False).head(20)

num_summary(df_sub, "vote_count")


df_sub.sort_values("vote_count", ascending=False).head(20)

df_sub[df_sub["vote_count"] > 400].sort_values("vote_average", ascending=False).head(20)



###################################################
# vote_average * vote_count
###################################################

df_sub["weighted_average"] = df_sub["vote_average"] * df_sub["vote_count"]
df_sub.sort_values("weighted_average", ascending=False).head(20)


###################################################
# weighted_rating
###################################################

# weighted_rating = (v/(v+m) * r) + (m/(v+m) * c)

# r = vote average
# v = vote count
# M = minimum votes required to be listed in the Top 250
# C = the mean vote across the whole report (currently 7.0)

M = 2500
C = df_sub['vote_average'].mean()


def weighted_rating(r, v, M, C):
    return (v / (v + M) * r) + (M / (v + M) * C)

df_sub.sort_values("weighted_average", ascending=False).head(20)

weighted_rating(8.30000, 9678, M, C)

df_sub["weighted_rating"] = weighted_rating(df_sub["vote_average"], df_sub["vote_count"], M, C)

df_sub.sort_values("weighted_rating", ascending=False).head(20)

# https://help.imdb.com/article/imdb/track-movies-tv/weighted-average-ratings/GWT2DSBYVT2F25SK?ref_=cons_tt_rt_wtavg#



###################################################
# bayesian_rating_products
###################################################

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

bayesian_rating_products([34733, 4355, 4704, 6561, 13515, 26183, 87368, 273082, 600260, 1295351])

bayesian_rating_products([37128, 5879, 6268, 8419, 16603, 30016, 78538, 199430, 402518, 837905])

bayesian_rating_products([30345, 7172, 8083, 11429, 23236, 49482, 137745, 354608, 649114, 1034843])



df = pd.read_csv("datasets/imdb_ratings.csv")
df = df.iloc[0:, 1:]
df.head()
df.shape


df["brp_score"] = df.apply(lambda x: bayesian_rating_products(x[["one", "two", "three", "four", "five",
                                                                  "six", "seven", "eight", "nine", "ten"]]), axis=1)

df.sort_values("brp_score", ascending=False).head(20)