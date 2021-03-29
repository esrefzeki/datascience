# Product Rating
# Sorting Products
# Sorting Products with 5 Star Rated
# Sorting Products with 10 Star Rated
# Sorting Comments with Thumbs_Up / Thumbs_Down (helpful or not) Interactions *

#############################################
# Sorting Comments with Thumbs_Up / Thumbs_Down (helpful or not) Interactions
#############################################

import pandas as pd
import math
import scipy.stats as st
import datetime as dt

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


###################################################
# Score = (Positive ratings) − (Negative ratings)
###################################################


def score_pos_neg_diff(pos, neg):
    return pos - neg


# Item 1: 600 up 400 down
# Item 2: 5500 up 4500 down


# Item 1 Score:
score_pos_neg_diff(600, 400)

# Item 2 Score
score_pos_neg_diff(5500, 4500)


# Item 1 pozitif yüzdesi nedir? Yüzde 60
# Item 2 pozitif yüzdesi nedir? Yüzde 55


###################################################
# Score = Average rating = (Positive ratings) / (Total ratings)
###################################################

def score_average_rating(pos, neg):
    if pos + neg == 0:
        return 0
    return pos / (pos + neg)

score_average_rating(600, 400)
score_average_rating(5500, 4500)



# Comment 1: 2 up 0 down
# Comment 2: 100 up 1 down

score_average_rating(2, 0)
score_average_rating(100, 1)

###################################################
# Wilson Lower Bound Score
###################################################

# p = 0.5

# 600 400

# p = 0.6

# 0.46-0.70

# urun iyidir
# orneklem
# 0.6
# 0.46-0.70



def wilson_lower_bound(pos, neg, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not: Eğer skorlar 1-5 arasıdaysa 1-3 down, 4-5 up olarak işaretlenir ve bernoulli'ye uygun hale getirilir.

    Parameters
    ----------
    pos: int
        pozitif yorum sayısı
    neg: int
        negatif yorum sayısı
    confidence: float
        güven aralığı

    Returns
    -------
    wilson score: float

    """
    n = pos + neg
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * pos / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


wilson_lower_bound(600, 400)
wilson_lower_bound(5500, 4500)

wilson_lower_bound(2, 0)
wilson_lower_bound(100, 1)


###################################################
# Case Study:
###################################################

up = [1115, 454, 258, 253, 220, 227, 127, 75, 60, 67, 38, 11, 26, 44, 1, 0, 6, 15, 20]
down = [43, 35, 26, 19, 9, 16, 8, 8, 4, 9, 1, 0, 0, 5, 0, 0, 0, 0, 3]
comments = pd.DataFrame({"up": up, "down": down})

comments["score_pos_neg_diff"] = comments.apply(lambda x: score_pos_neg_diff(x["up"],
                                                                             x["down"]),
                                                axis=1)


# score_average_rating
comments["score_average_rating"] = comments.apply(lambda x: score_average_rating(x["up"], x["down"]), axis=1)

# wilson_lower_bound
comments["wilson_lower_bound"] = comments.apply(lambda x: wilson_lower_bound(x["up"], x["down"]), axis=1)


comments.sort_values("wilson_lower_bound", ascending=False)



###################################################
# Case Study:
###################################################

up = [15, 70, 14, 4, 2, 5, 8, 37, 21, 52, 28, 147, 61, 30, 23, 40, 37, 61, 54, 18, 12, 68]
down = [0, 2, 2, 2, 15, 2, 6, 5, 23, 8, 12, 2, 1, 1, 5, 1, 2, 6, 2, 0, 2, 2]
comments = pd.DataFrame({"up": up, "down": down})


# score_pos_neg_diff
comments["score_pos_neg_diff"] = comments.apply(lambda x: score_pos_neg_diff(x["up"], x["down"]), axis=1)

# score_average_rating
comments["score_average_rating"] = comments.apply(lambda x: score_average_rating(x["up"], x["down"]), axis=1)

# wilson_lower_bound
comments["wilson_lower_bound"] = comments.apply(lambda x: wilson_lower_bound(x["up"], x["down"]), axis=1)

comments.sort_values("wilson_lower_bound", ascending=False)



###################################################
# PROJE: Rating Product & Sorting Reviews in Amazon
###################################################

# http://jmcauley.ucsd.edu/data/amazon/links.html

# reviewerID - ID of the reviewer, e.g. A2SUAM1J3GNN3B
# asin - ID of the product, e.g. 0000013714
# reviewerName - name of the reviewer
# helpful - helpfulness rating of the review, e.g. 2/3
# reviewText - text of the review
# overall - rating of the product
# summary - summary of the review
# unixReviewTime - time of the review (unix time)
# reviewTime - time of the review (raw)


###################################################
# GÖREV 1: Bir ürünün rating'ini güncel yorumlara göre hesaplayınız ve eski rating ile kıyaslayınız.
###################################################

###################################################
# Adım 1. Aşağıdaki adresten veri setini indiriniz ve veriyi okumak için aşağıdaki kodları kullanınız:
###################################################


# http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics.json.gz

import pandas as pd
import gzip

def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)

def get_df(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


df_ = get_df('datasets/reviews_Electronics_5.json.gz')
df = df_.copy()
df.head()
df.shape

###################################################
# Adım 2. Veri setindeki en fazla yorum alan ürünü bulunuz.
###################################################

# 1. SQL'de en çok satan ürünün id'sini bulunuz ve veriyi sql'de indirgeyerek buraya getiriniz.
# 2. Pandas ile en çok satan ürünün ....


###################################################
# Adım 3. En fazla yorum alan ürüne göre veri setini indirgeyiniz (df_sub)
###################################################

###################################################
# Adım 4. Ürünün ortalama puanı nedir?
###################################################

###################################################
# Adım 5. Tarihe ağırlıklı puan ortalaması hesaplayınız.
###################################################

# day_diff hesaplamak için: (yorum sonrası ne kadar gün geçmiş)
df_sub['reviewTime'] = pd.to_datetime(df_sub['reviewTime'], dayfirst=True)
current_date = pd.to_datetime('2014-12-08 0:0:0')
df_sub["day_diff"] = (current_date - df_sub['reviewTime']).dt.days

# Zamanı çeyrek değerlere göre bölüyorum.
a = df_sub["day_diff"].quantile(0.25)
b = df_sub["day_diff"].quantile(0.50)
c = df_sub["day_diff"].quantile(0.75)


###################################################
# Adım 6. Önceki maddeden gelen a,b,c değerlerine göre ağırlıklı puanı hesaplayınız.
###################################################

###################################################
# Görev 2: Product tanıtım sayfasında görüntülenecek ilk 20 yorumu belirleyiniz.
###################################################

###################################################
# Adım 1. Helpful değişkeni içerisinden 3 değişken türetiniz. 1: helpful_yes, 2: helpful_no,  3: total_vote
###################################################

# Helpful içerisinde 2 değer vardır. Birincisi yorumları faydalı bulan oy sayısı ikincisi toplam oy sayısı.
# Dolayısıyla önce ikisini ayrı ayrı çekmeli sonra da (total_vote - helpful_yes) yaparak helpful_no'yu hesaplamalısınız.

df_sub["helpful_yes"]
df_sub["total_vote"]

df_sub["helpful_no"] = df_sub["total_vote"] - df_sub["helpful_yes"]



###################################################
# Adım 2. score_pos_neg_diff'a göre skorlar oluşturunuz ve df_sub içerisinde score_pos_neg_diff ismiyle kaydediniz.
###################################################

###################################################
# Adım 3. score_average_rating'a göre skorlar oluşturunuz ve df_sub içerisinde score_average_rating ismiyle kaydediniz.
###################################################

##################################################
# Adım 4. wilson_lower_bound'a göre skorlar oluşturunuz ve df_sub içerisinde wilson_lower_bound ismiyle kaydediniz.
###################################################

##################################################
# Adım 5. Ürün sayfasında gösterilecek 20 yorumu belirleyiniz ve sonuçları yorumlayınız.
###################################################










###################################################
# PROJE: Product Sorting using Wilson Lower Bound and Customized Weights
###################################################

# product_sorting verisindeki product'ları wilson_lower_bound'a ve bazı diğer faktörlere göre sıralayınız.

###################################################
# Görev 1:
###################################################
# 5'li skaladaki puanları 2'li skalaya indirgeyiniz ve wilson_lower_bound score'ları hesaplayarak
# veri setine ekleyiniz.

df = pd.read_excel("datasets/product_sorting.xlsx")

df = df[["course_name", "rating", "purchase_count", "commment_count",
         "5_point", "4_point", "3_point", "2_point", "1_point"]]

df = df[df["course_name"].str.contains("Veri Bilimi")]
df["pos"] = df["5_point"] + df["4_point"]
df["neg"] = df["3_point"] + df["2_point"] + df["1_point"]


def wilson_lower_bound(pos, neg, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernolli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not: Eğer skorlar 1-5 arasıdaysa 1-3 negatif, 4-5 pozitif olarak işaretlenir ve bernoulli'ye uygun hale getirilir.

    Parameters
    ----------
    pos: int
        pozitif yorum sayısı
    neg: int
        negatif yorum sayısı
    confidence: float
        güven aralığı

    Returns
    -------
    wilson score: float

    """
    n = pos + neg
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * pos / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


for course in df.index:
    df.loc[course, "WLB_SCORE"] = wilson_lower_bound(df.loc[course, "pos"],
                                                     df.loc[course, "neg"])

df.sort_values("WLB_SCORE", ascending=False).head(20)

# Beklenen çıktı:

# 5   R ile Uygulamalı Veri Bilimi: İstatistik ve Ma... 4.80000  ...    4    0.95272
# 0   (50+ Saat) Python A-Z™: Veri Bilimi ve Machine... 4.80000  ...  237    0.94205
# 3     R ile Veri Bilimi ve Machine Learning (35 Saat) 4.60000  ...   82    0.90197
# 1   Python: Yapay Zeka ve Veri Bilimi için Python ... 4.60000  ...  404    0.90125
# 6   Her Seviyeye Uygun Uçtan Uca Veri Bilimi, Knim... 4.70000  ...   69    0.89923
# 2   Data Science ve Python: Sıfırdan Uzmanlığa Ver... 4.40000  ...  213    0.89760


###################################################
# Görev 2:
###################################################
# Sıralamada ortaya çıkan mantıksızlıkları  WLB, purchase_count, commment_count ve rating ile
# ağırlıklandırma yaparak gideriniz ve tekrar sıralayınız.












