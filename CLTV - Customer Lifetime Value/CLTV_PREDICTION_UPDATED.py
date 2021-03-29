##############################################################
# BG-NBD ve Gamma-Gamma ile CLTV Prediction
##############################################################


# Güncellendi:
# Not: Genel yorumlarımızda herhangi bir değişiklik yok. Her şey aynı.
# Kodlarda bazı iyileştirmeler yapıldı.
# 1. monetary basitleştirildi.
# 2. freq > 1 indirgemesi yapıldı.
# 3. recency kişiselleştirildi. (her bir kişi özelinde son alışveriş tarihi - ilk alışveriş tarihi)
# 4. Önceki maddeye bağlı olarak recency farklı bir recency olduğu için ismi recency_cltv_p olarak değiştirildi.
# 5. recency_weekly değişkeninin ismi "recency_weekly_cltv_p" olarak değiştirildi.
# 6. Sonuç olarak yorumlamalarımızda ve genel gidişatımızda herhangi bir değişiklik yok.


# 4 Adımda CLTV

# 1. Data Preperation
# 2. BG-NBD Modeli ile Expected Sale Forecasting değerlerini hesapla.
# 3. Gamma-Gamma Modeli ile Expected Average Profit değerlerini hesapla.
# 4. BG-NBD ve Gamma-Gamma modelleri ile belirli bir zaman periyodu için CLTV'yi hesapla.


# Önceli formülü hatırlayalım:
# CLTV = (Customer_Value / Churn_Rate) x Profit_margin.
# Customer_Value = Average_Order_Value * Purchase_Frequency


##############################################################
# 1. Data Preperation
##############################################################
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


df_ = pd.read_excel("datasets/online_retail_II.xlsx",
                    sheet_name="Year 2010-2011")
df = df_.copy()
df.shape
df.head()
df.info()
df.describe().T

df.dropna(inplace=True)
df = df[~df["Invoice"].str.contains("C", na=False)]
df = df[df["Quantity"] > 0]

replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")
df.describe().T

df["TotalPrice"] = df["Quantity"] * df["Price"]

df["InvoiceDate"].max()
today_date = dt.datetime(2011, 12, 11)

#############################################
# RFM Table
#############################################

## recency kullanıcıya özel dinamik.
rfm = df.groupby('Customer ID').agg({'InvoiceDate': [lambda date: (date.max() - date.min()).days,
                                                     lambda date: (today_date - date.min()).days],
                                     'Invoice': lambda num: num.nunique(),
                                     'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

rfm.columns = rfm.columns.droplevel(0)

## recency_cltv_p
rfm.columns = ['recency_cltv_p', 'T', 'frequency', 'monetary']

## basitleştirilmiş monetary_avg
rfm["monetary"] = rfm["monetary"] / rfm["frequency"]

rfm.rename(columns={"monetary": "monetary_avg"}, inplace=True)

# BGNBD için WEEKLY RECENCY VE WEEKLY T'nin HESAPLANMASI
## recency_weekly_p
rfm["recency_weekly_p"] = rfm["recency_cltv_p"] / 7
rfm["T_weekly"] = rfm["T"] / 7

# KONTROL
rfm = rfm[rfm["monetary_avg"] > 0]

## freq > 1
rfm = rfm[(rfm['frequency'] > 1)]
rfm["frequency"] = rfm["frequency"].astype(int)

##############################################################
# 2. BG/NBD Modelinin Kurulması
##############################################################

# pip install lifetimes

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(rfm['frequency'],
        rfm['recency_weekly_p'],
        rfm['T_weekly'])

################################################################
# 1 hafta içinde en çok satın alma beklediğimiz 10 müşteri kimdir?
################################################################

bgf.conditional_expected_number_of_purchases_up_to_time(1,
                                                        rfm['frequency'],
                                                        rfm['recency_weekly_p'],
                                                        rfm['T_weekly']).sort_values(ascending=False).head(10)

rfm["expected_number_of_purchases"] = bgf.predict(1,
                                                  rfm['frequency'],
                                                  rfm['recency_weekly_p'],
                                                  rfm['T_weekly'])

rfm.head()

################################################################
# 1 ay içinde en çok satın alma beklediğimiz 10 müşteri kimdir?
################################################################


bgf.predict(4,
            rfm['frequency'],
            rfm['recency_weekly_p'],
            rfm['T_weekly']).sort_values(ascending=False).head(10)

rfm["expected_number_of_purchases"] = bgf.predict(4,
                                                  rfm['frequency'],
                                                  rfm['recency_weekly_p'],
                                                  rfm['T_weekly'])

rfm.sort_values("expected_number_of_purchases", ascending=False).head(20)

################################################################
# 1 Ay içinde tüm Şirketin Beklenen Satış Sayısı Nedir?
################################################################

bgf.predict(4,
            rfm['frequency'],
            rfm['recency_weekly_p'],
            rfm['T_weekly']).sum()

################################################################
# 3 Ayda Tüm Şirketin Beklenen Satış Sayısı Nedir?
################################################################


bgf.predict(4 * 3,
            rfm['frequency'],
            rfm['recency_weekly_p'],
            rfm['T_weekly']).sum()

################################################################
# Tahmin Sonuçlarının Değerlendirilmesi
################################################################

plot_period_transactions(bgf)
plt.show()

##############################################################
# 3. GAMMA-GAMMA Modelinin Kurulması
##############################################################


ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(rfm['frequency'], rfm['monetary_avg'])

ggf.conditional_expected_average_profit(rfm['frequency'],
                                        rfm['monetary_avg']).head(10)

ggf.conditional_expected_average_profit(rfm['frequency'],
                                        rfm['monetary_avg']).sort_values(ascending=False).head(10)

rfm["expected_average_profit"] = ggf.conditional_expected_average_profit(rfm['frequency'],
                                                                         rfm['monetary_avg'])

rfm.sort_values("expected_average_profit", ascending=False).head(20)

##############################################################
# 4. BG-NBD ve GG modeli ile CLTV'nin hesaplanması.
##############################################################


cltv = ggf.customer_lifetime_value(bgf,
                                   rfm['frequency'],
                                   rfm['recency_weekly_p'],
                                   rfm['T_weekly'],
                                   rfm['monetary_avg'],
                                   time=3,  # 3 aylık
                                   freq="W",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)

cltv.head()

cltv.shape
cltv = cltv.reset_index()
cltv.sort_values(by="clv", ascending=False).head(50)
rfm_cltv_final = rfm.merge(cltv, on="Customer ID", how="left")
rfm_cltv_final.head()

# Bundan sonra ne olur?
# Holdout yöntemi ile zamana göre benchmark yapılması gerekir.

##############################################################
# PROJE: BGNBD ve GG Modellerini Kullanarak CLTV tahmini yapınız.
##############################################################

##############################################################
# GÖREV 1
##############################################################
# - 2010-2011 UK müşterileri için 6 aylık CLTV prediction yapınız.
# - Elde ettiğiniz sonuçları yorumlayıp üzerinde değerlendirme yapınız.
# - Mantıksız ya da çok isabetli olduğunu düşündüğünüz sonuçları vurgulayınız.
# - Dikkat! 6 aylık expected sales değil cltv prediction yapılmasını bekliyoruz.
#   Yani direk bgnbd ve gamma modellerini kurarak devam ediniz ve
# - cltv prediction için ay bölümüne 6 giriniz.


##############################################################
# GÖREV 2
##############################################################
# - 2010-2011 UK müşterileri için 1 aylık ve 12 aylık CLTV hesaplayınız.
# - 1 aylık CLTV'de en yüksek olan 10 kişi ile 12 aylık'taki en yüksek 10 kişiyi analiz ediniz.
# - Fark var mı? Varsa sizce neden olabilir?
# Dikkat! Sıfırdan model kurulmasına gerek yoktur.
# Var olan bgf ve ggf üzerinden direk cltv hesaplanabilir.


##############################################################
# GÖREV 3
##############################################################
# 2010-2011 UK müşterileri için 6 aylık CLTV'ye göre tüm müşterilerinizi 3 gruba (segmente) ayırınız ve
# grup isimlerini veri setine ekleyiniz. Örneğin (C, B, A)
# CLTV'ye göre en iyi yüzde 20'yi seçiniz. Ve bunlara top_flag yazınız. yüzde 20'ye 1.
# diğerlerine 0 yazınız.

# 3 grubu veri setindeki diğer değişkenler açısıdan analiz ediniz.
# 3 grup için yönetime 6 aylık aksiyon önerilerinde bulununuz. Kısa kısa.
