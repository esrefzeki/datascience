###########################
# AB Testing & Dynamic Pricing
###########################

# Sampling
# Descriptive Statistics
# Confidence Interval
# AB Testing (Bağımsız İki Örneklem T Testi)
# AB Testing (İki Örneklem Oran Testi)

# Projeler (caseler)
# AB Testing for Facebook Bidding Strategies
# Dynamic Pricing for Item Price


############################
# Sampling
############################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

populasyon = np.random.randint(0, 80, 10000)
populasyon.mean()

np.random.seed(115)
orneklem = np.random.choice(a=populasyon, size=100)
orneklem.mean()

np.random.seed(10)
orneklem1 = np.random.choice(a=populasyon, size=100)
orneklem2 = np.random.choice(a=populasyon, size=100)
orneklem3 = np.random.choice(a=populasyon, size=100)
orneklem4 = np.random.choice(a=populasyon, size=100)
orneklem5 = np.random.choice(a=populasyon, size=100)
orneklem6 = np.random.choice(a=populasyon, size=100)
orneklem7 = np.random.choice(a=populasyon, size=100)
orneklem8 = np.random.choice(a=populasyon, size=100)
orneklem9 = np.random.choice(a=populasyon, size=100)
orneklem10 = np.random.choice(a=populasyon, size=100)

(orneklem1.mean() + orneklem2.mean() + orneklem3.mean() + orneklem4.mean() + orneklem5.mean()
 + orneklem6.mean() + orneklem7.mean() + orneklem8.mean() + orneklem9.mean() + orneklem10.mean()) / 10


orneklem1.mean()
orneklem7.mean()
orneklem8.mean()



############################
# Descriptive Statistics
############################

import seaborn as sns
df = sns.load_dataset("tips")
df.describe().T

df.head()
df["sex"].value_counts()
df[["tip", "total_bill"]].corr()


############################
# Confidence Interval
############################

import statsmodels.stats.api as sms
df = sns.load_dataset("tips")
df.describe().T

sms.DescrStatsW(df["total_bill"]).tconfint_mean()
sms.DescrStatsW(df["tip"]).tconfint_mean()

df = pd.read_csv("datasets/titanic.csv")
df.describe().T
sms.DescrStatsW(df["Age"].dropna()).tconfint_mean()
sms.DescrStatsW(df["Fare"].dropna()).tconfint_mean()

df_ = pd.read_excel("datasets/online_retail_II.xlsx",
                    sheet_name="Year 2010-2011")

df = df_.copy()

sms.DescrStatsW(df["Quantity"].dropna()).tconfint_mean()
sms.DescrStatsW(df["Price"].dropna()).tconfint_mean()


############################
# AB Testing (Bağımsız İki Örneklem T Testi)
############################

# İki grup ortalaması arasında karşılaştırma yapılmak istenildiğinde kullanılır.

# Soru: tips veri setinde sigara içenlerle içmeyenlerin hesap ödemeleri
# ortalaması arasında fark var mı?

df = sns.load_dataset("tips")
df.groupby("smoker").agg({"total_bill": "mean"})


############################
# 1. Varsayım Kontrolü
############################

# 1.1 Normallik Varsayımı
# 1.2 Varyans Homojenliği

############################
# 1.1 Normallik Varsayımı
############################

# H0: Normal dağılım varsayımı sağlanmaktadır.
# H1:..sağlanmamaktadır.

from scipy.stats import shapiro

df.loc[df["smoker"] == "Yes", "total_bill"]
df.loc[df["smoker"] == "No", "total_bill"]

test_istatistigi, pvalue = shapiro(df.loc[df["smoker"] == "Yes", "total_bill"])
print('Test İstatistiği = %.4f, p-değeri = %.4f' % (test_istatistigi, pvalue))

# p-value < ise 0.05'ten HO RED.
# p-value < değilse 0.05 H0 REDDEDILEMEZ.
# yes grubu için varsayım sağlanmamaktadır.

test_istatistigi, pvalue = shapiro(df.loc[df["smoker"] == "No", "total_bill"])
print('Test İstatistiği = %.4f, p-değeri = %.4f' % (test_istatistigi, pvalue))

# p-value < ise 0.05'ten HO RED.
# p-value < değilse 0.05 H0 REDDEDILEMEZ.
# no grubu için varsayım sağlanmamaktadır.


############################
# 1.2 Varyans Homojenligi Varsayımı
############################

# H0: Varyanslar Homojendir
# H1: Varyanslar Homojen Değildir

from scipy import stats
stats.levene(df.loc[df["smoker"] == "Yes", "total_bill"],
             df.loc[df["smoker"] == "No", "total_bill"])


# p-value < ise 0.05'ten HO RED.
# p-value < değilse 0.05 H0 REDDEDILEMEZ.
# Varyanslar homojen değildir.


############################
# 2. Hipotezin Uygulanması
############################

# parametrik - nonparametrik
# mean - medyan

# 1.1 Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi (parametrik test)
# 1.2 Varsayımlar sağlanmıyorsa mannwhitneyu testi (non-parametrik test)

# Eger normallik sağlanır varyans homojenliği sağlanmazsa ne olacak?
# T test fonksiyonuna arguman gireceğiz.

# Eğer normallik sağlanmazsa her türlü nonparametrik test yapacağız.

# H0: M1 = M2 (... iki grup ortalamaları arasında ist ol.anl.fark yoktur.)
# H1: M1 != M2 (...vardır)

# Şimdi varsayım sağlanmış gibi kabul ediyoruz.
# Aslında varsayım sağlanmadı.
# Sağlanmış gibi kabul edip önce parametrik testi yapacağız sonra nonparametrik.

# 1.1 Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi (parametrik test)
test_istatistigi, pvalue = stats.ttest_ind(df.loc[df["smoker"] == "Yes", "total_bill"],
                                           df.loc[df["smoker"] == "No", "total_bill"],
                                           equal_var=True)

print('Test İstatistiği = %.4f, p-değeri = %.4f' % (test_istatistigi, pvalue))

# p-value < ise 0.05 'ten HO RED.
# p-value < değilse 0.05 H0 REDDEDILEMEZ.

# 1.2 Varsayımlar sağlanmıyorsa mannwhitneyu testi (non-parametrik test)
test_istatistigi, pvalue = stats.mannwhitneyu(df.loc[df["smoker"] == "Yes", "total_bill"],
                                              df.loc[df["smoker"] == "No", "total_bill"])


print('Test İstatistiği = %.4f, p-değeri = %.4f' % (test_istatistigi, pvalue))

# H0: M1 = M2 (... iki grup ortalamaları arasında ist ol.anl.fark yoktur.)
# H1: M1 != M2 (...vardır)

# p-value < ise 0.05 'ten HO RED.
# p-value < değilse 0.05 H0 REDDEDILEMEZ.

############################
# Hızlı Birkaç Uygulama
############################

############################
# Titanic kadın ve erkek yolcuların yaş ortalamaları arasında istatistiksel olarak anl. fark. var mıdır?
############################

df = pd.read_csv("datasets/titanic.csv")
df.groupby("Sex").agg({"Age": "mean"})

# Normallik varsayımı
# H0: Normal dağılım varsayımı sağlanmaktadır.
# H1:..sağlanmamaktadır.

test_istatistigi, pvalue = shapiro(df.loc[df["Sex"] == "female", "Age"].dropna())
print('Test İstatistiği = %.4f, p-değeri = %.4f' % (test_istatistigi, pvalue))
# p-value < ise 0.05 'ten HO RED.
# p-value < değilse 0.05 H0 REDDEDILEMEZ.

test_istatistigi, pvalue = shapiro(df.loc[df["Sex"] == "male", "Age"].dropna())
print('Test İstatistiği = %.4f, p-değeri = %.4f' % (test_istatistigi, pvalue))
# p-value < ise 0.05 'ten HO RED.
# p-value < değilse 0.05 H0 REDDEDILEMEZ.

# Varyans homojenliği
# H0: Varyanslar Homojendir
# H1: Varyanslar Homojen Değildir

test_istatistigi, pvalue = stats.levene(df.loc[df["Sex"] == "female", "Age"].dropna(),
                                        df.loc[df["Sex"] == "male", "Age"].dropna())
print('Test İstatistiği = %.4f, p-değeri = %.4f' % (test_istatistigi, pvalue))
# p-value < ise 0.05 'ten HO RED.
# p-value < değilse 0.05 H0 REDDEDILEMEZ.

# H0: M1 = M2 (... iki grup ortalamaları arasında ist ol.anl.fark yoktur.)
# H1: M1 != M2 (...vardır)

# mannwhitneyu
test_istatistigi, pvalue = stats.mannwhitneyu(df.loc[df["Sex"] == "female", "Age"].dropna(),
                                              df.loc[df["Sex"] == "male", "Age"].dropna())

print('Test İstatistiği = %.4f, p-değeri = %.4f' % (test_istatistigi, pvalue))

# p-value < ise 0.05 'ten HO RED.
# p-value < değilse 0.05 H0 REDDEDILEMEZ.

############################
# Diyabet hastası olan ya da olmayanların yaşları arasında ist. ol. anl. fark var mıdır?
############################

df = pd.read_csv("datasets/diabetes.csv")
df.groupby("Outcome").agg({"Age": "mean"})

# Normallik Varsayımı (H0: Normal dağılım varsayımı sağlanmaktadır.)
test_istatistigi, pvalue = shapiro(df.loc[df["Outcome"] == 1, "Age"].dropna())
print('Test İstatistiği = %.4f, p-değeri = %.4f' % (test_istatistigi, pvalue))
# p-value < ise 0.05 'ten HO RED.
# p-value < değilse 0.05 H0 REDDEDILEMEZ.

test_istatistigi, pvalue = shapiro(df.loc[df["Outcome"] == 0, "Age"].dropna())
print('Test İstatistiği = %.4f, p-değeri = %.4f' % (test_istatistigi, pvalue))
# p-value < ise 0.05 'ten HO RED.
# p-value < değilse 0.05 H0 REDDEDILEMEZ.

# Varyans Homojenliği Varsayımı (H0: Varyanslar homojendir)
test_istatistigi, pvalue = stats.levene(df.loc[df["Outcome"] == 1, "Age"].dropna(),
                                        df.loc[df["Outcome"] == 0, "Age"].dropna())

print('Test İstatistiği = %.4f, p-değeri = %.4f' % (test_istatistigi, pvalue))
# p-value < ise 0.05 'ten HO RED.
# p-value < değilse 0.05 H0 REDDEDILEMEZ.

# Testin yapılması (H0: M1 = M2)
test_istatistigi, pvalue = stats.mannwhitneyu(df.loc[df["Outcome"] == 1, "Age"].dropna(),
                                              df.loc[df["Outcome"] == 0, "Age"].dropna())
print('Test İstatistiği = %.4f, p-değeri = %.4f' % (test_istatistigi, pvalue))
# p-value < ise 0.05 'ten HO RED.
# p-value < değilse 0.05 H0 REDDEDILEMEZ.

##################################
# Retail data setimizdeki Germany ve UK denki gelirler farklı mıdır diye bir hipotez testide oluşturaiblir değil mi
##################################
df = df_.copy()
df.head()


df.groupby("Country").agg({"Price": "mean"})

# Normallik Varsayımı (H0: Normal dağılım varsayımı sağlanmaktadır.)
test_istatistigi, pvalue = shapiro(df.loc[df["Country"] == "United Kingdom", "Price"].dropna())
print('Test İstatistiği = %.4f, p-değeri = %.4f' % (test_istatistigi, pvalue))

# Normallik Varsayımı (H0: Normal dağılım varsayımı sağlanmaktadır.)
test_istatistigi, pvalue = shapiro(df.loc[df["Country"] == "Germany", "Price"].dropna())
print('Test İstatistiği = %.4f, p-değeri = %.4f' % (test_istatistigi, pvalue))

# Varyans Homojenliği Varsayımı (H0: Varyanslar homojendir)
test_istatistigi, pvalue = stats.levene(df.loc[df["Country"] == "United Kingdom", "Price"].dropna(),
                                        df.loc[df["Country"] == "Germany", "Price"].dropna())
print('Test İstatistiği = %.4f, p-değeri = %.4f' % (test_istatistigi, pvalue))

# Testin yapılması (H0: M1 = M2)
test_istatistigi, pvalue = stats.mannwhitneyu(df.loc[df["Country"] == "United Kingdom", "Price"].dropna(),
                                              df.loc[df["Country"] == "Germany", "Price"].dropna())
print('Test İstatistiği = %.4f, p-değeri = %.4f' % (test_istatistigi, pvalue))



############################
# AB Testing (İki Örneklem Oran Testi)
############################

# H0: Yeşil butonunun dönüşüm oranı ile kırmızı butonun dönüşüm oranı arasında ist.ol.anlamlı farklılık yoktur.
# H1: ... vardır


from statsmodels.stats.proportion import proportions_ztest
basari_sayisi = np.array([300, 250])
gozlem_sayilari = np.array([1000, 1100])

proportions_ztest(count=basari_sayisi, nobs=gozlem_sayilari)

############################
# Korelasyon Analizi
############################

# Bahşiş veri seti:
# total_bill: yemeğin toplam fiyatı (bahşiş ve vergi dahil)
# tip: bahşiş
# sex: ücreti ödeyen kişinin cinsiyeti (0=male, 1=female)
# smoker: grupta sigara içen var mı? (0=No, 1=Yes)
# day: gün (3=Thur, 4=Fri, 5=Sat, 6=Sun)
# time: ne zaman? (0=Day, 1=Night)
# size: grupta kaç kişi var?


df = sns.load_dataset('tips')
df["total_bill"] = df["total_bill"] - df["tip"]

df.head()

df.plot.scatter("total_bill", "tip")
plt.show()

###########################
# Varsayım Kontrolü
###########################

# Varsayım sağlanıyorsa pearson sağlanmıyorsa Spearman.

# H0: Normal dağılım varsayımı sağlanmaktadır.
# H1:..sağlanmamaktadır.

from scipy.stats import shapiro
test_istatistigi, pvalue = shapiro(df["tip"])
print('Test İstatistiği = %.4f, p-değeri = %.4f' % (test_istatistigi, pvalue))

test_istatistigi, pvalue = shapiro(df["total_bill"])
print('Test İstatistiği = %.4f, p-değeri = %.4f' % (test_istatistigi, pvalue))


###########################
# Hipotez Testi
###########################

# H0: Değişkenler arasında korelasyon yoktur.
# H1: Değişkenler arasında korelasyon vardır.


# Korelasyonunu Anlamlılığının Testi
from scipy.stats.stats import pearsonr
test_istatistigi, pvalue = pearsonr(df["tip"], df["total_bill"])
print('Korelasyon Katsayısı = %.4f, p-değeri = %.4f' % (test_istatistigi, pvalue))


# Nonparametrik Hipotez Testi
from scipy.stats import stats
stats.spearmanr(df["tip"], df["total_bill"])

test_istatistigi, pvalue = stats.spearmanr(df["tip"],df["total_bill"])
print('Korelasyon Katsayısı = %.4f, p-değeri = %.4f' % (test_istatistigi, pvalue))

test_istatistigi, pvalue = stats.kendalltau(df["tip"], df["total_bill"])
print('Korelasyon Katsayısı = %.4f, p-değeri = %.4f' % (test_istatistigi, pvalue))


# PRICING: Item fiyatı ne olmalı!
# Bir oyun şirketi bir oyununda kullanıcılarına item satın alımları için hediye paralar vermiştir.
# Kullanıcılar bu sanal paraları kullanarak karakterlerine çeşitli araçlar satın almaktadır. Oyun şirketi bir item için fiyat belirtmemiş ve kullanıcılardan bu item'ı istedikleri fiyattan almalarını sağlamış. Örneğin kalkan isimli item için kullanıcılar kendi uygun gördükleri miktarları ödeyerek bu kalkanı satın alacaklar. Örneğin bir kullanıcı kendisine verilen sanal paralardan 30 birim, diğer kullanıcı 45 birim ile ödeme yapabilir. Dolayısıyla kullanıcılar kendilerine göre ödemeyi göze aldıkları miktarlar ile bu item'ı satın alabilirler.
#
# Çözülmesi gereken problemler:
# Item'in fiyatı kategorilere göre farklılık göstermekte midir? İstatistiki olarak ifade ediniz.
# İlk soruya bağlı olarak item'ın fiyatı ne olmalıdır? Nedenini açıklayınız?
# Fiyat konusunda "hareket edebilir olmak" istenmektedir. Fiyat stratejisi için karar destek sistemi oluşturunuz.
# Olası fiyat değişiklikleri için item satın almalarını ve gelirlerini simüle ediniz.
