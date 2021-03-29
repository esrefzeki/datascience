############################################
# ASSOCIATION_RULE_LEARNING
############################################

# Amacımız retail_II veri setine birliktelik analizi uygulamak.

# 1. Veri Ön İşleme
#   1. Eksik değer, aykırı değer vs (rfm'deki klasik işler)
#   2. Invoice product (basket product) matrisini oluşturmak
# 2. Birliktelik Kurallarnın Çıkarılması

# Description   NINE DRAWER OFFICE TIDY   SET 2 TEA TOWELS I LOVE LONDON    SPACEBOY BABY GIFT SET
# Invoice
# 536370                              0                                 1                       0
# 536852                              1                                 0                       1
# 536974                              0                                 0                       0
# 537065                              1                                 0                       0
# 537463                              0                                 0                       1

############################################
# Veri Ön İşleme
############################################

# pip install mlxtend
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules

# Hemen verimizi hatırlayalım özlemişizdir.

df_ = pd.read_excel("datasets/online_retail_II.xlsx",
                    sheet_name="Year 2010-2011")
df = df_.copy()
df.info()
df.head()

from helpers.helpers import check_df
check_df(df)

from helpers.helpers import crm_data_prep

df = crm_data_prep(df)
check_df(df)

df_fr = df[df['Country'] == "France"]
check_df(df_fr)

df_fr.groupby(['Invoice', 'StockCode']).agg({"Quantity": "sum"}).head(100)

df_fr.groupby(['Invoice', 'StockCode']).agg({"Quantity": "sum"}).unstack().iloc[0:5, 0:5]

df[(df["StockCode"] == 10002) & (df["Invoice"] == 536370)]


df_fr.groupby(['Invoice', 'StockCode']).\
    agg({"Quantity": "sum"}).\
    unstack().fillna(0).iloc[0:5, 0:5]

df_fr.groupby(['Invoice', 'StockCode']).\
    agg({"Quantity": "sum"}).\
    unstack().fillna(0).\
    applymap(lambda x: 1 if x > 0 else 0).iloc[0:5, 0:5]


def create_invoice_product_df(dataframe):
    return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
        applymap(lambda x: 1 if x > 0 else 0)

fr_inv_pro_df = create_invoice_product_df(df_fr)

fr_inv_pro_df.head()

# Çıtır ödev.
# Her bir invoice'da kaç eşsiz ürün vardır.
# Her bir product kaç eşsiz sepettedir.

############################################
# Birliktelik Kurallarının Çıkarılması
############################################

frequent_itemsets = apriori(fr_inv_pro_df, min_support=0.01, use_colnames=True)
frequent_itemsets.sort_values("support", ascending=False)

rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
rules.head()
rules.sort_values("lift", ascending=False).head()



############################################
# Çalışmanın Fonksiyonlaştırılması
############################################


import pandas as pd
pd.set_option('display.max_columns', None)
from mlxtend.frequent_patterns import apriori, association_rules
from helpers.helpers import crm_data_prep, create_invoice_product_df

df_ = pd.read_excel("datasets/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()

df = crm_data_prep(df)

def create_rules(dataframe, country=False, head=5):
    if country:
        dataframe = dataframe[dataframe['Country'] == country]
        dataframe = create_invoice_product_df(dataframe)
        frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
        print(rules.sort_values("lift", ascending=False).head(head))
    else:
        dataframe = create_invoice_product_df(dataframe)
        frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
        print(rules.sort_values("lift", ascending=False).head(head))

    return rules


rules = create_rules(df)


############################################
# PROJE 4 (Zorunlu): Özelleştirilmiş Ürün Önerileri Geliştirme
############################################

# Yukarıdaki işlemleri Germany için yapınız ve dijital pazarlama departmanına
# kayda değer önerilerinizi paylaşınız.






############################################
# PROJE 4: Özelleştirilmiş Ürün Önerileri Geliştirme
############################################






