import pandas as pd
import numpy as np
import seaborn as sns
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

try:
    df = pd.read_csv("OnlineRetail.csv", encoding='utf-8')
except UnicodeDecodeError:
    print("Unable to decode using utf-8, trying other encodings...")
    try:
        df = pd.read_csv("OnlineRetail.csv", encoding='ISO-8859-1')
    except Exception as e:
        print("Error:", e)

        
# print(df)
# print(df.shape)
# print(df.info)

df_null = round(100*(df.isnull().sum())/len(df), 2)
# print(df_null)

df = df.dropna()

# print(df.head)

df['CustomerID'] = df['CustomerID'].astype(str)


df['Amount'] = df['Quantity'] * df['UnitPrice']
rfm_m = df.groupby('CustomerID')['Amount'].sum()
rfm_m = rfm_m.reset_index()
print(rfm_m.head)



rfm_f = df.groupby('CustomerID')['InvoiceNo'].count()
rfm_f = rfm_f.reset_index()
rfm_f.columns = ['CustomerID', 'Frequency']
print(rfm_f.head)


rfm = pd.merge(rfm_m, rfm_f, on='CustomerID', how='inner')
print(rfm.head)

df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'],format='%d-%m-%Y %H:%M')



max_date = max(df['InvoiceDate'])
print(max_date)



df['Diff'] = max_date - df['InvoiceDate']
print(df.head)



rfm_p = df.groupby('CustomerID')['Diff'].min()
rfm_p = rfm_p.reset_index()
print(rfm_p.head)


rfm_p['Diff'] = rfm_p['Diff'].dt.days
print(rfm_p.head)



rfm = pd.merge(rfm, rfm_p, on='CustomerID', how='inner')
rfm.columns = ['CustomerID', 'Amount', 'Frequency', 'Recency']

rfm = rfm.rename(columns={'Amount': 'Monetary'})
print(rfm.head)


attributes = ['Monetary','Frequency','Recency']
# plt.rcParams['figure.figsize'] = [10,8]
# sns.boxplot(data = rfm[attributes], orient="v", palette="Set2" ,whis=1.5,saturation=1, width=0.7)
# plt.title("Outliers Variable Distribution", fontsize = 14, fontweight = 'bold')
# plt.ylabel("Range", fontweight = 'bold')
# plt.xlabel("Attributes", fontweight = 'bold')
# plt.savefig('outliers.png')




Q1 = rfm.Monetary.quantile(0.05)
Q3 = rfm.Monetary.quantile(0.95)
IQR = Q3 - Q1
rfm = rfm[(rfm.Monetary >= Q1 - 1.5*IQR) & (rfm.Monetary <= Q3 + 1.5*IQR)]

# Removing (statistical) outliers for Recency
Q1 = rfm.Recency.quantile(0.05)
Q3 = rfm.Recency.quantile(0.95)
IQR = Q3 - Q1
rfm = rfm[(rfm.Recency >= Q1 - 1.5*IQR) & (rfm.Recency <= Q3 + 1.5*IQR)]

# Removing (statistical) outliers for Frequency
Q1 = rfm.Frequency.quantile(0.05)
Q3 = rfm.Frequency.quantile(0.95)
IQR = Q3 - Q1
rfm = rfm[(rfm.Frequency >= Q1 - 1.5*IQR) & (rfm.Frequency <= Q3 + 1.5*IQR)]




rfm_df = rfm[['Monetary', 'Frequency', 'Recency']]

# Instantiate
scaler = StandardScaler()

# fit_transform
rfm_scaled = scaler.fit_transform(rfm_df)
print(rfm_scaled.shape)


rfm_scaled = pd.DataFrame(rfm_scaled)
rfm_scaled.columns = ['Monetary', 'Frequency', 'Recency']
print(rfm_scaled.head)