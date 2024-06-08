import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


file_path = "//content//Unemployment in India.csv"
data = pd.read_csv("//content//Unemployment in India.csv")


data.columns = data.columns.str.strip()


data_info = data.info()
data_head = data.head()
data_description = data.describe()
data_null_sum = data.isnull().sum()


if 'Date' in data.columns:
    data['Date'] = pd.to_datetime(data['Date'])


plt.figure(figsize=(12, 6))
sns.lineplot(x='Date', y='Estimated Unemployment Rate (%)', data=data)
plt.title('Unemployment Rate Over Time')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate (%)')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

data_info, data_head, data_description, data_null_sum
