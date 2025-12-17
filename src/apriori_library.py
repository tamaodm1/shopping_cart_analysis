import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import os
import plotly.express as px
# QUAY VỀ IMPORT APRIORI CHUẨN (Đúng yêu cầu PDF)
from mlxtend.frequent_patterns import apriori, association_rules

# ==========================================
# CLASS 1: LÀM SẠCH DỮ LIỆU
# ==========================================
class DataCleaner:
    def __init__(self, path):
        self.path = path
        self.df = None

    def load_data(self):
        if self.path:
            self.df = pd.read_csv(self.path, encoding="ISO-8859-1")
        return self.df

    def clean_data(self):
        if self.df is None: self.load_data()
        print("--- [1/5] Đang làm sạch dữ liệu ---")
        self.df['InvoiceNo'] = self.df['InvoiceNo'].astype(str)
        self.df = self.df[~self.df['InvoiceNo'].str.startswith('C')]
        self.df = self.df[self.df['Country'] == "United Kingdom"]
        self.df = self.df.dropna(subset=['CustomerID'])
        self.df = self.df[(self.df['Quantity'] > 0) & (self.df['UnitPrice'] > 0)]
        self.df['TotalPrice'] = self.df['Quantity'] * self.df['UnitPrice']
        return self.df

    def create_time_features(self):
        print("--- [2/5] Đang tạo đặc trưng thời gian ---")
        if self.df is None: return None
        self.df['InvoiceDate'] = pd.to_datetime(self.df['InvoiceDate'])
        self.df['Hour'] = self.df['InvoiceDate'].dt.hour
        self.df['Month'] = self.df['InvoiceDate'].dt.month
        self.df['DayOfWeek'] = self.df['InvoiceDate'].dt.day_name()
        return self.df

    def compute_rfm(self):
        print("--- [3/5] Đang tính toán RFM ---")
        if self.df is None: return None
        self.df['InvoiceDate'] = pd.to_datetime(self.df['InvoiceDate'])
        snapshot_date = self.df['InvoiceDate'].max() + pd.Timedelta(days=1)
        rfm = self.df.groupby(['CustomerID']).agg({
            'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
            'InvoiceNo': 'nunique',
            'TotalPrice': 'sum'
        })
        rfm.rename(columns={'InvoiceDate': 'Recency', 'InvoiceNo': 'Frequency', 'TotalPrice': 'Monetary'}, inplace=True)
        return rfm

    def save_cleaned_data(self, output_dir):
        print(f"--- [4/5] Đang lưu dữ liệu đã sạch xuống: {output_dir} ---")
        if self.df is None: return
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "cleaned_data.csv")
        self.df.to_csv(save_path, index=False)
        print(f"-> Đã lưu xong: {save_path}")

# ==========================================
# CLASS 2: CHUẨN BỊ GIỎ HÀNG
# ==========================================
class BasketPreparer:
    def __init__(self, df, invoice_col='InvoiceNo', item_col='Description', quantity_col='Quantity'):
        self.df = df
        self.invoice_col = invoice_col
        self.item_col = item_col
        self.quantity_col = quantity_col
        self.basket = None
        self.basket_bool = None

    def create_basket(self):
        print("--- Đang tạo giỏ hàng ---")
        self.basket = (self.df.groupby([self.invoice_col, self.item_col])[self.quantity_col]
                       .sum().unstack().reset_index().fillna(0)
                       .set_index(self.invoice_col))
        return self.basket

    def encode_basket(self, threshold=1):
        if self.basket is None: self.create_basket()
        print(f"--- Đang mã hóa One-Hot (Ngưỡng={threshold}) ---")
        self.basket_bool = self.basket.applymap(lambda x: 1 if x >= threshold else 0)
        return self.basket_bool
    
    def save_basket_bool(self, output_path):
        print(f"--- Đang lưu file Basket Bool xuống: {output_path} ---")
        if self.basket_bool is None: return
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.basket_bool.index = self.basket_bool.index.astype(str)
        self.basket_bool.to_parquet(output_path)
        print(f"-> Đã lưu xong: {output_path}")

# ==========================================
# CLASS 3: KHAI PHÁ LUẬT (DÙNG APRIORI CHUẨN)
# ==========================================
class AssociationRulesMiner:
    def __init__(self, basket_bool):
        self.basket_bool = basket_bool
        self.frequent_itemsets = None
        self.rules = None

    def mine_frequent_itemsets(self, min_support=0.01, max_len=None, use_colnames=True):
        print(f"--- Đang tìm tập phổ biến (Apriori, Support={min_support}) ---")
        # Dùng apriori chuẩn + low_memory=True để giảm tải RAM mà vẫn đúng đề bài
        self.frequent_itemsets = apriori(self.basket_bool, 
                                         min_support=min_support, 
                                         max_len=max_len,
                                         use_colnames=use_colnames,
                                         low_memory=True) # <-- Cờ quan trọng
        return self.frequent_itemsets

    def generate_rules(self, metric="lift", min_threshold=1.2):
        if self.frequent_itemsets is None: return None
        print(f"--- Đang sinh luật (Metric={metric}) ---")
        self.rules = association_rules(self.frequent_itemsets, 
                                       metric=metric, 
                                       min_threshold=min_threshold)
        return self.rules

    def add_readable_rule_str(self):
        if self.rules is None: return None
        self.rules['antecedents_str'] = self.rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        self.rules['consequents_str'] = self.rules['consequents'].apply(lambda x: ', '.join(list(x)))
        return self.rules

    def filter_rules(self, min_support, min_confidence, min_lift, max_len_antecedents, max_len_consequents):
        if self.rules is None: return None
        filtered = self.rules.copy()
        filtered = filtered[
            (filtered['support'] >= min_support) &
            (filtered['confidence'] >= min_confidence) &
            (filtered['lift'] >= min_lift) &
            (filtered['antecedents'].apply(len) <= max_len_antecedents) &
            (filtered['consequents'].apply(len) <= max_len_consequents)
        ]
        return filtered

    def save_rules(self, output_path, rules_df):
        print(f"--- Đang lưu luật xuống file: {output_path} ---")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        rules_df.to_csv(output_path, index=False)
        print("-> Đã lưu xong.")

# ==========================================
# CLASS 4: TRỰC QUAN HÓA
# ==========================================
class DataVisualizer:
    def __init__(self, rules=None):
        self.rules = rules

    # --- Phần EDA ---
    def plot_revenue_over_time(self, df):
        df.groupby('Month')['TotalPrice'].sum().plot(kind='line')
        plt.title('Doanh thu theo tháng')
        plt.show()

    def plot_time_patterns(self, df):
        df.groupby('Hour')['InvoiceNo'].nunique().plot(kind='bar')
        plt.title('Số đơn hàng theo giờ')
        plt.show()

    def plot_product_analysis(self, df, top_n=10):
        df['Description'].value_counts().head(top_n).plot(kind='barh')
        plt.title('Top sản phẩm bán chạy')
        plt.show()
    
    def plot_customer_distribution(self, df):
        df.groupby('Country')['CustomerID'].nunique().plot(kind='bar')
        plt.title('Phân bố khách hàng')
        plt.show()

    def plot_rfm_analysis(self, rfm):
        sns.histplot(rfm['Recency'])
        plt.title('Phân phối Recency')
        plt.show()

    # --- Phần Apriori ---
    def plot_itemset_length_distribution(self, frequent_itemsets):
        frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(len)
        sns.countplot(x='length', data=frequent_itemsets)
        plt.title('Phân phối độ dài tập phổ biến')
        plt.show()

    def plot_top_frequent_itemsets(self, frequent_itemsets, top_n=10, min_len=1):
        df_plot = frequent_itemsets[frequent_itemsets['itemsets'].apply(len) >= min_len].copy()
        df_plot = df_plot.sort_values('support', ascending=False).head(top_n)
        df_plot['items_str'] = df_plot['itemsets'].apply(lambda x: ', '.join(list(x)))
        plt.figure(figsize=(10, 6))
        sns.barplot(x='support', y='items_str', data=df_plot, palette='viridis')
        plt.title(f'Top {top_n} tập phổ biến')
        plt.show()

    def plot_top_rules_lift(self, rules_df, top_n=10):
        top_rules = rules_df.sort_values('lift', ascending=False).head(top_n)
        top_rules['rule_name'] = top_rules['antecedents_str'] + " -> " + top_rules['consequents_str']
        plt.figure(figsize=(10, 6))
        sns.barplot(x="lift", y="rule_name", data=top_rules, palette="magma")
        plt.title(f'Top {top_n} luật theo Lift')
        plt.show()

    def plot_top_rules_confidence(self, rules_df, top_n=10):
        top_rules = rules_df.sort_values('confidence', ascending=False).head(top_n)
        top_rules['rule_name'] = top_rules['antecedents_str'] + " -> " + top_rules['consequents_str']
        plt.figure(figsize=(10, 6))
        sns.barplot(x="confidence", y="rule_name", data=top_rules, palette="Blues_r")
        plt.title(f'Top {top_n} luật theo Confidence')
        plt.show()

    def plot_rules_support_confidence_scatter(self, rules_df):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x="support", y="confidence", size="lift", hue="lift", 
                        data=rules_df, palette="viridis", sizes=(20, 200))
        plt.title('Support vs Confidence')
        plt.show()

    def plot_rules_support_confidence_scatter_interactive(self, rules_df):
        fig = px.scatter(rules_df, x="support", y="confidence", color="lift",
                         size="lift", hover_data=["antecedents_str", "consequents_str"],
                         title="Interactive Support vs Confidence")
        fig.show()

    def plot_rules_network(self, rules_df, max_rules=30):
        G = nx.DiGraph()
        plot_df = rules_df.head(max_rules)
        for i, row in plot_df.iterrows():
            ant = row['antecedents_str']
            con = row['consequents_str']
            G.add_edge(ant, con, weight=row['lift'])
        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(G, k=1)
        nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                node_size=1500, font_size=8, edge_color='gray', arrowsize=15)
        plt.title(f'Network Graph ({max_rules} luật đầu)')
        plt.show()