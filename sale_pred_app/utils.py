import pandas as pd
import numpy as np
import editdistance
import os

from sklearn.tree import DecisionTreeRegressor
from math import ceil, floor, isnan

from sklearn.preprocessing import LabelEncoder

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
APP_DATA = os.path.join(APP_ROOT, 'data')

def load_data(years):
    #print("Beginning load_data()")

    sales_by_year = {}
    
    try:
        for year in years:
            sales_by_year[year] = pd.read_csv(os.path.join(APP_DATA, f"{year}.csv"))
    except (FileNotFoundError):
        print(f"/data/{year}.csv not found")

    #print("Completed data collection")
    
    sales = pd.concat(list(sales_by_year.values()))

    categories = ["Achronim", "Biography", "Chassidus", "Children's", "Cookbooks", "English Halacha", "English Mussar & Machshava", "English Tanach", "Gemara", "Haggada", "Hebrew Halacha", "Hebrew Mussar & Machshava", "Hebrew Tanach", "History", "Judaica", "Midrashim", "Mishna", "Music", "Novels", "Reference", "Rishonim", "Scholarly Works", "Shailot u'Teshuvot", "Siddurim", "Tefillah", "Yeshiva University", "Featured", "Set Land", "Misc"]

    def transform_category(category):
        if not isinstance(category, str):
            return "Misc"
        else:
            if category == "YU": category = "Yeshiva University"
            return min(categories, key = lambda s:editdistance.eval("".join(filter(str.isalnum, category)).lower(), "".join(filter(str.isalnum, s)).lower()))

    def correct_quant(x):
        if isnan(x) or x < 0:
            return 0
        else:
            return x

    titles = []

    for year in sales_by_year.keys():
        sales_by_year[year] = sales_by_year[year].drop(columns=["gross_sales", "discounts",	"returns", "net_sales", "taxes", "total_sales"])
        sales_by_year[year]['product_title'].apply(lambda x: titles.append(x))
        sales_by_year[year]['product_type'] = sales_by_year[year]['product_type'].apply(lambda x: transform_category(x))
        sales_by_year[year]['net_quantity'] = sales_by_year[year]['net_quantity'].apply(lambda x: correct_quant(x))
        sales_by_year[year]['year'] = year

    titles = set(titles)

    title_sales_df = pd.DataFrame(index=titles, columns=sales_by_year.keys())
    
    title_sales_df = title_sales_df.fillna(0)
    
    sales = pd.concat(list(sales_by_year.values()))

    for idx, sale in sales.iterrows():
        title = sale['product_title']
        year = sale['year']
        quant = sale['net_quantity']

        title_sales_df.at[title, year] += quant

    cat_enc = LabelEncoder()
    cat_enc.fit(categories)

    # get all sales in one df again, continuous index
    sales = pd.concat(list(sales_by_year.values()), ignore_index=True)

    def encode_category(x):
      # find all rows which match the title in question
      title_matches = sales[sales['product_title'] == x.name]
    
      # find the row with the most recent year
      latest_yr_idx = title_matches['year'].idxmax()
    
      # return encoded category
      return cat_enc.transform([sales.loc[latest_yr_idx, 'product_type']])[0]

    title_sales_df['category'] = title_sales_df.apply(lambda x: encode_category(x), axis=1)

    #print("Completed data cleaning")
    return title_sales_df

def predict(train_years):
    #print("Beginning predict()")
    if len(train_years) < 4:
        raise Exception(f"Must have 4 or more years' worth of data, only found {len(train_years)}") 
    
    df = load_data(train_years)
    #print("Completed load_data()")


    med_data = df[train_years[-4:]]

    time_series = []

    years = train_years[2:]

    for year in years:
      row = []
    
      # inference years = two years prior to regression target year
      inf_year_1 = year - 2
      inf_year_2 = year - 1

      # append year label for first inference year
      row.append(inf_year_1)
      # append product quantities from first inference year
      row += list(df[inf_year_1].to_numpy())

      # append year label for second inference year
      row.append(inf_year_2)
      # append product quantities from second inference year
      row += list(df[inf_year_2].to_numpy())

      # append encoded category labels for all products
      row += list(df['category'].to_numpy())

      # append product quantities for regression target years
      row += list(df[year].to_numpy())

      time_series.append(row)

    dt_data = pd.DataFrame(time_series)

    X = dt_data.iloc[:, :-len(df)]
    y = dt_data.iloc[:, -len(df):]

    #print("Completed data setup")

    dt_model = DecisionTreeRegressor(splitter='best', max_depth=1, min_samples_leaf=1, min_weight_fraction_leaf=0.4, max_features='auto', max_leaf_nodes=None)
    dt_model.fit(X, y)

    #print("Completed DT training")

    X_pred = []

    X_pred.append(train_years[-2])
    X_pred += list(df[train_years[-2]].to_numpy())
    X_pred.append(train_years[-1])
    X_pred += list(df[train_years[-1]].to_numpy())
    X_pred += list(df['category'].to_numpy())
    X_pred = np.array(X_pred)

    results = pd.DataFrame(index=df.index.values)

    results['four_yr_med'] = med_data.median(axis=1)
    results['dt_reg'] = dt_model.predict(X_pred.reshape(1, -1)).T

    #print("Collected raw results")

    def set_lower_bound(x1, x2):
        x = min(x1, x2)
        if x < 0: return 0
        else: return floor(x)
    
    results['upper_bound'] = results.apply(lambda x: ceil(max(x['four_yr_med'], x['dt_reg'])), axis=1)
    results['lower_bound'] = results.apply(lambda x: set_lower_bound(x['four_yr_med'], x['dt_reg']), axis=1)

    #print('Created final results')

    return results[['lower_bound', 'upper_bound']]