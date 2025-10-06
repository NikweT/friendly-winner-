using System.Net;
using System.Reflection.Emit;
using static System.Runtime.InteropServices.JavaScript.JSType;

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.set_page_config(layout="wide")

st.title("South Africa Crime Statistics Analysis Dashboard")

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('/content/SouthAfricaCrimeStats_v2.csv.zip')
    return df

df = load_data()

st.header("Dataset Overview")
st.write("Here's a look at the first 5 rows of the dataset:")
st.dataframe(df.head())

st.write("Dataset Information:")
st.text(df.info())

st.write("Dataset Description:")
st.dataframe(df.describe())

st.write("Missing values per column:")
st.dataframe(df.isnull().sum())

# Data Transformation and Aggregation for Visualizations
df_long = df.melt(id_vars = ['Province', 'Station', 'Category'], var_name = 'Year', value_name = 'Crime_Count')
df_long['Year'] = df_long['Year'].str.split('-').str[0].astype(int)

df_agg = df_long.groupby(['Year', 'Category'])['Crime_Count'].sum().reset_index()
crime_category_totals = df_long.groupby('Category')['Crime_Count'].sum().reset_index()


st.header("Crime Trends Over Time")
st.write("Line graph showing crime count by category over time:")
plt.figure(figsize = (15, 8))
sns.lineplot(data = df_agg, x = 'Year', y = 'Crime_Count', hue = 'Category')
plt.title('Crime Count by Category Over Time')
plt.xlabel('Year')
plt.ylabel('Crime Count')
plt.legend(bbox_to_anchor = (1.05, 1), loc = 'upper left')
st.pyplot(plt)
plt.close() # Close the plot to free memory


st.header("Distribution of Crime Categories")
st.write("Pie chart showing the proportion of each crime category (Top 10 + Other):")

crime_category_totals_sorted = crime_category_totals.sort_values(by = 'Crime_Count', ascending = False)
top_n = 10
top_categories = crime_category_totals_sorted.head(top_n).copy()
other_crime_count = crime_category_totals_sorted.iloc[top_n:]['Crime_Count'].sum()
other_category_df = pd.DataFrame({ 'Category': ['Other'], 'Crime_Count': [other_crime_count]})
df_pie_chart = pd.concat([top_categories, other_category_df])
total_crime_count = df_pie_chart['Crime_Count'].sum()
df_pie_chart['Percentage'] = (df_pie_chart['Crime_Count'] / total_crime_count) * 100

plt.figure(figsize = (10, 10))
plt.pie(df_pie_chart['Crime_Count'], labels = df_pie_chart['Category'], autopct = '%1.1f%%', startangle = 140)
plt.title('Distribution of Crime Categories')
plt.axis('equal')
st.pyplot(plt)
plt.close() # Close the plot to free memory


st.header("Total Crime Count by Category")
st.write("Bar chart showing the total crime count for each category (Top 15):")

top_n_bar = 15
top_crime_categories_bar = crime_category_totals_sorted.head(top_n_bar).copy()

plt.figure(figsize = (12, 6))
sns.barplot(data = top_crime_categories_bar, x = 'Category', y = 'Crime_Count')
plt.title(f'Total Crime Count by Category (Top {top_n_bar})')
plt.xlabel('Crime Category')
plt.ylabel('Total Crime Count')
plt.xticks(rotation = 90)
plt.tight_layout()
st.pyplot(plt)
plt.close() # Close the plot to free memory

st.header("Station with the Most Crimes")
crime_by_station = df_long.groupby('Station')['Crime_Count'].sum().reset_index()
most_crime_station = crime_by_station.loc[crime_by_station['Crime_Count'].idxmax()]
st.write(f"The station with the most crimes is: **{most_crime_station['Station']}** with a total crime count of **{most_crime_station['Crime_Count']}**.")

st.header("Classification Model (Example)")
st.write("Below is an example of the classification model we built, demonstrating how it would predict a crime category based on yearly crime counts.")

# Note: Integrating the full classification model for interactive prediction in Streamlit
# would require loading the trained model and encoder, and creating input widgets
# for the user to enter yearly crime counts. This is a more complex integration
# and is not included in this initial dashboard structure but can be added
# if needed. The previous analysis showed the model's low accuracy with these features.



# To run this Streamlit app in Colab, you would typically save this code as a .py file
# (e.g., app.py) and then run it using:
# !streamlit run app.py & npx localtunnel --port 8501