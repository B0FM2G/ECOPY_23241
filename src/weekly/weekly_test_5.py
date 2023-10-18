def change_price_to_float(input_df):
    new_df=input_df.copy()
    new_df['item_price'] = new_df['item_price'].str.replace('$', '').astype(float)
    return new_df

def number_of_observations(input_df):
    new_df=input_df.copy()
    return len(new_df)

def items_and_prices(input_df):
    new_df=input_df.copy()
    return new_df.loc[:, ['item_name', 'item_price']]

def sorted_by_price(input_df):
    new_df=input_df.copy()
    return items_and_prices(new_df).sort_values('item_price', ascending=False)

def avg_price(input_df):
    new_df=input_df.copy()
    return new_df['item_price'].mean()

def unique_items_over_ten_dollars(input_df):
    new_df=input_df.copy()
    new_df = new_df[new_df['item_price'] > 10]
    new_df = new_df.drop_duplicates(subset=['item_name', 'choice_description', 'item_price'])
    return new_df[['item_name', 'choice_description', 'item_price']]

def items_starting_with_s(input_df):
    new_df=input_df.copy()
    new_df = new_df.drop_duplicates(subset=['item_name'])
    new_df = new_df['item_name'][new_df['item_name'].str.startswith('S')]
    return new_df

def first_three_columns(input_df):
    new_df = input_df.copy()
    return new_df.iloc[:, 0:3]

def every_column_except_last_two(input_df):
    new_df = input_df.copy()
    return new_df.iloc[:,:-2]

def sliced_view(input_df, columns_to_keep, column_to_filter, rows_to_keep):
    new_df=input_df.copy()
    return new_df.loc[[x in rows_to_keep for x in new_df[column_to_filter]], columns_to_keep]

def generate_quartile(input_df):
    new_df = input_df.copy()
    arak = list(new_df['item_price'])
    kvartilis = []
    for i in arak:
        if i > 29.99:
            kvartilis.append('premium')
        elif i > 19.99 and i < 30.00:
            kvartilis.append('high-cost')
        elif i > 9.99 and i < 20.00:
            kvartilis.append('medium-cost')
        else:
            kvartilis.append('low-cost')
    new_df['Quartile'] = kvartilis
    return new_df

def average_price_in_quartiles(input_df):
    new_df = input_df.copy()
    new_df=generate_quartile(new_df)
    return new_df.groupby('Quartile')['item_price'].mean()

def minmaxmean_price_in_quartile(input_df):
    new_df = input_df.copy()
    grouped = new_df.groupby('Quartile')
    minmaxmean_values = grouped['item_price'].agg(['min', 'max','mean'])
    return minmaxmean_values

import random
import src.weekly.weekly_test_5 as wt
from src.utils.distributions import UniformDistribution, LogisticDistribution, CauchyDistribution, \
    ChiSquaredDistribution
from src.weekly.weekly_test_2 import LaplaceDistribution

def gen_uniform_mean_trajectories(uniform_distribution, number_of_trajectories, length_of_trajectory):
    random.seed(42)
    trajectories = []

    for _ in range(number_of_trajectories):

        mean_trajectory = []
        trajectory = []
        for i in range(length_of_trajectory):
            trajectory.append(uniform_distribution.gen_rand())
            mean_trajectory.append(sum(trajectory) / (i + 1))

        trajectories.append(mean_trajectory)

    return trajectories

def gen_logistic_mean_trajectories(logistic_distribution, number_of_trajectories, length_of_trajectory):
    random.seed(42)
    trajectories = []

    for _ in range(number_of_trajectories):

        mean_trajectory = []
        trajectory = []
        for i in range(length_of_trajectory):
            trajectory.append(logistic_distribution.gen_rand())
            mean_trajectory.append(sum(trajectory) / (i + 1))

        trajectories.append(mean_trajectory)

    return trajectories

def gen_laplace_mean_trajectories(laplace_distribution, number_of_trajectories, length_of_trajectory):
    random.seed(42)
    trajectories = []

    for _ in range(number_of_trajectories):

        mean_trajectory = []
        trajectory = []
        for i in range(length_of_trajectory):
            trajectory.append(laplace_distribution.gen_rand())
            mean_trajectory.append(sum(trajectory) / (i + 1))

        trajectories.append(mean_trajectory)

    return trajectories

def gen_cauchy_mean_trajectories(cauchy_distribution, number_of_trajectories, length_of_trajectory):
    random.seed(42)
    trajectories = []

    for _ in range(number_of_trajectories):

        mean_trajectory = []
        trajectory = []
        for i in range(length_of_trajectory):
            trajectory.append(cauchy_distribution.gen_rand())
            mean_trajectory.append(sum(trajectory) / (i + 1))

        trajectories.append(mean_trajectory)

    return trajectories

def gen_chi2_mean_trajectories(chi2_distribution, number_of_trajectories, length_of_trajectory):
    random.seed(42)
    trajectories = []

    for _ in range(number_of_trajectories):

        mean_trajectory = []
        trajectory = []
        for i in range(length_of_trajectory):
            trajectory.append(chi2_distribution.gen_rand())
            mean_trajectory.append(sum(trajectory) / (i + 1))

        trajectories.append(mean_trajectory)

    return trajectories