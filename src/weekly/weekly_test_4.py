import pandas as pd
from pytest import approx
import random
from src.weekly.weekly_test_2 import ParetoDistribution

euro12 = pd.read_csv('../../data/Euro_2012_stats_TEAM.csv')
new_df = euro12.copy()

def number_of_participants(input_df):
    csapat = input_df['Team'].count()
    return csapat

def goals(input_df):
    return input_df[['Team', 'Goals']]

def sorted_by_goal(input_df):
    return goals(input_df).sort_values("Goals", ascending=False)

def avg_goal(input_df):
    return input_df['Goals'].mean()

def countries_over_five(input_df):
    return pd.DataFrame(input_df['Team'][input_df['Goals']>=6])

def countries_starting_with_g(input_df):
    return input_df['Team'][input_df['Team'].str.startswith('G')]

def first_seven_columns(input_df):
    return input_df.iloc[:,0:7]

def every_column_except_last_three(input_df):
    return input_df.iloc[:,:-3]

def sliced_view(input_df, columns_to_keep, column_to_filter, rows_to_keep):
    return input_df.loc[[x in rows_to_keep for x in input_df[column_to_filter]], columns_to_keep]

def generate_quarters(input_df):
    golok = list(input_df['Goals'])
    kvartilis = []
    for i in golok:
        if i > 5:
            kvartilis.append(1)
        elif i == 5:
            kvartilis.append(2)
        elif i > 2 and i < 5:
            kvartilis.append(3)
        else:
            kvartilis.append(4)
    input_df['Quartile'] = kvartilis
    return input_df

def average_yellow_in_quartiles(input_df):
    new_df2 = input_df.copy()
    return new_df2.groupby('Quartile')['Passes'].mean()

def minmax_block_in_quartile(input_df):
    grouped = input_df.groupby('Quartile')
    minmax_values = grouped['Blocks'].agg(['min', 'max'])
    return minmax_values

import matplotlib.pyplot as plt
def scatter_goals_shots(input_df):
    fig, ax = plt.subplots()
    plt.scatter(input_df['Goals'], input_df['Shots on target'])
    ax.set_title('Goals and Shot on target')
    ax.set_xlabel('Goals')
    ax.set_ylabel('Shots on target')
    return fig

def scatter_goals_shots_by_quartile(input_df):
    fig, ax = plt.subplots()
    plt.scatter(input_df['Goals'], input_df['Shots on target'], c=input_df['Quartile'])
    ax.set_title('Goals and Shot on target')
    ax.set_xlabel('Goals')
    ax.set_ylabel('Shots on target')
    plt.legend('Quartiles')
    return fig

import random

def gen_pareto_mean_trajectories(pareto_distribution, number_of_trajectories, length_of_trajectory):
    pareto_distribution.rand.seed(42)
    trajectories = []

    for _ in range(number_of_trajectories):

        mean_trajectory = []
        trajectory = []
        for i in range(length_of_trajectory):
            trajectory.append(pareto_distribution.gen_rand())
            mean_trajectory.append(sum(trajectory) / (i + 1))

        trajectories.append(mean_trajectory)

    return trajectories