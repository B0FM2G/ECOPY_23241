import pandas as pd
df_data = pd.read_csv('../data/Euro_2012_stats_TEAM.csv')
df_data

new_df = df_data.copy()
def number_of_participants(input_df):
    csapat = input_df['Team'].count()
    return csapat

new_df = df_data.copy()
def goals(input_df):
    return input_df[['Team', 'Goals']]

new_df = df_data.copy()
def sorted_by_goal(input_df):
    return goals(input_df).sort_values("Goals", ascending=False)

new_df = df_data.copy()
def avg_goal(input_df):
    return input_df['Goals'].mean()

new_df = df_data.copy()
def countries_over_five(input_df):
    return input_df['Team'][input_df['Goals']>=6]

new_df = df_data.copy()
def countries_starting_with_g(input_df):
    return input_df['Team'][input_df['Team'].str.startswith('G')]

new_df = df_data.copy()
def first_seven_columns(input_df):
    return input_df.iloc[:,0:7]

new_df = df_data.copy()
def every_column_except_last_three(input_df):
    return input_df.iloc[:,:-3]

new_df = df_data.copy()
def sliced_view(input_df, column_to_keep, column_to_filter,rows_to_keep):
    selected_columns = input_df[columns_to_keep]
    filtered_rows = input_df[input_df[column_to_filter].isin(rows_to_keep)]
    return filtered_rows[selected_columns]

new_df = df_data.copy()
def generate_quartile(input_df):
    input_df['Quartile'] = pd.cut(input_df['Goals'], [0, 2, 4, 5, 12], labels=[4, 3, 2, 1])
    return input_df

def average_yellow_in_quartiles(input_df):
    return input_df.pivot_table(values="Passes", index="Quartile")

def minmax_block_in_quartile(input_df):
    grouped = input_df.groupby('Quartile')
    minmax_values = grouped['Blocks'].agg(['min', 'max']).reset_index()
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
random.seed(42)
pareto = ParetoDistribution(random, 1, 1)
def generate_mean_trajectories(pareto_distribution, number_of_trajectories, length_of_trajectory):
    result = []
    for _ in range(number_of_trajectories):
        trajectory = []
        cumulative_sum = 0
        for _ in range(length_of_trajectory):
            random_number = pareto_distribution.gen_rand()
            cumulative_sum += random_number
            trajectory.append(cumulative_sum / (len(trajectory) + 1))
        result.append(trajectory)
    return result
