import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

# Read the YAML file and store it in a dictionary named config_data
with open('config.yml', 'r') as stream:
    try:
        config_data = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# Read the TSV file into a DataFrame
df_train = pd.read_csv(f'{config_data["train_path"]}', sep='\t', index_col=None)
df_val = pd.read_csv(f'{config_data["val_path"]}', sep='\t', index_col=None)
df_test = pd.read_csv(f'{config_data["test_path"]}', sep='\t', index_col=None)

def createDictOfLibs(df: pd.DataFrame) -> dict:
    """
    Create a dictionary of libraries and their counts from the given DataFrame.

    Parameters:
    df (DataFrame): The input DataFrame containing the 'libs' column.

    Returns:
    dict: A dictionary with the count of each library being imported.
    """
    list_of_libs = []
    for i in range(len(df)):
        # s: list of each .exe files being imported in 1 line 
        s = df['libs'][i].split(',')
        list_of_libs.append(s)

    dict_of_libs = {}
    # count the number of .exe files being imported
    for i in range(len(list_of_libs)):
        for j in range(len(list_of_libs[i])):
            if list_of_libs[i][j] in dict_of_libs:
                dict_of_libs[list_of_libs[i][j]] += 1
            else:
                dict_of_libs[list_of_libs[i][j]] = 1
       
    return dict_of_libs

def plotPieChart(data: dict, data_type: str = 'train') -> None:
    """
    Plot a pie chart based on the input data and its type.

    :param data: A dictionary containing the data to be plotted
    :type data: dict
    :param data_type: The type of the data (default is 'train')
    :type data_type: str
    :return: None
    :rtype: None
    """

    # Calculate the total sum of values
    total_sum = sum(data.values())
    threshold = int(0.009 * total_sum)
    # Create a new dictionary without elements less than n% of the total sum
    filtered_data_dict = {key: value for key, value in data.items() if value >= threshold}
    colors = plt.cm.viridis(np.linspace(0.3, 1, len(filtered_data_dict)))
    plt.figure(figsize=(8, 9))
    plt.pie(filtered_data_dict.values(), labels=filtered_data_dict.keys(),
            autopct='%1.1f%%', 
            labeldistance=1.1, 
            pctdistance=0.83,
            colors=colors)
    plt.tight_layout()
    plt.title(f'Pie Chart of Library Imports on {data_type} data, the most frequent\nlibraries are shown (those of > {threshold} imports; total {total_sum} imports)') 
    plt.savefig(f'pie_chart_{data_type}.png')
