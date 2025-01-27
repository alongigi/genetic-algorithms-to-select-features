import re
import pandas as pd
from matplotlib import pyplot as plt


def main(file_name):
    # Read the file content
    with open(f"job-{file_name}.out", 'r') as file:
        text = file.read()

    # Regular expressions to extract the required data
    features_to_select = [int(float(s.group(1))) for s in re.finditer(r'Features to select: (\d+)', text)]
    best_fitness_other = [round(float(s.group(1)), 2) for s in re.finditer(r'Fitness value of the best solution = ([\d\.]+)', text)]
    training_time_other = [int(float(s.group(1))) for s in re.finditer(r'Training time: ([\d\.]+)', text)]
    # best_fitness_genetic = re.search(r'Fitness value of the best solution = ([\d\.]+)', text.split('simple genetic algorithm')[1])

    # Extracting chi2, mutual_info_classif, mutual_info_reg, f_classif, f_reg accuracies
    chi2 = [round(float(s.group(1)), 2) for s in re.finditer(r'Feature selection method: classifier_chi2\. Accuracy: ([\d\.]+)', text)]
    mutual_info_classif = [round(float(s.group(1)), 2) for s in re.finditer(r'Feature selection method: classifier_mutual_info_classif\. Accuracy: ([\d\.]+)', text)]
    mutual_info_reg = [round(float(s.group(1)), 2) for s in re.finditer(r'Feature selection method: classifier_mutual_info_regression\. Accuracy: ([\d\.]+)', text)]
    f_classif = [round(float(s.group(1)), 2) for s in re.finditer(r'Feature selection method: classifier_f_classif\. Accuracy: ([\d\.]+)', text)]
    f_reg = [round(float(s.group(1)), 2) for s in re.finditer(r'Feature selection method: classifier_f_regression\. Accuracy: ([\d\.]+)', text)]

    min_len = int(min(len(features_to_select), len(best_fitness_other)/2,
                  len(training_time_other)/2, len(chi2), len(mutual_info_classif),
                  len(mutual_info_reg), len(f_classif), len(f_reg)))
    # Build data dictionary for the DataFrame
    data = {
        'num features to select': features_to_select[:min_len],
        'time': training_time_other[:min_len*2:2],
        'best fitness': best_fitness_other[:min_len*2:2],
        'ev-time': training_time_other[1:min_len*2:2],
        'ev-fitness': best_fitness_other[1:min_len*2:2],
        'chi2': chi2[:min_len],
        'mutual info classif': mutual_info_classif[:min_len],
        'mutual info reg': mutual_info_reg[:min_len],
        'f classif': f_classif[:min_len],
        'f reg': f_reg[:min_len]
    }

    # Create the DataFrame
    df = pd.DataFrame(data)
    df.to_excel(f'job-{file_name}.xlsx', index=False)
    # Show the DataFrame
    # print(df)

def graphs(file_name):
    with open(f"outputs_isolet/job-{file_name}.out", 'r') as file:
        text = file.read()

    # Adjusted approach to extract floats directly from square brackets
    float64_pattern = r"\[([^\]]+)\]"  # Match contents within square brackets

    # Extract matches
    matches = re.findall(float64_pattern, text)

    # Filter matches to those containing "np.float64"
    filtered_matches = [match for match in matches if "np.float64" in match]

    # Process each match: remove "np.float64(" and ")" and convert to float lists
    list_of_lists = []
    for match in filtered_matches:
        cleaned = re.sub(r"np\.float64\(|\)", "", match)  # Remove np.float64 syntax
        float_list = list(map(float, cleaned.split(", ")))  # Convert to list of floats
        list_of_lists.append(float_list)

    our_method = list_of_lists[::2][:8]
    ga = list_of_lists[1::2][:8]
    # Plot each list as a separate line
    for idx, values in enumerate(ga):
        plt.plot(values, label=f'{(idx + 1)*5} features')

    # Add labels and legend
    plt.xlabel('Generation')
    plt.ylabel('Fitness Value')
    plt.title('Fitness Value Over Generations')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    # Show the plot
    plt.savefig(f"outputs_isolet/sss_graphs/general_ga/{file_name}.png")
    plt.show()

if __name__ == '__main__':
    # File path
    file_names = list(range(2463382, 2463396, 2))
    for file_name in file_names:
    #     main(file_name)
        graphs(file_name)