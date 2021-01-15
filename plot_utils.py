import json
import matplotlib.pyplot as plt

def plot_accuracies(json_file, name):
    """Plot accuracies from a json file containing {epoch # : accuracy pairs}"""
    with open(json_file) as file: 
        data = json.load(file) 
    
    accuracies = []
    for key, value in data.items():
        accuracies.append(value)

    plt.figure(figsize=(12,10))
    plt.plot(accuracies, marker='o', markersize=3, markeredgecolor='red')
    plt.title(name)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.savefig('plots/{}.png'.format(name))
    plt.show()