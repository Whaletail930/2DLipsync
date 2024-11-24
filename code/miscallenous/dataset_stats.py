import os
import json
from collections import Counter
import matplotlib.pyplot as plt


def count_visemes_in_folder(folder_path):
    """
    Counts the occurrences of each viseme across all JSON files in the given folder.

    Parameters:
        folder_path (str): Path to the folder containing JSON files.

    Returns:
        Counter: A Counter object with viseme counts.
    """
    viseme_counts = Counter()

    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r") as file:
                try:
                    data = json.load(file)
                    for entry in data:
                        for mouthcue in entry.get("mouthcues", []):
                            viseme = mouthcue.get("mouthcue")
                            if viseme:
                                viseme_counts[viseme] += 1
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in file: {filename}")

    return viseme_counts


def analyze_viseme_percentage(folder_path, target_viseme='B'):
    """
    Analyzes the percentage of a specific viseme in each file and categorizes files into dynamic percentage ranges.

    Parameters:
        folder_path (str): Path to the folder containing JSON files.
        target_viseme (str): The viseme to calculate the percentage for.

    Returns:
        dict: A dictionary with category ranges and the number of files in each range.
    """
    percentage_ranges = {
        "0-25%": 0,
        "26-35%": 0,
        "36-50%": 0,
        "51-75%": 0,
        "76-100%": 0
    }

    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r") as file:
                try:
                    data = json.load(file)
                    total_visemes = 0
                    target_count = 0

                    for entry in data:
                        for mouthcue in entry.get("mouthcues", []):
                            viseme = mouthcue.get("mouthcue")
                            total_visemes += 1
                            if viseme == target_viseme:
                                target_count += 1

                    if total_visemes > 0:
                        percentage = (target_count / total_visemes) * 100
                        if 0 <= percentage <= 25:
                            percentage_ranges["0-25%"] += 1
                        elif 26 <= percentage <= 35:
                            percentage_ranges["26-35%"] += 1
                        elif 36 <= percentage <= 50:
                            percentage_ranges["36-50%"] += 1
                        elif 51 <= percentage <= 75:
                            percentage_ranges["51-75%"] += 1
                        elif 76 <= percentage <= 100:
                            percentage_ranges["76-100%"] += 1
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in file: {filename}")

    return percentage_ranges


def plot_viseme_counts(viseme_counts):
    """
    Plots the distribution of visemes in a bar chart.

    Parameters:
        viseme_counts (Counter): A Counter object with viseme counts.
    """
    if viseme_counts:
        visemes, counts = zip(*viseme_counts.items())

        plt.figure(figsize=(10, 6))
        plt.bar(visemes, counts, color='skyblue', alpha=0.8, edgecolor='black')
        plt.title("Distribution of Visemes in JSON Files")
        plt.xlabel("Visemes")
        plt.ylabel("Frequency")
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
    else:
        print("No viseme data to display.")


def plot_percentage_ranges(percentage_ranges):
    """
    Plots the distribution of files across percentage ranges in a bar chart.

    Parameters:
        percentage_ranges (dict): A dictionary with percentage ranges as keys and file counts as values.
    """
    ranges, counts = zip(*percentage_ranges.items())

    plt.figure(figsize=(10, 6))
    plt.bar(ranges, counts, color='salmon', alpha=0.8, edgecolor='black')
    plt.title("Distribution of Files by 'B' Viseme Percentage Ranges")
    plt.xlabel("Percentage Range of Target Viseme")
    plt.ylabel("Number of Files")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


folder_path = r"/OUTPUT/TRAIN"

# viseme_counts = count_visemes_in_folder(folder_path)
# plot_viseme_counts(viseme_counts)

# percentage_ranges = analyze_viseme_percentage(folder_path, target_viseme='B')
# plot_percentage_ranges(percentage_ranges)
