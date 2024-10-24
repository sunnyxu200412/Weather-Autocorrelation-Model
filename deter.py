# import pandas as pd
# import numpy as np
# import math

# # i by t 
# # i = 2, t = 61
# data = []
# mean_array = []

# # def create():
# #     df = pd.read_csv('max_dhi_by_month_day.csv')
# #     data_temp = []
# #     data_temp.append(df['2021'])
# #     data_temp.append(df['2022'])
# #     data = data_temp
# #     # for i in range(1, 3):
# #     #     temp = []
# #     #     for k in range(1, 62):
# #     #         temp.append(k)
# #     #     data.append(temp)

# def create():
#     # Read the CSV file
#     df = pd.read_csv('max_dhi_by_month_day.csv')

#     # Check if the DataFrame is empty
#     if df.empty:
#         print("DataFrame is empty. Please check the CSV file.")
#         return []

#     # Initialize a list to hold the yearly data
#     data_temp = []
    
#     # Append the data for 2021 and 2022
#     data_temp.append(df['2021'].tolist())  # Convert to list
#     data_temp.append(df['2022'].tolist())  # Convert to list
    
#     return data_temp

# def mean(data):
#     n = len(data)        # Number of rows
#     t = len(data[0])
#     for i in range(t):
#         sum = 0
#         for k in range(n):
#             sum += data[k][i]
#         mean_array.append(sum/n)


# def whole_mean():
#     w = 61
#     temp = 0
#     for i in range(w):
#         temp += mean_array[i]
#     return temp / 61

# def deviation_func(mean):
#     w = 61
#     sum = 0
#     for i in range(w):
#         sum += pow(mean_array[i] - mean, 2)
#     sum /= (w - 1)
#     return sum

# def predicted(t):
#     w = 61
#     mean = whole_mean()
#     deviation = deviation_func(mean)
#     print(deviation)
#     f_i = 1/61
#     a_1 = 2/w * deviation * math.cos(2*math.pi*t/61)
#     b_1 = 2/w * deviation * math.sin(2*math.pi*t/61)
#     return mean + a_1 * math.cos(2 * math.pi * t / w) + b_1 * math.sin(2 * math.pi * t / w)

# data = create()
# mean(data)
# print(mean_array)
# predicted(3)


import numpy as np
import matplotlib.pyplot as plt
import math

# Your data stored in a variable called 'data'
data = [
    [527, 527, 408, 486, 560, 575, 530, 512, 535, 556, 578, 545, 560, 558, 553, 547, 567, 575, 578, 579, 516, 575, 536, 540, 581, 541, 578, 579, 592, 577, 566, 571, 581, 576, 584, 582, 598, 588, 594, 557, 595, 607, 603, 571, 597, 605, 552, 561, 598, 582, 589, 590, 616, 609, 605, 608, 610, 591, 605, 511, 590],
    [555, 559, 538, 545, 505, 550, 538, 559, 550, 560, 561, 569, 555, 578, 556, 554, 561, 590, 592, 542, 371, 551, 592, 565, 577, 578, 557, 580, 571, 577, 579, 563, 590, 581, 574, 534, 495, 593, 588, 559, 563, 580, 536, 587, 595, 589, 593, 594, 578, 584, 588, 596, 604, 581, 586, 588, 589, 586, 612, 597, 376]
]

# Assuming you want to work with the first series in 'data'
values = data[0]

# Create a time array (assuming uniform spacing for simplicity)
time = np.arange(len(values))  # 0, 1, 2, ..., len(values)-1

# Define the Fourier Series approximation function
def fourier_series_approximation(t, values, n_terms):
    mean = np.mean(values)  # Calculate the mean from the data
    result = mean  # Start with the mean

    # Calculate the period based on the time data
    w = np.max(t) - np.min(t)

    for n in range(1, n_terms + 1):
        # Calculate Fourier coefficients
        a_n = (2 / w) * np.trapz(values * np.cos(2 * np.pi * n * t / w), t)  # Coefficient for cosine
        b_n = (2 / w) * np.trapz(values * np.sin(2 * np.pi * n * t / w), t)  # Coefficient for sine
        
        # Add terms to the result using the specified format
        result += a_n * np.cos(2 * np.pi * n * t / w) + b_n * np.sin(2 * np.pi * n * t / w)
    
    return result

def residual_time_series(deter_predict):
    mean = deter_predict.mean()
    # Step 2: Subtract the mean from each data point and square the result
    squared_diffs = [(deter - mean) ** 2 for deter in deter_predict]

    # Step 3: Sum the squared differences
    sum_squared_diffs = sum(squared_diffs)

    # Step 4: Divide by n-1 (Bessel's correction)
    n = len(deter_predict)
    variance = sum_squared_diffs / (n - 1)

    # Step 5: Take the square root to get the standard deviation
    std_dev = math.sqrt(variance)

    residual = []
    for deter in deter_predict:
        residual.append((deter-mean)/std_dev)
        
    return residual

def markov_matrix(residual):
    n = len(residual)
    pos_to_neg = 0
    pos_to_pos = 0
    neg_to_neg = 0
    neg_to_pos = 0
    
    for i in range(1, n):  # Iterating up to n-1, as residual[i] refers to index i
        if residual[i] >= 0 and residual[i-1] >= 0:
            pos_to_pos += 1
        elif residual[i] >= 0 and residual[i-1] < 0:
            neg_to_pos += 1
        elif residual[i] < 0 and residual[i-1] >= 0:
            pos_to_neg += 1
        else: 
            neg_to_neg += 1
    matrix = [
    [pos_to_pos/(pos_to_neg+pos_to_pos), pos_to_neg/(pos_to_neg+pos_to_pos)],
    [neg_to_pos/(neg_to_neg+neg_to_pos), neg_to_neg/(neg_to_neg+neg_to_pos)]
    ]
    return matrix

# Parameters
n_terms = 10  # Number of terms in the Fourier series

# Calculate Fourier series approximation
approximation = fourier_series_approximation(time, values, n_terms)
residual = residual_time_series(approximation)
matrix = markov_matrix(residual)
print(matrix)

# # Plotting the original data and the approximation
# plt.figure(figsize=(12, 6))
# plt.plot(time, values, label='Original Data', color='orange', marker='o')
# plt.plot(time, approximation, label=f'Fourier Series Approximation (n={n_terms})', color='blue')
# plt.title('Fourier Series Approximation of Your Data')
# plt.xlabel('Time')
# plt.ylabel('Value')
# plt.legend()
# plt.grid()
# plt.show()
