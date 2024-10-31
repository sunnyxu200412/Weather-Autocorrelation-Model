import matplotlib.pyplot as plt

# Data for each year in dictionary format
data = {
    2007: [[0.9393939393939394, 0.06060606060606061], [0.07407407407407407, 0.9259259259259259]],
    2008: [[0.9142857142857143, 0.08571428571428572], [0.12, 0.88]],
    2009: [[0.875, 0.125], [0.08333333333333333, 0.9166666666666666]],
    2010: [[0.9428571428571428, 0.05714285714285714], [0.08, 0.92]],
    2011: [[0.875, 0.125], [0.08333333333333333, 0.9166666666666666]],
    2012: [[0.90625, 0.09375], [0.10714285714285714, 0.8928571428571429]],
    2013: [[0.967741935483871, 0.03225806451612903], [0.034482758620689655, 0.9655172413793104]],
    2014: [[0.9090909090909091, 0.09090909090909091], [0.1111111111111111, 0.8888888888888888]],
    2015: [[0.9583333333333334, 0.041666666666666664], [0.027777777777777776, 0.9722222222222222]],
    2016: [[0.9285714285714286, 0.07142857142857142], [0.0625, 0.9375]],
    2017: [[0.9166666666666666, 0.08333333333333333], [0.125, 0.875]],
    2018: [[0.9375, 0.0625], [0.07142857142857142, 0.9285714285714286]],
    2019: [[0.9696969696969697, 0.030303030303030304], [0.037037037037037035, 0.9629629629629629]],
    2020: [[0.9714285714285714, 0.02857142857142857], [0.04, 0.96]],
    2021: [[0.9166666666666666, 0.08333333333333333], [0.05555555555555555, 0.9444444444444444]],
    2022: [[0.9666666666666667, 0.03333333333333333], [0.03333333333333333, 0.9666666666666667]]
}

# Separate each matrix position into individual lists for plotting
years = list(data.keys())
top_left = [data[year][0][0] for year in years]
top_right = [data[year][0][1] for year in years]
bottom_left = [data[year][1][0] for year in years]
bottom_right = [data[year][1][1] for year in years]

# Plotting each matrix entry over the years
plt.figure(figsize=(10, 6))
plt.plot(years, top_left, label="Top Left", marker="o")
plt.plot(years, top_right, label="Top Right", marker="o")
plt.plot(years, bottom_left, label="Bottom Left", marker="o")
plt.plot(years, bottom_right, label="Bottom Right", marker="o")

# Adding titles and labels
plt.title("Matrix Entries Over the Years")
plt.xlabel("Year")
plt.ylabel("Matrix Entry Value")
plt.legend()
plt.grid(True)
plt.show()
