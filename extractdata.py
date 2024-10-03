import pandas as pd
import numpy as np
import re

#pulling data from csv to calculate the average solar radiance per day cause right now they are recorded every hour and the time line is from october to november
#im lost from the model too but ig we can do that first today and ehhh try understanding teh model again on our own before wednesa?
#Sure, I see


#structure:
#3D arary [year][day][by hour]

df = pd.read_csv('filtered_data.csv')
#in 2D [year][data]
solardata = df[['DHI_2021', 'DHI_2022']].to_numpy()
#in 2D [year][data] Note: have not pulled from 2021
period_start = df['PeriodStart_2022'].astype(str)
period_start = period_start.str.extract(r'(\d{4}-\d{2}-\d{2})')

#print(period_start)
prev = period_start[0][0]
for date in period_start[0]:
    if(prev == date):
        finaldata = 



#print(period_start_2022.head())
# Models: ask advice from Asaf Cohen