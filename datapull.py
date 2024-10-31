import pandas as pd

def pull_data(yearStart, yearEnd):
    # Initialize an empty DataFrame to store the final result
    df_final = pd.DataFrame()

    for year in range(yearStart, yearEnd + 1):
        # Read the CSV for the current year
        print(f"Processing data for the year {year}")
        df = pd.read_csv(f'{year}.csv')

        # Convert 'PeriodStart' to datetime and extract month and day
        df['PeriodStart'] = pd.to_datetime(df['PeriodStart'])
        df['Month_Day'] = df['PeriodStart'].dt.strftime('%m-%d')  # Create a column for month and day

        # Group by 'Month_Day' and get the max value of 'DHI'
        max_dhi = df.groupby('Month_Day')['GHI'].max().reset_index()
        max_dhi = max_dhi.rename(columns={'GHI': str(year)})  # Rename 'DHI' column to the year
        # Merge into final DataFrame
        if df_final.empty:
            df_final = max_dhi
        else:
            df_final = pd.merge(df_final, max_dhi, on='Month_Day', how='outer')

    # Rename the first column to 'Date'
    df_final.rename(columns={'Month_Day': 'Date'}, inplace=True)

    # Save the final result to a CSV
    df_final.to_csv('max_dhi_by_month_day.csv', index=False)

    return df_final