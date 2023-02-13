"""
Takes a datasets (listed below) and returns term structure calculations
- historical VIX Futures contracts time series data
- historical VIX cash time series data
"""
# %%  # # #   # # # #   # # # #   # # # #   # # # #   # # # #   # # # #   # # # #   # # # #  # # # #
# # # # #   Imports, Load Data # # # #   # # # #   # # # #   # # # #   # # # #   # # # #   # # # #
# # # #   # # # #   # # # #   # # # #   # # # #   # # # #   # # # #   # # # #   # # # #   # # # #
# # #   # # # #   # # # #   # # # #   # # # #   # # # #   # # # #   # # # #   # # # #   # # # #
import pandas as pd
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import glob
import os



# read all futures contract csv files in data directory
path = r'.\data\contracts-interpolated'
all_files_gen = (f for f in glob.glob(os.path.join(path,"*.csv")))

vix_cash_filepath = 'excel_sheets/vix_cash_5m_interpolated.csv'


def csv_path_to_df(f):
    """Helper function to read csv files, format new columns, 
    resample if required, and return a dataframe"""
    
    df = pd.read_csv(f, index_col=0, parse_dates=True)
    # df = df.resample('30T').agg({'Close':'last', 'Open':'first', 'High':'max', 'Low':'min', 'Volume':'sum'})   ###### resample to x min timeframe
    df.dropna(inplace=True)
    df['Symbol'] = f[-9:-4] # e.g. 'VIF19'
    df['Symbol_Year'] = f[-6:-4] # e.g. '19'
    df['Symbol_Month'] = f[-7:-6] # e.g. 'F'
    df.reset_index(inplace=True)
    df.set_index(['Symbol', 'Date'], inplace=True)

    return df if len(df) > 0 else None

gen = (csv_path_to_df(f) for f in all_files_gen)
df = pd.concat(gen)

# drop NaN values
df = df[df['Close'].notna()]

# Load VIX cash data

vix_cash = pd.read_csv(vix_cash_filepath, index_col=0, parse_dates=True)
# vix_cash = vix_cash.resample('30T').agg({'Close':'last', 'Open':'first', 'High':'max', 'Low':'min'})
vix_cash['Symbol'] = "C"
vix_cash.reset_index(inplace=True)
vix_cash.set_index(['Symbol', 'Date'], inplace=True)
vix_cash['Volume'] = 0
vix_cash['Symbol_Year'] = vix_cash.index.get_level_values(1).year
vix_cash['Symbol_Year'] = vix_cash['Symbol_Year'].astype(str).str[-2:]
vix_cash['Symbol_Month'] = "C"  # C is for cash

# print(df.info())
# print(vix_cash.info())
# concat vix_cash to df
df = pd.concat([df, vix_cash])


# sort df by Symbol Year and Month
df.sort_values(by=['Date', 'Symbol_Year', 'Symbol_Month'], inplace=True)

# Create list of unique timestamps that have more than 4 contracts (inclyding VIX cash) present
df_grouped = df.groupby(level=1)
df_filtered = df_grouped.filter(lambda x: len(x) > 4)

# # for testing a date range..
# df_filtered = df_filtered[df_filtered.index.get_level_values(level=1) > '2022-04-02 20:55:00']
# df_filtered = df_filtered[df_filtered.index.get_level_values(level=1) < '2022-04-06 20:55:00']

time_stamps = list(df_filtered.index.get_level_values(1).unique())
time_stamps.sort(reverse=True)

print("timestamps to sort:", len(time_stamps))

# %%  # # #   # # # # #   # # # #   # # # #   # # # #   # # # #   # # # #   # # # #   # # # #
# # # # #   Calculate Term Structure  # # # #   # # # #   # # # #   # # # #   # # # #   # # # #
# # # #   # # # #   # # # #   # # # #   # # # #   # # # #   # # # #   # # # #   # # # #   # # # #
# # #   # # # #   # # # #   # # # #   # # # #   # # # #   # # # #   # # # #   # # # #   # # # #


# declare helper function the validate the contracts present and in correct order

def next_four_contracts(contract):
    """Helper function. Given a contract letter (the front month contract)
    return the next 3 contracts in order"""
    contract_month_order = ['F', 'G', 'H', 'J',
                            'K', 'M', 'N', 'Q', 
                            'U', 'V', 'X', 'Z']
    # Find the index of the given letter in the contract month order list
    index = contract_month_order.index(contract)
    # Create a new list with the next 3 letters from the contract month order list
    next_three_letters = [contract] + contract_month_order[index+1:index+3]

    # If the letter was "Z", we need to wrap around to the beginning of the list
    # to get the remaining letters
    if len(next_three_letters) < 3:
        next_three_letters += contract_month_order[:3-len(next_three_letters)]

    return next_three_letters


file_name = 'term_structure_5m.csv'

# df to store term structure data C (for cash), VX1, VX2, VX3, VX4 etc.
term_structure = pd.DataFrame()

for i, date in enumerate(time_stamps):
    df_date = df[df.index.get_level_values(level=1) == date]

    if i % 2000 == 0 and i != 0:
        print(i)

    # check there are atleast 4 contracts present (including VIX cash)
    if len(df_date) > 3:
        """Declare variables used to make sure that at least the 
        first 4 contracts (VX1, VX2, VX3, VX4) are in order."""
        correct_order = (next_four_contracts(df_date.Symbol_Month[1]))
        contract_order = (list(df_date.Symbol_Month[1:4]))

        # if contracts are in order, unstack and rename columns as per
        if contract_order == correct_order:
            unstack = df_date.Close.unstack(level=0)
            # rename the columns to VX1 for the 'front month', VX2, 3, 4... for the 'back months'
            unstack.columns = ['C'] + \
                [f'VX{c+1}' for c in range(len(unstack.columns)-1)]
            
            # concatentate the unstacked df to the term_structure df
            term_structure = pd.concat([term_structure, unstack])

    # periodically save progress to csv file
    if (len(term_structure) % 25000 == 0) and (len(term_structure) > 0):
        if os.path.isfile(file_name):
            os.rename(file_name, file_name[:-4] + f'_{i}.bak.csv')
            term_structure.to_csv(file_name)
        else:
            term_structure.to_csv(file_name)

    # # for debugging
    # if (len(term_structure) > 1000):
    #     print(f"index: {i}")
    #     print(f"final date: {date}")
    #     print(f"number skipped {x}")
    #     break

print(f"Term Structure rows created: {term_structure.shape}, from a total of timestamps sorted {len(time_stamps)}")

# rename the most recent csv backup
os.rename(file_name, file_name[:-4] + f'_{i}.bak.csv')

# save the final dataframe
term_structure.to_csv(file_name, index=True)