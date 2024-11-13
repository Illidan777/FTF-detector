def data_summary(df):
    """
    A simple exploratory data analysis (EDA) function to summarize a dataset.

    Parameters:
    df : pandas.DataFrame
        The input DataFrame to be analyzed.

    Prints:
    - Dimensions of the dataset (rows and columns)
    - Total missing values (NA's)
    - Information about duplicates (if any)
    - Data type, distinct count, and missing values for each column
    """
    total_na = df.isna().sum().sum()
    col_name = df.columns
    dtypes = df.dtypes
    uniq = df.nunique()
    na_val = df.isna().sum()
    duplicate_indices = df[df.duplicated()].index

    print(f'Dimensions: {df.shape[0]} rows, {df.shape[1]} columns')
    print(f'Total NA\'s: {total_na}')

    if len(duplicate_indices) > 0:
        print(f'Duplicate rows indices: {duplicate_indices}')
    else:
        print('There are no duplicates in this dataset!')

    print(f"{'Column Name':<38} {'Data Type':<10} {'Count Distinct':<15} {'NA Values':<10}")

    for i in range(len(df.columns)):
        dtype = str(dtypes.iloc[i])
        distinct_count = uniq.iloc[i]
        na_count = na_val.iloc[i]

        # Handle object types for proper formatting
        if dtype == 'object':
            print(f"{col_name[i]:<38} {dtype:<10} {distinct_count:<15} {na_count:<10}")
        else:
            print(f"{col_name[i]:<38} {dtype:<10} {distinct_count:<15} {na_count:<10}")

def calculate_percentage(part, total):
    if total == 0:
        return 0  # to avoid division by zero
    percentage = (part / total) * 100
    print(f"Number {part} is {percentage}% of {total}")

