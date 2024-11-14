from typing import Any, Dict, Tuple

import pandas as pd
import streamlit as st
from pages.ml_reference.lib.exposition.export import display_config_download_links
from pages.ml_reference.lib.utils.load import download_toy_dataset, load_custom_config, load_dataset, fetch_dataset
from itertools import product


def input_dataset(
    config: Dict[Any, Any], readme: Dict[Any, Any], instructions: Dict[Any, Any], custom_query: str = ""
) -> Tuple[pd.DataFrame, Dict[Any, Any], Dict[Any, Any], Dict[Any, Any]]:
    """Lets the user decide whether to use a default dataset, enter a custom query, or upload a CSV file.

    Parameters
    ----------
    config : Dict
        Lib config dictionary containing information about toy datasets (download links).
    readme : Dict
        Dictionary containing tooltips to guide user's choices.
    instructions : Dict
        Dictionary containing instructions to provide a custom config.
    custom_query : str, optional
        Custom SQL query to fetch dataset, by default ""

    Returns
    -------
    pd.DataFrame
        Selected dataset loaded into a dataframe.
    dict
        Loading options selected by user (upload or download, dataset name if download).
    dict
        Lib configuration dictionary.
    dict
        Dictionary containing all datasets.
    """
    load_options, datasets = dict(), dict()
    load_options["adnex_dataset"] = st.checkbox(
        "Load a ADNEX dataset", True, help=readme["tooltips"]["upload_choice"]
    )
    if load_options["adnex_dataset"]:
        dataset_name = st.selectbox(
            "Select a dataset",
            options=list(config["datasets"].keys()),
            format_func=lambda x: config["datasets"][x]["name"],
            help=readme["tooltips"]["adnex_data_description"],
        )
        df = fetch_dataset(config["datasets"][dataset_name]["sql"])

        # df.set_index('date', inplace=True)
        # df.index = pd.to_datetime(df.index)

        # Add the regressor for online_revenue
        if dataset_name == 'Online_Revenue':
            # New data to add
            data = {
                'date': ['11/8/2021', '11/9/2021', '11/10/2021', '11/11/2021', '11/12/2021', 
                        '11/9/2022', '11/10/2022', '11/11/2022', '11/9/2023', '11/10/2023', '11/11/2023'],
                'peak_promotion': ['FALSE', 'FALSE', 'TRUE', 'FALSE', 'FALSE', 'TRUE', 'FALSE', 'TRUE', 'TRUE', 'FALSE', 'TRUE'],
                'big_promotion': ['FALSE', 'FALSE', 'FALSE', 'TRUE', 'FALSE', 'FALSE', 'FALSE', 'TRUE', 'FALSE', 'FALSE', 'TRUE'],
                'range_promotion': ['TRUE', 'TRUE', 'TRUE', 'TRUE', 'TRUE', 'TRUE', 'TRUE', 'TRUE', 'TRUE', 'TRUE', 'TRUE']
            }

           # Initialize the columns in df with 'FALSE'
            df['peak_promotion'] = 'FALSE'
            df['big_promotion'] = 'FALSE'
            df['range_promotion'] = 'FALSE'

            # Convert 'date' columns to datetime
            data_df = pd.DataFrame(data)
            data_df['date'] = pd.to_datetime(data_df['date'])
            df['date'] = pd.to_datetime(df['date'])

            # Merge the DataFrames on 'date' using a left join
            merged_df = pd.merge(df, data_df, on='date', how='left', suffixes=('', '_new'))

            # Update the promotion columns with values from data_df where they exist
            merged_df['peak_promotion'] = merged_df['peak_promotion_new'].combine_first(merged_df['peak_promotion'])
            merged_df['big_promotion'] = merged_df['big_promotion_new'].combine_first(merged_df['big_promotion'])
            merged_df['range_promotion'] = merged_df['range_promotion_new'].combine_first(merged_df['range_promotion'])

            # Drop the temporary columns
            merged_df.drop(columns=['peak_promotion_new', 'big_promotion_new', 'range_promotion_new'], inplace=True)

            df = merged_df.copy()
            # print(merged_df)
        
        # Fixed issue with days without revenue
        if dataset_name == 'Offline_Revenue':
            ##### Convert 'date' to datetime if not already
            df['date'] = pd.to_datetime(df['date'])

            # Function to fill missing dates for each store
            def fill_missing_dates(store_df):
                min_date = store_df['date'].min()
                max_date = store_df['date'].max()
                all_dates = pd.date_range(start=min_date, end=max_date, freq='D')

                complete_df = pd.DataFrame({
                    'date': all_dates,
                    'store_name': store_df['store_name'].iloc[0]
                })

                merged_df = pd.merge(complete_df, store_df, on=['date', 'store_name'], how='left')
                merged_df['total_revenue'] = merged_df['total_revenue'].fillna(0)

                return merged_df

            # Apply the function to each store
            filled_data = df.groupby('store_name').apply(fill_missing_dates).reset_index(drop=True)

            # Ensure sorting is correct
            # filled_data = filled_data.sort_values(by=['store_name', 'date'])
            df = filled_data.copy()

            ##### Add Promotion Data
            # Load the promotions data from CSV
            promotions_df = pd.read_csv('/Users/kdanmobile/gitlab/adnex-infra/streamlit/pages/ml_reference/references/all_promotions.csv')


            # Convert date columns in both DataFrames to datetime to ensure proper merging
            promotions_df['date'] = pd.to_datetime(promotions_df['date'])

            # Assuming your existing DataFrame is named df and has similar 'date' and 'store_name' columns
            df['date'] = pd.to_datetime(df['date'])

            # No need to replace 'TRUE' with True, since they are already True or NaN in CSV
            # Directly handle the promotion columns during merging
            promotions_columns = ['big_promotion', 'small_promotion', 'propose_promotion', 
                                'sales_promotion', 'anniversary_promotion', 'first_day_anniversary_promotion', 'first_segment_anniversary'
                                ,'middle_segment_anniversary', 'last_segment_anniversary']

            # Merge the DataFrames on 'store_name' and 'date'
            merged_df = pd.merge(df, promotions_df, on=['store_name', 'date'], how='left')

            # Fill NaN values with False for cases where there's no promotion data available
            merged_df.fillna({col: False for col in promotions_columns}, inplace=True)
            merged_df = merged_df.infer_objects(copy=False)

            df = merged_df.copy() 


        if dataset_name == 'Item_Sale_Quantity':
            # Extract all unique values for date, store_name, and item_name
            unique_dates = df['date'].unique()
            unique_stores = df['store_name'].unique()
            unique_items = df['item_name'].unique()

            # Create all combinations of date, store_name, and item_name
            all_combinations = pd.DataFrame(product(unique_dates, unique_stores, unique_items), 
                                            columns=['date', 'store_name', 'item_name'])

            # Merge with the original dataframe to ensure all combinations are present
            df_complete = pd.merge(all_combinations, df, on=['date', 'store_name', 'item_name'], how='left')

            # Fill NaN values in total_quantity with 0
            df_complete['total_quantity'].fillna(0, inplace=True)
            df_complete = df_complete.infer_objects(copy=False)

            # Creating a mapping of item_name to item_category
            item_category_mapping = df[['item_name', 'item_category']].drop_duplicates().set_index('item_name')['item_category']

            # Filling NaN values in item_category based on the mapping
            df_complete['item_category'] = df_complete['item_name'].map(item_category_mapping)

            # Load the promotions data
            promotions_df = pd.read_csv('/Users/kdanmobile/gitlab/adnex-infra/streamlit/pages/ml_reference/references/all_promotions.csv')

            # Convert the 'date' column to datetime format
            df_complete['date'] = pd.to_datetime(df_complete['date'])
            promotions_df['date'] = pd.to_datetime(promotions_df['date'])

            # Fill NaN values with False and convert to boolean, then use infer_objects
            promotion_columns = ['big_promotion', 'small_promotion', 'propose_promotion', 
                                'sales_promotion', 'anniversary_promotion', 'first_day_anniversary_promotion', 'first_segment_anniversary'
                                ,'middle_segment_anniversary', 'last_segment_anniversary']

            promotions_df[promotion_columns] = promotions_df[promotion_columns].fillna(False)
            promotions_df = promotions_df.infer_objects(copy=False)

            # Merge the promotions data with the existing sales data
            df_merged = pd.merge(df_complete, promotions_df, on=['date', 'store_name'], how='left')

            # Fill NaN values in the promotion columns with False and use infer_objects
            df_merged[promotion_columns] = df_merged[promotion_columns].fillna(False)
            df_merged = df_merged.infer_objects(copy=False)

            # Your final dataframe
            df = df_merged.copy()

            #print(df)


        load_options["dataset"] = dataset_name
        load_options["date_format"] = config["dataprep"]["date_format"]
        load_options["separator"] = ","
    else:
        file = st.file_uploader(
            "Upload a csv file", type="csv", help=readme["tooltips"]["dataset_upload"]
        )
        load_options["separator"] = st.selectbox(
            "What is the separator?", [",", ";", "|"], help=readme["tooltips"]["separator"]
        )
        load_options["date_format"] = st.text_input(
            "What is the date format?",
            config["dataprep"]["date_format"],
            help=readme["tooltips"]["date_format"],
        )
        if st.checkbox(
            "Upload my own config file", False, help=readme["tooltips"]["custom_config_choice"]
        ):
            with st.sidebar.expander("Configuration", expanded=True):
                display_config_download_links(
                    config,
                    "config.toml",
                    "Template",
                    instructions,
                    "instructions.toml",
                    "Instructions",
                )
                config_file = st.file_uploader(
                    "Upload custom config", type="toml", help=readme["tooltips"]["custom_config"]
                )
                if config_file:
                    config = load_custom_config(config_file)
                else:
                    st.stop()
        if file:
            df = load_dataset(file, load_options)
        else:
            st.stop()

    datasets["uploaded"] = df.copy()
    return df, load_options, config, datasets


def input_columns(
    config: Dict[Any, Any], readme: Dict[Any, Any], df: pd.DataFrame, load_options: Dict[Any, Any]
) -> Tuple[str, str]:
    """Lets the user specify date and target column names.

    Parameters
    ----------
    config : Dict
        Lib config dictionary containing information about toy datasets (date and target column names).
    readme : Dict
        Dictionary containing tooltips to guide user's choices.
    df : pd.DataFrame
        Loaded dataset.
    load_options : Dict
        Loading options selected by user (upload or download, dataset name if download).

    Returns
    -------
    str
        Date column name.
    str
        Target column name.
    """
    if load_options["adnex_dataset"]:
        date_col = st.selectbox(
            "Date column",
            [config["datasets"][load_options["dataset"]]["date"]],
            help=readme["tooltips"]["date_column"],
        )
        target_options = [value for key, value in config["datasets"][load_options["dataset"]]["target"].items()]
        target_col = st.selectbox(
            "Target column",
            target_options,
            help=readme["tooltips"]["target_column"],
        )
    else:
        date_col = st.selectbox(
            "Date column",
            sorted(df.columns)
            if config["columns"]["date"] in ["false", False]
            else [config["columns"]["date"]],
            help=readme["tooltips"]["date_column"],
        )
        target_col = st.selectbox(
            "Target column",
            sorted(set(df.columns) - {date_col})
            if config["columns"]["target"] in ["false", False]
            else [config["columns"]["target"]],
            help=readme["tooltips"]["target_column"],
        )
    return date_col, target_col


def input_future_regressors(
    datasets: Dict[Any, Any],
    dates: Dict[Any, Any],
    params: Dict[Any, Any],
    dimensions: Dict[Any, Any],
    load_options: Dict[Any, Any],
    date_col: str,
) -> pd.DataFrame:
    """Adds future regressors dataframe in datasets dictionary's values.

    Parameters
    ----------
    datasets : Dict
        Dictionary storing all dataframes.
    dates : Dict
        Dictionary containing future forecasting dates information.
    params : Dict
        Dictionary containing all model parameters and list of selected regressors.
    dimensions : Dict
        Dictionary containing dimensions information.
    load_options : Dict
        Loading options selected by user (including csv delimiter).
    date_col : str
        Name of date column.

    Returns
    -------
    dict
        The datasets dictionary containing future regressors dataframe.
    """
    if len(params["regressors"].keys()) > 0:
        regressors_col = list(params["regressors"].keys())
        start, end = dates["forecast_start_date"], dates["forecast_end_date"]
        tooltip = (
            f"Please upload a csv file with delimiter '{load_options['separator']}' "
            "and the same format as input dataset, ie with the following specifications: \n"
        )
        tooltip += (
            f"- Date column named `{date_col}`, going from **{start.strftime('%Y-%m-%d')}** "
            f"to **{end.strftime('%Y-%m-%d')}** at the same frequency as input dataset "
            f"and at format **{load_options['date_format']}**. \n"
        )
        dimensions_col = [col for col in dimensions.keys() if col != "agg"]
        if len(dimensions_col) > 0:
            if len(dimensions_col) > 1:
                tooltip += (
                    f"- Columns with the following names for dimensions: `{', '.join(dimensions_col[:-1])}, "
                    f"{dimensions_col[-1]}`. \n"
                )
            else:
                tooltip += f"- Dimension column named `{dimensions_col[0]}`. \n"
        if len(regressors_col) > 1:
            tooltip += (
                f"- Columns with the following names for regressors: `{', '.join(regressors_col[:-1])}, "
                f"{regressors_col[-1]}`."
            )
        else:
            tooltip += f"- Regressor column named `{regressors_col[0]}`."
        regressors_file = st.file_uploader(
            "Upload a csv file for regressors", type="csv", help=tooltip
        )
        if regressors_file:
            datasets["future_regressors"] = load_dataset(regressors_file, load_options)
    else:
        st.write("There are no regressors selected.")
    return datasets
