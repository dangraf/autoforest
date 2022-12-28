import streamlit as st
import pandas as pd
from pathlib import Path
from autoforest.gui_df_clean.gui_categorical import *
import matplotlib.pyplot as plt
from autoforest.gui_df_clean.st_api import *
from autoforest.clean_data import *

__all__ = ['run_state_machine']


def read_dataframe(uploaded_file):
    p = Path(uploaded_file.name)
    print(p.suffix)
    if p.suffix == '.csv':
        df = pd.read_csv(uploaded_file, index_col=[0])
    elif p.suffix == '.json':
        df = pd.read_json(uploaded_file)
    elif p.suffix == '.pcl':
        df = pd.read_picke(uploaded_file)
    return df


def show_type_conversion(stobj):
    label = get_label()
    df = get_df()

    def try_convert():
        df[label] = NormalizedDtype.try_convert(column=df[label],
                                                stype=st.session_state.try_convert)
        dt = df_shrink(df[[label]])
        print(dt.dtypes)
        #df[label].astype(dt[0])

    stobj.selectbox(label='Column Type:',
                    options=NormalizedDtype.get_list_of_types(),
                    index=NormalizedDtype.get_index_from_dtype(df[label].dtype),
                    on_change=try_convert,
                    key='try_convert')
    return


def show_header(stobj) -> str:
    df = get_df()
    label = get_label()
    col_index = get_col_index()
    all = len(df.columns)
    stobj.write(f"# {df.columns[col_index]} {col_index}/{all}")
    na_mask = get_na_mask(df, label)
    num_na = na_mask.sum()
    na_pct = num_na.sum() / len(df) * 100
    stobj.write(f" **dtype:** {df[label].dtype.name}, {na_pct:.2f}% NaN values")

    try:
        stobj.dataframe(df[~na_mask][label].iloc[:5])
    except BaseException as e:
        stobj.write(f"Error plotting dataframe: {e}")


def show_navigation_buttons(strobj):
    col1, col2 = strobj.columns(2)
    col_index = get_col_index()
    if col2.button('next'):
        set_col_index(col_index + 1)
        st.experimental_rerun()

    if col1.button('prev'):
        set_col_index(col_index - 1)
        st.experimental_rerun()


def show_fillna(stobj):
    label = get_label()
    df = get_df()
    na_mask = get_na_mask(df, label)

    if na_mask.sum() != 0:
        stobj.write(
            '## Missing values\n',
            'Please use sampling methods below to fill missing values, note the "drop-na" method cant be undone')
        option = stobj.selectbox('NA sampeling func',
                                 options=['', 'Random Sampling', 'Median', 'drop rows NA'])
        if option == 'drop rows NA':
            df = df[~na_mask].reset_index(drop=True)
        elif option == 'Median':
            fill_median(df, label)
        elif option == 'Random Sampling':
            fill_random_sampling(df, label)
        # todo: ffill, bfill, interpolate, mean
        set_df(df)




def start_gui():
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        df = read_dataframe(uploaded_file)
        st.session_state['df'] = df_shrink(df)
        st.session_state['state'] = CleanState.ITERATE_COLUMNS
        st.session_state['col_index'] = 0
        st.experimental_rerun()

def iterate_columns():

    show_header(st.sidebar)
    show_type_conversion(st.sidebar)
    show_fillna(st)

    df = get_df()
    label = get_label()
    ntype = NormalizedDtype.get_normalized_dtype(df[label].dtype)
    print(ntype.value)
    if ntype.value == NormalizedDtype.Categorical.value:
        print('Enter Categorical')
        show_categorigal_stats()
    else:
        try:
            df[label].plot()
            st.pyplot(plt.gcf())
        except BaseException as e:
            st.write(f"error plotting: {e}")
    show_navigation_buttons(st.sidebar)

def run_state_machine():
    if 'state' not in st.session_state:
        st.session_state['state'] = CleanState.SEL_FILE
    if 'col_index' not in st.session_state:
        st.session_state['col_index'] = -1

    if st.session_state['state'].value == CleanState.SEL_FILE.value:
        start_gui()
    if st.session_state['state'].value == CleanState.ITERATE_COLUMNS.value:
        iterate_columns()


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    run_state_machine()
