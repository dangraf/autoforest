import streamlit as st
import pandas as pd
from pathlib import Path
from autoforest.gui_df_clean.gui_categorical import *
from autoforest.gui_df_clean.gui_continuous import *
from autoforest.gui_df_clean.st_api import *
from autoforest.clean_data import *
from io import BytesIO

__all__ = ['run_state_machine']

from autoforest.operations import *


def read_dataframe(uploaded_file):
    p = Path(uploaded_file.name)
    print(p.suffix)
    if p.suffix == '.csv':
        df = pd.read_csv(uploaded_file, index_col=[0])
    elif p.suffix == '.json':
        df = pd.read_json(uploaded_file)
    elif p.suffix == '.pcl':
        df = pd.read_picke(uploaded_file)
    elif p.suffix == '.feather':
        df = pd.read_feather(uploaded_file)
    else:
        df = pd.DataFrame()
    return df


def show_type_conversion(stobj):
    label = get_label()
    df = get_df()

    def try_convert():
        df[label] = NormalizedDtype.try_convert(column=df[label],
                                                stype=st.session_state.try_convert)

    stobj.selectbox(label='Column Type:',
                    options=NormalizedDtype.get_list_of_types(),
                    index=NormalizedDtype.get_index_from_dtype(df[label].dtype),
                    on_change=try_convert,
                    key='try_convert')
    return


def show_header(stobj):
    df = get_df()
    label = get_label()
    col_index = get_col_index()

    stobj.write(f"# {df.columns[col_index]} {col_index}/{len(df.columns)}")
    na_mask = get_na_mask(df, label)
    num_na = na_mask.sum()
    na_pct = num_na.sum() / len(df) * 100
    ntype = NormalizedDtype.get_normalized_dtype(df[label].dtype)
    if ntype.value == NormalizedDtype.Int.value or \
            ntype.value == NormalizedDtype.Float.value:
        inf_mask = get_inf_mask(df, label)
        inf_pct = inf_mask.sum() / len(df) * 100

        stobj.write(f"**dtype:** {df[label].dtype.name}")
        stobj.write(f"**NaN values:** {na_pct:.2f}% num: {num_na}")
        stobj.write(f"**Inf values:** {inf_pct:.2f}%: num: {inf_mask.sum()}")
        stobj.write(f"**min:** {df[label].min():.2f} **max:** {df[label].max():.2f}")
        stobj.write(f"**std:** {df[label].std():.2f} **mean:** {df[label].mean():.2f}")
    else:
        stobj.write(f" **dtype:** {df[label].dtype.name},\n {na_pct:.2f}% NaN values")

    try:
        stobj.dataframe(df[~na_mask][label].iloc[:5])
    except BaseException as e:
        stobj.write(f"Error plotting dataframe: {e}")


def show_navigation_buttons(strobj):
    col1, col2 = strobj.columns(2)
    if col1.button('reset data'):
        label = get_label()
        df = get_df()
        orig_df = get_backup_df()
        df[label] = orig_df[label]
        st.experimental_rerun()

    if col2.button('save dataframe'):
        output = BytesIO()
        df = get_df()
        df.to_feather(output)
        st.download_button(
            "Press to Download",
            output,
            "dataframe.feather",
            "feather",
            key='download-feather')

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
        dtype = get_col_type()
        all = {'': None,
               'Na As Category': FillNaAsCategory,
               'Random Sampling': FillRandomSampling,
               'Median': FillMedian,
               'Mean': FillMean,
               'Fill Fwd': FillFwd,
               'Fill Bwd': FillBwd,
               'FillConstant': FillConstant,
               'Interpolate': FillInterpolate,
               'drop rows NA': DropNA}
        # Nan values does only not exist for int-values
        exclude_lookup = {NormalizedDtype.Float.value: ['Na As Category'],
                          NormalizedDtype.Categorical: ['Mean', 'Interpolate'],
                          NormalizedDtype.Datetime: ['Na As Category', 'Mean']}
        options = [value for value in all.keys() if value not in exclude_lookup[dtype.value]]

        option = stobj.selectbox('NA sampeling func', options=options)
        operation = all[option]
        label = get_label()
        kwargs = {'label': label}
        if option == 'FillConstant':
            operation = None
            # todo, add form to submit answer
            pass

        if operation is not None:
            tfm = option(**kwargs)
            df = tfm.encodes(df)
            add_operation(tfm)
        set_df(df)


def start_gui():
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        df = read_dataframe(uploaded_file)
        set_df(df_shrink(df))
        set_backup_df(df)
        set_state(CleanState.ITERATE_COLUMNS)
        set_col_index(0)
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
            show_continuous_stats()
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
