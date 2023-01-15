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


convert_options = NormalizedDtype.get_list_of_types()


def show_type_conversion(stobj):
    global convert_options
    label = get_label()

    cols = stobj.columns([3, 1])
    print(NormalizedDtype.get_list_of_types())
    selection = cols[0].selectbox(label='Change Column Type:',
                                  options=NormalizedDtype.get_list_of_types())
    if cols[1].button('convert') and selection != ' ':
        try:
            cast_tfm = SetDType(label=label, dtype=selection)
            df = get_df()
            df = cast_tfm.encodes(df)
            set_df(df)
            replace_operation(cast_tfm)
        except BaseException as e:
            print(f'error {e}')
            pass

    return


def small_font(mystr: str):
    return f"<p class='small-font'>{mystr}</p>"


def show_header(stobj):
    df = get_df()
    label = get_label()
    col_index = get_col_index()
    cols = stobj.columns(2)
    cols[0].metric('Column:', f"{df.columns[col_index]}")
    cols[1].metric('index:', f"{col_index}/{len(df.columns)}")

    na_mask = get_na_mask(df, label)
    num_na = na_mask.sum()
    na_pct = num_na.sum() / len(df) * 100
    ntype = NormalizedDtype.get_normalized_dtype(df[label].dtype)
    if ntype.value == NormalizedDtype.Int.value or \
            ntype.value == NormalizedDtype.Float.value:
        inf_mask = get_inf_mask(df, label)
        inf_pct = inf_mask.sum() / len(df) * 100

        stobj.write(f"**dtype:** {df[label].dtype.name}")
        cols = stobj.columns(2)
        cols[0].write(f"**NaN values:** {na_pct:.2f}%")
        cols[1].write(f"**num:** {num_na}")

        cols = stobj.columns(2)
        cols[0].write(f"**Inf values:** {inf_pct:.2f}%")
        cols[1].write(f"**num:** {inf_mask.sum()}")

        cols = stobj.columns(2)
        cols[0].write(f"**min:** {df[label].min():.2f}")
        cols[1].write(f"**max:**  {df[label].max():.2f}")

        cols[0].write(f"**std:** {df[label].std():.2f}")
        cols[1].write(f"**mean**, {df[label].mean():.2f}")
    else:
        cols = stobj.columns(2)
        cols[0].write(f"**dtype:** {df[label].dtype.name}")
        cols[1].write(f"**NaN values:** {na_pct:.2f}% ")

    try:
        cols = stobj.columns(2)
        cols[0].dataframe(df[~na_mask][label].iloc[:5])
        cols[1].dataframe(df[~na_mask][label].iloc[-5:])

    except BaseException as e:
        stobj.metric("Error", "plotting dataframe: {e}")
    reps = [str(p) for p in get_operations()]
    stobj.write('Pipeline:', '->'.join(reps))


def show_navigation_buttons(strobj):
    col1, col2 = strobj.columns(2)
    col_index = get_col_index()
    if col2.button('next'):
        set_col_index(col_index + 1)

    if col1.button('prev'):
        set_col_index(col_index - 1)

    col1, col2 = strobj.columns(2)
    if col1.button('reset data'):
        label = get_label()
        df = get_df()
        orig_df = get_backup_df()
        df[label] = orig_df[label]
        clear_operations()

    if col2.button('save dataframe'):
        output = BytesIO()
        df = get_df()
        df.to_feather(output)
        st.download_button(
            label="Press to Download",
            data=output,
            file_name="dataframe.feather",
            mime="feather")


def show_fillna(stobj):
    label = get_label()
    df = get_df()
    na_mask = get_na_mask(df, label)

    if na_mask.sum() != 0:
        stobj.write(
            '## Missing values\n',
            'Please use sampling methods below to fill missing values, note the "drop-na" method cant be undone')
        dtype = get_col_type()
        op_list = {'': None,
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
        options = [value for value in op_list.keys() if value not in exclude_lookup.get(dtype.value, [])]

        selection = stobj.selectbox('NA sampeling func', options=options, key='na_selectbox')

        operation: BaseTransform = op_list[selection]
        label = get_label()
        tfm = None
        if operation is not None:
            tfm = operation.show_form(stobj, None, label)
        if tfm:
            df = get_df()
            df = tfm.encodes(df)
            ops = get_operations()
            if len(ops) > 0:
                print(f" name: {ops[-1].name}")
            if len(ops) > 0 and ops[-1].name.startswith('Fill'):
                # we can only have one fill transform
                ops[-1] = tfm
            else:
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


def iterate_columns():
    show_type_conversion(st.sidebar)
    show_navigation_buttons(st.sidebar)
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
            pass
            show_continuous_stats()
        except BaseException as e:
            st.write(f"error plotting: {e}")
    show_header(st.sidebar)


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
    init_states()
    st.set_page_config(layout="wide")
    st.markdown("""
    <style>
    .small-font {
        font-size:12px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    run_state_machine()
