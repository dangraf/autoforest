import streamlit as st

if __name__ == "__main__":
    if "state" not in st.session_state:
        st.session_state['state'] = 1

    if st.session_state['state'] == 1:
        st.write('state 1')

    if st.session_state['state'] == 2:
        st.write('state 2')

    if st.button('next'):
        st.session_state['state'] += 1
        st.experimental_rerun()
    if st.button('reset'):
        st.session_state['state'] = 1
        st.experimental_rerun()
