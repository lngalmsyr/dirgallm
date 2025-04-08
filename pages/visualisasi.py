import streamlit as st
import pandas as pd
from streamlit_extras.dataframe_explorer import dataframe_explorer
from streamlit_extras.colored_header import colored_header
from st_pages import add_page_title
import time

add_page_title(layout="wide")

colored_header(
    label="Data Berita Palsu/Hoax",
    description="",
    color_name="violet-70",
)
