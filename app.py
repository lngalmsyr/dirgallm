import streamlit as st
from st_pages import Page, show_pages


show_pages(
    [
        Page("pages/home.py", "Home", "🏠"),
        Page("pages/chat.py", "Tanya DirgaInsight Sekarangg", ":books:"),
        Page("pages/visualisasi.py", "Halaman Visualisasi", "🧐")
    ])