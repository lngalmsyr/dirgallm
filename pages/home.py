import streamlit as st
from st_pages import add_page_title
from streamlit_extras.colored_header import colored_header

add_page_title(layout="wide")

colored_header(
    label="MESIN ARTIFICIAL INTELLIGENCE PT DIRGAPUTRA EKAPRATAMA", 
    description="Masih dalam tahap pengembangan",
    color_name="violet-70"
)

st.write("""
        Aplikasi ini dirancang untuk membantu Anda menggali *insight* berharga dari data, sehingga dapat mendukung pengambilan keputusan yang lebih akurat karena berbasis data.
    """)

st.write("""
        **Keterbatasan dan Pengembangan Mendatang**

        Sebagai aplikasi yang masih dalam tahap pengembangan, terdapat beberapa hal yang perlu diperhatikan:
        *   Untuk hasil yang lebih optimal, disarankan untuk menuliskan prompt yang spesifik dan relevan dengan data yang ingin Anda analisis.
        *   Kami sangat menghargai masukan dan kontribusi dari Anda untuk pengembangan aplikasi ini lebih lanjut. 
""")

st.write("""
    **Terima kasih telah menggunakan aplikasi ini!**
""")

st.write("""
    Kode sumber tersedia di GitHub:
        [Link ke Repository GitHub](https://github.com/lngalmsyr/hoax-app.git)
""")