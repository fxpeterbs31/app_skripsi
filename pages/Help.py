import streamlit as st
from PIL import Image

st.markdown('# HELP :question:')
st.markdown('\n')

st.markdown('## Overview Aplikasi')
st.write(
    'Aplikasi ini adalah Aplikasi Simulasi Pergerakan Jembatan dengan menggunakan metode beda hingga nonstandar. Aplikasi ini memberikan simulasi pergerakan dengan bentuk tabel dan grafik berdasarkan apa yang telah terjadi pada Jembatan Tacoma pada tahun 1940 pada saat runtuh.'
)

st.markdown('## Cara Menggunakan Aplikasi')
st.write(
    'Aplikasi ini memiliki 3 halaman'
)

st.markdown('### 1. Home')
st.write(
    'Halaman ini berisi judul skripsi dan profil penulis.'
)
st.write('\n')

st.markdown('### 2. Help')
st.write(
    'Halaman ini berisi penjelasan singkat mengenai aplikasi dan daftar isi aplikasi'
)
st.write('\n')

st.markdown('### 3. Simulasi')
st.write(
    'Halaman ini berisi form input dan hasil untuk simulasi pergerakan jembatan.'
)
st.markdown('#### 3.1 Input')

img = Image.open('D:\Skripsi\Python\Screenshot 2023-06-05 190129.png')
st.image(img, 
         caption = 'Ilustrasi jembatan gantung tampak depan'
)

st.markdown('##### 3.1.1 Parameter')
st.write(
    'Parameter yang dibutuhkan berupa: '
)
st.write(
    '1. Massa Jembatan (m)'
)
st.write(
    '2. Lebar Jembatan (l)'
)
st.write(
    '3. Konstanta Pegas Jembatan (k)'
)
st.write(
    '4. Konstanta Peredam Pegas Jembatan (\u03B4)'
)
st.write(
    '5. Konstanta Nonlinearitas (\u03B1)'
)
st.write(
    '6. Step Size (h)'
)
st.write('\n')

st.markdown('##### 3.1.2 Initial Value')
st.write(
    'Nilai awal / Initial value yang dibutuhkan berupa: '
)
st.write(
    '1. Kemiringan jembatan (\u03B8)'
)
st.write(
    '2. Kecepatan sudut jembatan (v)'
)
st.write(
    '3. Selisih pososi jembatan terhadap posisi ekuilibrium (y)'
)
st.write(
    '4. Kecepatan jembatan (w)'
)

st.write('\n')
st.markdown('#### 3.2 Output')
st.markdown('##### 3.2.1 Hasil')
st.write(
    'Hasil yang ditampilkan ada 2, yaitu:  '
)
st.write(
    '1. Grafik'
)
st.write(
    '2. Tabel'
)

st.write('\n')

st.markdown('##### 3.2.2 Metode')
st.write(
    'Metode yang digunakan adalah: '
)
st.write(
    '1. Metode NSFD'
)
st.write(
    '2. Metode Euler'
)