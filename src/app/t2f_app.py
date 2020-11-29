import streamlit as st
from PIL import Image
from os.path import join
from os import listdir

from src.trainer import GANTrainer
from src.miscc.config import cfg

MODEL_FILE = '/home/sondn/DIY/StackGAN-Pytorch/output/res/step2.pth'
IMGS_DIR = '/home/sondn/DIY/StackGAN-Pytorch/output/res/step2'

st.title('Text-to-face app')
desc = st.text_area('Type your face description here')
ok = st.button('OK')
if ok:
    algo = GANTrainer('')
    algo.sample_one(desc=desc, samples=50, save_dir=IMGS_DIR, truncate_dir=True)

columns = st.beta_columns(5)
next = st.button('Next')
if next:
    for idx, img_file in enumerate(listdir(IMGS_DIR)):
        with columns[idx % 5]:
            img = Image.open(join(IMGS_DIR, img_file))
            st.image(img, channels='BGR', use_column_width=True)
