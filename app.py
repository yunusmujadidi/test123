import streamlit as st
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer
import torch

@st.cache(allow_output_mutation=True)
def get_model():
    tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
    model = BertForSequenceClassification.from_pretrained('blacklotusid/id-hs-indobert')
    return tokenizer,model


tokenizer,model = get_model()

user_input = st.text_area('Masukkan text yang ingin di prediksi')
button = st.button("Prediksi")

d = {
    
  1:'Hate Speech',
  0:'Non HateSpeech'
}

if user_input and button :
    test_sample = tokenizer([user_input], padding=True, truncation=True, max_length=512,return_tensors='pt')
    # test_sample
    output = model(**test_sample)
    st.write("Logits: ",output.logits)
    y_pred = np.argmax(output.logits.detach().numpy(),axis=1)
    st.write("Prediksi: ",d[y_pred[0]])
