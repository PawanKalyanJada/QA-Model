import streamlit as st
import tensorflow
from tensorflow.keras.layers import Input,Dense,Flatten,Activation
from tensorflow.keras.models import Model
from transformers import TFBertModel
from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
import numpy as np

#@st.cache(allow_output_mutation=True)
@st.experimental_singleton
def get_model():
    model=TFBertModel.from_pretrained('bert-base-uncased')
    inp1=Input((512,),dtype='int32')
    inp2=Input((512,),dtype='int32')
    inp3=Input((512,),dtype='int32')
    emb=model(inp1,attention_mask=inp2,token_type_ids=inp3)[0]
    s1=Dense(1,use_bias=False)(emb)
    s1=Flatten()(s1)
    s1=Activation(tensorflow.keras.activations.softmax)(s1)
    s2=Dense(1,use_bias=False)(emb)
    s2=Flatten()(s2)
    s2=Activation(tensorflow.keras.activations.softmax)(s2)
    m=Model(inputs=[inp1,inp2,inp3],outputs=[s1,s2])
    m.load_weights('Question Answering Model/model_weights')
    return m


def find_answer(context,question):
    m=get_model()
    enc=tokenizer(question,context,padding='max_length',max_length=512,truncation=True)
    k = np.array([enc['input_ids']])
    k1 = np.array([enc['attention_mask']])
    k2 = np.array([enc['token_type_ids']])
    res=m([k,k1,k2])
    start=np.argmax(res[0].numpy()[0])
    end=np.argmax(res[1].numpy()[0])
    return tokenizer.decode(k[0][start:end+1])

st.title("Question And Answering WebApp!")
st.subheader("Context:")
st.markdown("One direction has 5 members. Zayn malik left the band in the year 2015. Zayn wrote so many songs but did not release due to company's constraints. As of 2022, the band have sold a total of 70 million records worldwide,making them one of the best-selling boy bands of all time. Forbes ranked them as the fourth highest-earning celebrities in the world in 2015 and 2016.")
st.subheader("Question:")
st.markdown("How many people are there in one direction?")

form = st.form(key="form")
context = form.text_area("Enter the context here")
query = form.text_input("Enter the question here")
predict_button = form.form_submit_button(label='Submit')


if predict_button:
    st.write("Response:")
    with st.spinner('Finding Answer'):
        answer = find_answer(context,query)
        st.write("Answer:",answer)


footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Developed by <a style='display: block; text-align: center;' href="https://www.linkedin.com/in/pawan-kalyan-9704991aa/" target="_blank">Pawan Kalyan Jada</a></p>
<p>Email ID : <a style='display: block; text-align: center;' target="_blank">pawankalyanjada@gmail.com</a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)
