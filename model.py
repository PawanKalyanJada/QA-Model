import streamlit as st
import tensorflow
from tensorflow.keras.layers import Input,Dense,Flatten,Activation
from tensorflow.keras.models import Model
from transformers import TFBertModel
from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
import numpy as np

@st.cache(allow_output_mutation=True)
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

m = get_model()
def find_answer(context,question):
    enc=tokenizer(question,context,padding='max_length',max_length=512,truncation=True)
    k = np.array([enc['input_ids']])
    k1 = np.array([enc['attention_mask']])
    k2 = np.array([enc['token_type_ids']])
    res=m([k,k1,k2])
    start=np.argmax(res[0].numpy()[0])
    end=np.argmax(res[1].numpy()[0])
    return tokenizer.decode(k[0][start:end+1])

st.title("Question And Answering WebApp!!!")
context = st.text_area("Enter the context here")
query = st.text_input("Enter the query")
predict_button = st.button('Submit')
if predict_button:
    answer = find_answer(context,query)
    st.text(answer)
