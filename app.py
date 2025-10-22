import os
import gdown
import streamlit as st
import tensorflow as tf

@st.cache_resource
def carrega_modelo():
    url = "https://drive.google.com/uc?id=1Ck0fIdSIMbkGuCwZ4h-lUwW4mT7hmf0u"
    output = "modelo_quantizado16bits.tflite"

    try:
        if not os.path.exists(output):
            st.info("Baixando o modelo, aguarde...")
            gdown.download(url, output, quiet=False)
        interpreter = tf.lite.Interpreter(model_path=output)
        interpreter.allocate_tensors()
        return interpreter

    except Exception as e:
        st.error("❌ Erro ao baixar o modelo do Google Drive. Verifique se o link está público.")
        st.exception(e)
        return None
