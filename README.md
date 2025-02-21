# Instalar y activar un entorno virtual
/virtualenv env         /env/Scripts/activate.bat

# requerimientos
pip install psutil

pip install langchain langchain-community langchain-ollama langgraph streamlit pypdf typing-extensions

Ollama: Debes tener Ollama instalado y funcionando localmente, con el modelo mistral:latest descargado.

# ejecución
py chatbot.py

