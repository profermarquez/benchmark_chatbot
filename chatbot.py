from typing import Annotated, List, Dict
from langchain_ollama import OllamaLLM
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from typing_extensions import TypedDict
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
import time
import logging
import os
import re

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

graph_builder = StateGraph(State)

llm = OllamaLLM(
    base_url="http://localhost:11434",
    model="mistral:latest",
    verbose=False,
    callbacks=[StreamingStdOutCallbackHandler()]
)

def chatbot(state: State):
    result = llm.invoke(
        "\n".join([message.content for message in state["messages"]])
    )
    return {"messages": [AIMessage(content=result)]}

graph_builder.add_node("chatbot", chatbot)
graph_builder.set_entry_point("chatbot")

memory = InMemorySaver()
graph = graph_builder.compile(checkpointer=memory)

def medir_tiempo_respuesta(funcion, *args, **kwargs):
    inicio = time.perf_counter()
    resultado = funcion(*args, **kwargs)
    fin = time.perf_counter()
    tiempo_ejecucion = fin - inicio
    return resultado, tiempo_ejecucion

def configurar_logging(nombre_archivo="chatbot_benchmark.log"):
    if os.path.exists(nombre_archivo):
        os.remove(nombre_archivo)

    logging.basicConfig(
        filename=nombre_archivo,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

def ejecutar_benchmark(graph, chat_history, thread_id, checkpoint_ns, checkpoint_id):
    resultado, tiempo_respuesta = medir_tiempo_respuesta(
        graph.stream,
        {"messages": chat_history},
        config={"thread_id": thread_id, "checkpoint_ns": checkpoint_ns, "checkpoint_id": checkpoint_id},
    )

    full_response = ""
    for output in resultado:
        for key, value in output.items():
            if isinstance(value, dict) and "messages" in value:
                for msg in value["messages"]:
                    if isinstance(msg, AIMessage):
                        full_response += msg.content + " "

    logging.info(f"Entrada: {chat_history[-1].content}")
    logging.info(f"Respuesta: {full_response}")
    logging.info(f"Tiempo de respuesta: {tiempo_respuesta:.8f} segundos")
    return full_response

def run_chat_console():
    chat_history = []
    while True:
        user_input = input("Usuario: ")
        if user_input.lower() in ["salir", "adiós", "terminar"]:
            print("Chat finalizado.")
            break

        chat_history.append(HumanMessage(content=user_input))

        full_response = ejecutar_benchmark(
            graph,
            chat_history,
            thread_id="1",
            checkpoint_ns="ns",
            checkpoint_id="id",
        )

        print("Asistente:", full_response)
        chat_history.append(AIMessage(content=full_response))

def evaluar_respuesta(respuesta: str, respuesta_correcta: str) -> bool:
    respuesta = respuesta.lower()
    respuesta_correcta = respuesta_correcta.lower()
    respuesta_extraida = re.search(r"[a-d]\b",respuesta)
    if respuesta_extraida:
        respuesta = respuesta_extraida.group(0)
    return respuesta == respuesta_correcta

def clasificar_comprension(precision: float) -> str:
    if precision == 1.0:
        return "Excelente"
    elif precision >= 0.8:
        return "Bueno"
    elif precision >= 0.6:
        return "Regular"
    else:
        return "Malo"

def ejecutar_benchmark_opcion_multiple(preguntas: List[Dict[str, str]]):
    resultados = []
    tiempos = []
    errores = []
    for pregunta in preguntas:
        entrada = pregunta["pregunta"]
        respuesta_correcta = pregunta["respuesta"]
        chat_history = [HumanMessage(content=entrada)]
        respuesta, tiempo = medir_tiempo_respuesta(ejecutar_benchmark, graph, chat_history, "1", "ns", "id")
        evaluacion = evaluar_respuesta(respuesta, respuesta_correcta)
        resultados.append(evaluacion)
        tiempos.append(tiempo)
        if not evaluacion:
            errores.append({"pregunta": entrada, "respuesta_modelo": respuesta, "respuesta_correcta": respuesta_correcta})
        logging.info(f"Pregunta: {entrada}")
        logging.info(f"Respuesta correcta: {respuesta_correcta}")
        logging.info(f"Respuesta del modelo: {respuesta}")
        logging.info(f"Evaluación: {evaluacion}")
        logging.info(f"Tiempo: {tiempo:.8f} segundos")

    precision = sum(resultados) / len(resultados) if resultados else 0
    tiempo_promedio = sum(tiempos) / len(tiempos) if tiempos else 0
    nivel_comprension = clasificar_comprension(precision)

    logging.info(f"Precisión total: {precision:.4f}")
    logging.info(f"Tiempo promedio: {tiempo_promedio:.8f} segundos")
    logging.info("-------------------------------0-----------------------------------")
    logging.info(f"Nivel de comprensión: {nivel_comprension}")

    if errores:
        logging.info(f"Errores: {errores}")

    return precision, tiempo_promedio

if __name__ == "__main__":
    configurar_logging()

    preguntas_opcion_multiple = [
        {
            "pregunta": "Cuál es la capital de Francia?\nA) Londres\nB) París\nC) Berlín\nD) Roma",
            "respuesta": "b",
        },
        {
            "pregunta": "Si todos los gatos son mamíferos y todos los mamíferos respiran, entonces todos los gatos:\nA) Vuelan\nB) Respiran\nC) Nadar\nD) No respiran",
            "respuesta": "b",
        },
        {
            "pregunta": "Que color es el cielo?\nA) verde\nB) rojo\nC) azul\nD) amarillo",
            "respuesta": "c"
        }
    ]

    ejecutar_benchmark_opcion_multiple(preguntas_opcion_multiple)
    #run_chat_console() #Descomentar para usar el modo chat interactivo.