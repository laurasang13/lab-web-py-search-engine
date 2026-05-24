"""
indexar.py — Indexación de artículos con embeddings en ChromaDB
"""

import os
import chromadb
from google import genai
from dotenv import load_dotenv

load_dotenv()

# ── Configuración ────────────────────────────────────────────────────────────
MODELO_EMBEDDING = "models/gemini-embedding-001"
COSTE_POR_1K_TOKENS = 0.00002          # USD, text-embedding-3-small
CHROMA_PATH = "./chroma_db"
COLECCION_NOMBRE = "articulos_tech"

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ── Dataset ──────────────────────────────────────────────────────────────────
articulos = [
    {"id": "1", "titulo": "FastAPI vs Flask",
     "contenido": "FastAPI ofrece validación automática con Pydantic, documentación Swagger integrada y mejor rendimiento asíncrono que Flask."},
    {"id": "2", "titulo": "React vs Vue",
     "contenido": "React tiene un ecosistema más grande y es mantenido por Meta. Vue tiene una curva de aprendizaje más suave y una sintaxis más intuitiva."},
    {"id": "3", "titulo": "PostgreSQL para principiantes",
     "contenido": "PostgreSQL es una base de datos relacional open source con soporte para JSON, búsqueda de texto completo y extensiones como pgvector."},
    {"id": "4", "titulo": "Introducción a los LLMs",
     "contenido": "Los Large Language Models son redes neuronales entrenadas para predecir la siguiente palabra. GPT-4, Claude y Gemini son ejemplos populares."},
    {"id": "5", "titulo": "Despliegue con Docker",
     "contenido": "Docker permite empaquetar aplicaciones en contenedores que se ejecutan de forma consistente en cualquier entorno."},
    {"id": "6", "titulo": "Autenticación JWT",
     "contenido": "JSON Web Tokens permiten transmitir información verificada entre partes. Se usan para autenticación stateless en APIs REST."},
    {"id": "7", "titulo": "LangChain para agentes",
     "contenido": "LangChain simplifica la construcción de aplicaciones con LLMs, proporcionando abstracciones para cadenas, agentes y memoria."},
    {"id": "8", "titulo": "Python vs JavaScript para IA",
     "contenido": "Python domina el ecosistema de IA gracias a librerías como NumPy, PyTorch y HuggingFace. JavaScript tiene opciones como TensorFlow.js pero es menos maduro."},
    # ── artículos extra (bonus: más variedad semántica) ──
    {"id": "9", "titulo": "GraphQL vs REST",
     "contenido": "GraphQL permite al cliente pedir exactamente los campos que necesita, evitando over-fetching. REST sigue siendo el estándar más extendido para APIs públicas."},
    {"id": "10", "titulo": "Kubernetes para desarrolladores",
     "contenido": "Kubernetes orquesta contenedores Docker a escala, gestionando réplicas, balanceo de carga y auto-escalado de servicios en producción."},
]

def contar_tokens(textos: list[str], modelo: str = MODELO_EMBEDDING) -> int:
    return sum(len(t.split()) for t in textos)

def crear_embedding(texto: str) -> list[float]:
    """Crea un embedding para un texto usando la API de OpenAI."""
    respuesta = client.models.embed_content(model=MODELO_EMBEDDING, contents=texto)
    return respuesta.embeddings[0].values


def ya_indexado(coleccion: chromadb.Collection, articulo_id: str) -> bool:
    """Comprueba si un artículo ya existe en la colección (indexación incremental)."""
    resultado = coleccion.get(ids=[articulo_id])
    return len(resultado["ids"]) > 0


# ── Indexación principal ─────────────────────────────────────────────────────

def indexar():
    # Conectar / crear base de datos ChromaDB persistente
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    coleccion = chroma_client.get_or_create_collection(
        name=COLECCION_NOMBRE,
        metadata={"hnsw:space": "cosine"},   # similitud coseno
    )

    print(f"📂 Colección '{COLECCION_NOMBRE}' — documentos existentes: {coleccion.count()}\n")

    # Filtrar artículos que aún no están indexados (bonus incremental)
    pendientes = [a for a in articulos if not ya_indexado(coleccion, a["id"])]

    if not pendientes:
        print("✅ Todos los artículos ya están indexados. Nada que hacer.")
        return

    print(f"🔄 Artículos a indexar: {len(pendientes)}")

    # Estimar tokens y coste ANTES de llamar a la API
    textos_contenido = [a["contenido"] for a in pendientes]
    total_tokens = contar_tokens(textos_contenido)
    coste_estimado = (total_tokens / 1000) * COSTE_POR_1K_TOKENS

    print(f"📊 Tokens totales : {total_tokens:,}")
    print(f"💵 Coste estimado : ${coste_estimado:.6f} USD\n")

    # Crear embeddings e insertar en ChromaDB
    ids, embeddings, documentos, metadatos = [], [], [], []

    for art in pendientes:
        print(f"  ⚙️  Embeddiendo: [{art['id']}] {art['titulo']}")
        embedding = crear_embedding(art["contenido"])

        ids.append(art["id"])
        embeddings.append(embedding)
        documentos.append(art["contenido"])
        metadatos.append({"titulo": art["titulo"], "id": art["id"]})

    coleccion.add(
        ids=ids,
        embeddings=embeddings,
        documents=documentos,
        metadatas=metadatos,
    )

    print(f"\n✅ Indexación completada. Total en colección: {coleccion.count()} artículos.")


if __name__ == "__main__":
    indexar()
