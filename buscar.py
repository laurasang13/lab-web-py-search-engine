"""
buscar.py — Motor de búsqueda semántica sobre ChromaDB
"""

import os
import chromadb
from google import genai
from dotenv import load_dotenv

load_dotenv()

# ── Configuración ────────────────────────────────────────────────────────────
MODELO_EMBEDDING = "models/gemini-embedding-001"
CHROMA_PATH = "./chroma_db"
COLECCION_NOMBRE = "articulos_tech"

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
coleccion = chroma_client.get_or_create_collection(
    name=COLECCION_NOMBRE,
    metadata={"hnsw:space": "cosine"},
)


# ── Función principal ─────────────────────────────────────────────────────────

def buscar(query: str, n_resultados: int = 3) -> list[dict]:
    """
    Busca los N artículos más similares a la query.

    Retorna lista de dicts con:
        - id, titulo, contenido, score (distancia coseno → similitud)
    """
    # 1. Embedding de la query
    respuesta = client.models.embed_content(model=MODELO_EMBEDDING, contents=query)
    query_embedding = respuesta.embeddings[0].values

    # 2. Búsqueda en ChromaDB
    resultados = coleccion.query(
        query_embeddings=[query_embedding],
        n_results=min(n_resultados, coleccion.count()),
        include=["documents", "metadatas", "distances"],
    )

    # 3. Formatear salida
    # ChromaDB devuelve distancia coseno [0,2]; convertimos a similitud [0,1]
    salida = []
    for doc, meta, dist in zip(
        resultados["documents"][0],
        resultados["metadatas"][0],
        resultados["distances"][0],
    ):
        similitud = 1 - (dist / 2)          # normalizar a [0,1]
        salida.append({
            "id": meta["id"],
            "titulo": meta["titulo"],
            "contenido": doc,
            "score": round(similitud, 4),
        })

    return salida


# ── BONUS: búsqueda combinada título + contenido ─────────────────────────────

def buscar_con_titulo(query: str, n_resultados: int = 3) -> list[dict]:
    """
    Versión extendida: crea el texto de búsqueda combinando título y contenido
    para que el título también influya en la similitud semántica.

    Para esto usamos una colección separada donde el documento indexado
    es 'titulo + contenido'.  Si la colección no existe se indica al usuario.
    """
    COLECCION_TITULO = "articulos_tech_titulo"
    try:
        col_titulo = chroma_client.get_collection(COLECCION_TITULO)
    except Exception:
        print(
            "⚠️  Colección con título no encontrada. "
            "Ejecuta indexar_con_titulo() primero."
        )
        return buscar(query, n_resultados)

    respuesta = client.models.embed_content(model=MODELO_EMBEDDING, contents=query)
    query_embedding = respuesta.embeddings[0].values

    resultados = col_titulo.query(
        query_embeddings=[query_embedding],
        n_results=min(n_resultados, col_titulo.count()),
        include=["documents", "metadatas", "distances"],
    )

    salida = []
    for doc, meta, dist in zip(
        resultados["documents"][0],
        resultados["metadatas"][0],
        resultados["distances"][0],
    ):
        similitud = 1 - (dist / 2)
        salida.append({
            "id": meta["id"],
            "titulo": meta["titulo"],
            "contenido": meta.get("contenido", doc),
            "score": round(similitud, 4),
        })
    return salida


# ── CLI de prueba ─────────────────────────────────────────────────────────────

QUERIES_PRUEBA = [
    "¿cómo hacer una API en Python?",
    "diferencias entre frameworks de frontend",
    "cómo funciona la autenticación en aplicaciones web",
    "herramientas para trabajar con modelos de lenguaje",
]


def _separador(char="─", ancho=70):
    print(char * ancho)


def demo():
    print("\n🔍 Motor de búsqueda semántica — Demo\n")
    _separador()

    for query in QUERIES_PRUEBA:
        print(f"\n📌 Query: «{query}»\n")
        resultados = buscar(query, n_resultados=3)
        for i, r in enumerate(resultados, 1):
            barra = "█" * int(r["score"] * 20)
            print(f"  {i}. [{r['score']:.4f}] {barra}")
            print(f"     📄 {r['titulo']}")
            print(f"     {r['contenido'][:80]}{'...' if len(r['contenido']) > 80 else ''}")
        _separador()


if __name__ == "__main__":
    demo()
