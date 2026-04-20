from argparse import ArgumentParser

from llama_index.core import VectorStoreIndex, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.web import SimpleWebPageReader


def main() -> None:
    parser = ArgumentParser()

    parser.add_argument(
        "llm",
        help="The Ollama model name. Make sure it has already been pulled."
    )
    parser.add_argument(
        "url",
        help="Known URL to fetch and analyze."
    )
    parser.add_argument(
        "--embedding",
        help="HuggingFace embedding model.",
        default="BAAI/bge-small-en-v1.5",
    )
    parser.add_argument(
        "--html_to_text",
        action="store_true",
        help="Convert HTML to plain text before indexing. Leave off if you want to preserve raw HTML."
    )
    parser.add_argument(
        "--query",
        default=(
            "Explain whether this candidate is suitable for a chef"
        ),
        help="Query to ask after indexing the page."
    )

    args = parser.parse_args()

    Settings.llm = Ollama(model=args.llm, request_timeout=3600.0)
    Settings.embed_model = HuggingFaceEmbedding(model_name=args.embedding)

    print(f"Loading webpage from: {args.url}")
    documents = SimpleWebPageReader(html_to_text=args.html_to_text).load_data([args.url])

    print(f"Loaded {len(documents)} document(s)")
    if documents:
        print(f"Preview length: {len(documents[0].text)} chars")

    print("Building vector index...")
    index = VectorStoreIndex.from_documents(documents)

    print("Creating query engine...")
    query_engine = index.as_query_engine()

    print("Running query...")
    response = query_engine.query(args.query)

    print("\n--- RESPONSE ---")
    print(response)


if __name__ == "__main__":
    main()
