"""
hyphals are the representations of any and or all information
either as a single latent embedding
or as a list of embeddings
"""

from pathlib import Path
import utils.cli as cli
import numpy as np
from scipy.special import softmax

from annoy import AnnoyIndex
from langchain.embeddings import OllamaEmbeddings
from langchain_core.documents.base import Document
from langchain.document_loaders import PyPDFLoader

DEFAULT_PATH_ROOT = ".embeddings"
INDEX_TREES = 2


class HyphalDocument:
    root: Path  # path to the root directory of the dataset
    path: Path  # path to hyphal node
    content: str  # plain text representation of this hyphal
    embedding: list[float]  # embedding vector of this hyphal document
    documents: list[Document]  # list of documents that represent this node

    def __init__(self, source_path: Path, path: Path = Path("")):
        self.root = source_path
        self.path = path
        if path.suffix == ".pdf":
            self.from_pdf()

    def __str__(self) -> str:
        return self.content

    def get_source_path(self):
        source_path = self.root / self.path
        return source_path

    def from_pdf(self):
        doc_loader = PyPDFLoader(str(self.get_source_path()))
        documents = []
        content = ""
        chunks = doc_loader.load()

        for i, doc in enumerate(chunks):
            documents.append(doc)
            structure = f"start of chunk {i+1} of {len(chunks)}:"
            content = f"{content}\n{structure}\n{doc.page_content}"

        self.content = content
        self.documents = documents

    @staticmethod
    def embed(hyphal_document: "HyphalDocument"):
        oembed = OllamaEmbeddings(
            base_url="http://localhost:11434",
            model="mistral",
            show_progress=True
        )

        def embed_list(hyphal_documents: list[HyphalDocument]):
            content_strings = [str(i.content) for i in hyphal_documents]

            embedded_documents = oembed.embed_documents(content_strings)

            return embedded_documents

        embedding = embed_list([hyphal_document])[0]

        hyphal_document.embedding = embedding

    @staticmethod
    def from_tree(source_path: Path, tree: list[Path]) -> "HyphalDocument":
        documents = []
        for path in tree:
            hyphal_document = HyphalDocument(source_path, path)
            HyphalDocument.embed(hyphal_document)
            documents.append(hyphal_document)

        hyphal_document = HyphalDocument(source_path)
        HyphalDocument.embed(hyphal_document)

        return hyphal_document


def make_hyphal(source_path):
    print(f"making hyphal at: {source_path}")

    source_path = Path(source_path)

    def get_tree():
        tree = []
        for file in source_path.rglob("*"):
            if file.is_file() and not any(
                part.startswith(".") for part in file.relative_to(source_path).parts
            ):
                relative_path = file.relative_to(source_path)
                target_file = relative_path
                tree.append(target_file)
        return tree

    tree = get_tree()
    print(tree)

    hyphal_document = HyphalDocument.from_tree(source_path, tree)

    def write_hyphals():
        path_hyphal = source_path / ".hyphal"
        for i, filename in enumerate(tree):
            path = path_hyphal / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path = path.with_suffix(".embedding")
            print(f"saving embedding to {path}")
            np.save(path, hyphal_document.embedding)
        return path_hyphal

    write_hyphals()

    def make_index() -> tuple[str, AnnoyIndex, dict]:
        embeddings = [i.embedding for i in hyphal_documents]
        f = len(embeddings[0])
        t = AnnoyIndex(f, "angular")
        index_to_path = {}  # Mapping from index to file path
        for i, embedding in enumerate(embeddings):
            t.add_item(i, embedding)
            index_to_path[i] = str(tree[i])

        t.build(INDEX_TREES)
        annoy_index_path = source_path / ".hyphal" / "index.ann"
        t.save(str(annoy_index_path))
        return str(annoy_index_path), t, index_to_path

    path_index, index, index_to_path = make_index()

    return


def find_similar_documents(new_vector, index, index_to_path, n=10):
    similar_indices = index.get_nns_by_vector(new_vector, n, include_distances=True)
    distances = similar_indices[1]
    softmax_scores = softmax(-np.array(distances))

    results = []
    for idx, score in zip(similar_indices[0], softmax_scores):
        document_path = index_to_path[idx]
        embedding_path = Path(document_path).with_suffix(".embedding")
        embedding_value = np.load(embedding_path)
        results.append(
            {
                "document_path": document_path,
                "confidence": score,
                "embedding_path": str(embedding_path),
                "embedding": embedding_value,
            }
        )

    return results


def main():
    hyphal = cli.run(
        description="creates skeleton based on tree",
        func=make_hyphal,
        default_argument="./data",
        required=False,
        help_text="path to directory to generate mirrored file paths",
    )

    # query = "what should i do now"
    # embedded_query = embed_documents()

    print(hyphal)


if __name__ == "__main__":
    main()
