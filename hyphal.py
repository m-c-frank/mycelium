from pathlib import Path
import utils.cli as cli
import numpy as np

from annoy import AnnoyIndex
from langchain.embeddings import OllamaEmbeddings


DEFAULT_PATH_ROOT = ".embeddings"
INDEX_TREES = 2

def make_hyphal(source_path):
    print(f"making hyphal at: {source_path}")

    source_path = Path(source_path)

    oembed = OllamaEmbeddings(
        base_url="http://localhost:11434",
        model="mistral"
    )

    def get_tree():
        tree = []
        for file in source_path.rglob("*"):
            if file.is_file() and not any(part.startswith('.') for part in file.relative_to(source_path).parts):
                relative_path = file.relative_to(source_path)
                target_file = relative_path
                tree.append(target_file)
        return tree

    tree = get_tree()
    print(tree)
    embeddings = oembed.embed_documents(tree)

    def write_hyphals():
        for i, filename in enumerate(tree):
            path = source_path / ".hyphal" / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path = path.with_suffix('.embedding')
            print(f"saving embedding to {path}")
            np.save(path, embeddings[i])

    write_hyphals()

    def make_index():
        f = len(embeddings[0])
        t = AnnoyIndex(f, "angular")
        for i, embedding in enumerate(embeddings):
            t.add_item(i, embedding)

        t.build(INDEX_TREES)  # binary? either similar or not similar for now
        annoy_index_path = source_path / ".hyphal" / "index.ann"
        t.save(str(annoy_index_path))

    make_index()

    return embeddings


def main():
    hyphal = cli.run(
        description="creates skeleton based on tree",
        func=make_hyphal,
        default_argument="./data",
        required=False,
        help_text="path to directory to generate mirrored file paths",
    )
    print(hyphal)


if __name__ == "__main__":
    main()
