"""
hyphal nodes are the representations of any and or all information
they have their own representation
and they have child nodes
"""

import yaml

from pathlib import Path

from hyphal_embedding import HyphalEmbedding

DEFAULT_PATH_DATA = Path("data")
DEFAULT_PATH_HYPHAL = Path(".hyphal/tree.yaml")
INDEX_TREES = 2
CHUNK_SIZE = 512  # 512 characters in document
CHUNK_OVERLAP = 64  # 64 characters overlap


class HyphalNode:
    """
    this is the file tree representation of the data directory
    it contains the root path, i.e. where the data lives
    and the list of directories inside the root directory
    """

    root: Path
    embedding: HyphalEmbedding
    children: list[Path]

    def __init__(self, root: Path = DEFAULT_PATH_DATA, children: list[Path] = []):
        self.root = root
        # now we have all children
        if children == []:
            self.children = self.get_children(self.root)
        else:
            self.children = children

        self.embedding = HyphalEmbedding.from_node(self)

    def __repr__(self):
        root = str(self.root)
        children = [yaml.safe_load(repr(child)) for child in self.children]
        node_dict = {"root": root, "children": children}
        return yaml.dump(node_dict, sort_keys=False, default_flow_style=False)

    def save(self, path_to_hyphal: Path = DEFAULT_PATH_HYPHAL):
        path = self.root / path_to_hyphal
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(repr(self))

    def get_children(self, path) -> list[Path]:
        directories = [
            child
            for child in path.iterdir()
            if child.is_dir() and not child.name.startswith(".")
        ]

        child_nodes = []
        for directory in directories:
            filenames = [filename.stem for filename in directory.iterdir()]
            child_node = HyphalNode(root=directory.relative_to(path), children=filenames)
            child_nodes.append(child_node)

        return child_nodes


if __name__ == "__main__":
    hyphal_tree = HyphalNode()
    hyphal_tree.save()
    print(hyphal_tree)
