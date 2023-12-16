"""
hyphal nodes are the representations of any and or all information
they have their own representation
and they have child nodes
"""

import yaml

from pathlib import Path

from hyphal_embedding import HyphalEmbedding

DEFAULT_PATH_DATA = Path("data")
DEFAULT_PATH_HYPHAL = Path(".hyphal")


class HyphalNode:
    """
    this is the file tree representation of the data directory
    it contains the root path, i.e. where the data lives
    and the list of directories inside the root directory
    """

    hyphal: Path
    root: Path
    embedding: HyphalEmbedding
    children: list[Path]

    def __init__(
        self,
        hyphal: Path = DEFAULT_PATH_HYPHAL,
        root: Path = DEFAULT_PATH_DATA,
        children: list[Path] = []
    ):
        self.root = root
        self.hyphal = hyphal

        if children == []:
            self.children = self.get_children(self.root)
        else:
            self.children = children

        self.save_index()
        self.embedding = HyphalEmbedding.embed_str(repr(self), self.get_hyphal())

    def __repr__(self):
        root = str(self.root)
        children = [yaml.safe_load(repr(child)) for child in self.children]
        node_dict = {"root": root, "children": children}
        return yaml.dump(node_dict, sort_keys=False, default_flow_style=False)

    def get_hyphal(self, hyphal_dir="index"):
        return DEFAULT_PATH_DATA / ".hyphal" / hyphal_dir / self.root

    def save_index(self):
        path = self.get_hyphal().with_suffix(".yaml")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            print(f"saving node to {path}")
            f.write(repr(self))

    def get_children(self, path) -> list[Path]:
        directories = [
            child
            for child in path.iterdir()
            if child.is_dir() and not child.name.startswith(".")
        ]

        child_nodes = []
        for directory in directories:
            filenames = [filename for filename in directory.iterdir()]
            child_node = HyphalNode(
                root=directory.relative_to(path), hyphal=self.hyphal / directory ,children=filenames
            )
            child_nodes.append(child_node)

        return child_nodes


if __name__ == "__main__":
    hyphal_tree = HyphalNode()
    print(hyphal_tree)
