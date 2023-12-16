"""
hyphal nodes are the representations of any and or all information
they have their own representation
and they have child nodes
"""

import yaml

from pathlib import Path

DEFAULT_PATH_ROOT = ".embeddings"
DEFAULT_PATH_DATA = Path("data")
INDEX_TREES = 2
CHUNK_SIZE = 512  # 512 characters in document
CHUNK_OVERLAP = 64  # 64 characters overlap


class HyphalFileNode:
    """
    this is the file tree representation of the data directory
    it contains the root path, i.e. where the data lives
    and the list of directories inside the root directory
    """

    root: Path
    children: list[Path]

    def __init__(self, root: Path = DEFAULT_PATH_DATA, children: list[Path] = []):
        self.root = root
        # now we have all children
        if children == []:
            self.children = self.get_children(self.root)
        else:
            self.children = children

    def __repr__(self):
        root = str(self.root)

        children = []
        for child in self.children:
            child_root = str(child.root)
            child_children = [str(grandchild) for grandchild in child.children]
            children.append({"root": child_root, "children": child_children})

        node_dict = {"root": root, "children": children}
        return yaml.dump(node_dict, sort_keys=False, default_flow_style=False)

    def get_children(self, path) -> list[Path]:
        directories = [
            child
            for child in path.iterdir()
            if child.is_dir() and not child.name.startswith(".")
        ]

        child_nodes = []
        for directory in directories:
            print(directory)
            filenames = [filename for filename in directory.iterdir()]
            child_node = HyphalFileNode(root=directory, children=filenames)
            child_nodes.append(child_node)

        return child_nodes


if __name__ == "__main__":
    hyphal_tree = HyphalFileNode()
    print(hyphal_tree)
