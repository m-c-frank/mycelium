from pathlib import Path


def get_tree(root: Path):
    """
    returns paths similar to tree but as a list of paths
    excluding .files and .directories.
    """
    tree = []
    for file in root.rglob("*"):
        if file.is_file() and not any(
            part.startswith(".") for part in file.relative_to(root).parts
        ):
            relative_path = file.relative_to(root)
            target_file = relative_path
            tree.append(target_file)
    return tree


if __name__ == "__main__":
    tree = get_tree(Path("data"))
    print(tree)
