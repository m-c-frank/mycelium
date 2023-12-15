from pathlib import Path
import utils.cli as cli

DEFAULT_PATH_ROOT = ".embeddings"


def get_hyphal(source_path):
    source_path = Path(source_path)
    mirrored_files = []

    for file in source_path.rglob("*"):
        if file.is_file() and not file.name.startswith("."):
            relative_path = file.relative_to(source_path)
            target_file = relative_path
            mirrored_files.append(str(target_file))

    return mirrored_files


def main():
    hyphal = cli.run(
        description="creates skeleton based on tree",
        func=get_hyphal,
        default_argument="./data",
        required=False,
        help_text="path to directory to generate mirrored file paths",
    )
    print(hyphal)


if __name__ == "__main__":
    main()
