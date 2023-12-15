import argparse

REQUIRED = True
DEFAULT_ARGUMENT = "./data"

def default_func(path):
    return "add your own function"


def run(
    description="takes in path and does something",
    func=default_func,
    required=REQUIRED,
    default_argument=DEFAULT_ARGUMENT,
    help_text="runs default function"
):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--path", default=default_argument, type=str, help=help_text, required=required)
    args = parser.parse_args()
    return func(args.path)


def main():
    print(run())

if __name__ == "__main__":
    main()
