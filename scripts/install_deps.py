import argparse
import subprocess
import sys
import tomllib
from typing import Dict
from typing import List

PYPROJECT_FILE = "pyproject.toml"


def load_optional_dependencies() -> Dict[str, List[str]]:
    """Load optional dependencies from pyproject.toml using tomllib."""
    with open(PYPROJECT_FILE, "rb") as f:  # Open file in binary mode
        pyproject = tomllib.load(f)
    return pyproject.get("project", {}).get("optional-dependencies", {})


def resolve_dependencies(
    groups: List[str], optional_deps: Dict[str, List[str]]
) -> List[str]:
    """Resolve dependencies, including references to other groups."""
    resolved = set()
    stack = list(groups)  # Stack to process groups

    while stack:
        group = stack.pop()
        if group not in optional_deps:
            raise ValueError(
                f"Group '{group}' does not exist in optional dependencies. Optional"
                f" groups: {', '.join(optional_deps)}"
            )

        for dep in optional_deps[group]:
            if dep.startswith("arctic_training["):  # Handle group references
                ref_group = dep.split("[")[1].rstrip("]")
                stack.append(ref_group)
            else:
                resolved.add(dep)

    return sorted(resolved)  # Return sorted list for consistency


def install_dependencies(groups: List[str]) -> None:
    """Install dependencies for the specified groups."""
    optional_deps = load_optional_dependencies()
    all_deps = resolve_dependencies(groups, optional_deps)

    if not all_deps:
        print("No dependencies to install.")
        return

    subprocess.check_call([sys.executable, "-m", "pip", "install"] + all_deps)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Install optional dependencies from pyproject.toml."
    )
    parser.add_argument(
        "groups",
        nargs="+",
        help=(
            "The names of the optional dependency groups to install (e.g., docs, dev)."
        ),
    )
    args = parser.parse_args()
    install_dependencies(args.groups)


if __name__ == "__main__":
    main()
