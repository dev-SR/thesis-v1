from rich.console import Console
console = Console()
from rich.tree import Tree
from pprint import pprint
from rich import print


def withLoader(cb, message="", spinner='aesthetic'):
    done = False
    returns = None
    with console.status(f"[bold yellow] {message}...", spinner=spinner) as s:
        while not done:
            returns = cb()
            done = True
    return returns


def withLoaderListAsParam(cb, param, message="", spinner='aesthetic'):
    done = False
    returns = None
    with console.status(f"[bold yellow] {message}...", spinner=spinner) as s:
        while not done:
            returns = cb(*param)
            done = True
    return returns


def withLoaderDictAsParam(cb, param, message="", spinner='aesthetic'):
    done = False
    returns = None
    with console.status(f"[bold yellow] {message}...", spinner=spinner) as s:
        while not done:
            returns = cb(**param)
            done = True
    return returns


def print_tree(paper_list):
    tree = Tree(
        f"Recommendation Order: 📄↘️",
        guide_style="bold bright_blue",
    )
    branch = tree
    for paper in paper_list:
        branch = branch.add(f"📄 {paper}")

    print(tree)
