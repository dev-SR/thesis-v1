from subprocess import check_output, CalledProcessError


def handleGitCommit(m):
    try:
        check_output("git add .", shell=True)
        s = check_output(f"git commit -m \"{m}\"", shell=True).decode()
        return f"[green]Successfully Committed:[/green] {s}"
    except CalledProcessError as e:
        return f"[red]Error: {e.output.decode()}[/]"


def handleGitPull():
    try:
        s = check_output("git pull origin main", shell=True).decode()
        return f"[green]Pulled From Remote Repo:[/green] {s}"
    except CalledProcessError as e:
        return f"[red]Error: {e.output.decode()}[/]"


def handleGitPush():
    try:
        s = check_output("git push origin main", shell=True).decode()
        return f"[green]Pushed To Remote Repo,[/green] {s}"
    except CalledProcessError as e:
        return f"[red]Error: {e.output.decode()}[/]"
