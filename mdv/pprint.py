import subprocess


def figlet(text: str) -> str:
    """
    Generate a stylized ASCII art representation of the given text.

    This function attempts to use the `figlet` command-line tool to create an ASCII art version of the input text. 
    If the tool is not available, it falls back to a simple text representation with borders.

    Args:
        text (str): The text to be converted into ASCII art.

    Returns:
        str: The ASCII art representation of the text, or a fallback representation if figlet is unavailable.
    """

    def fallback(text: str) -> str:
        n_chars = min(60, len(text) + 4)
        borders = "+" + "-" * (n_chars-2) + "+"
        text = f"|{text.center(n_chars-2)}|"
        out = borders + "\n" + text + "\n" + borders
        return out

    try:
        ret = subprocess.check_output(["figlet", text])
        return ret.decode("utf-8").rstrip()
    except subprocess.CalledProcessError as e:
        return fallback(text)


def heading(text: str, *args):
    """
    Generate and print a stylized ASCII art heading with optional additional text.

    This function creates an ASCII art representation of the provided text using the `figlet` function. 
    It prints each line of the ASCII art, appending any additional arguments to the corresponding lines.

    Args:
        text (str): The main text to be converted into ASCII art.
        *args: Additional text to be appended to each line of the ASCII art.

    Returns:
        None

    Example:
        heading("Hello World!", "This is", "a test")
        >>> _   _      _ _         __        __         _     _ _
        >>> | | | | ___| | | ___    \ \      / /__  _ __| | __| | |
        >>> | |_| |/ _ \ | |/ _ \    \ \ /\ / / _ \| '__| |/ _` | |
        >>> |  _  |  __/ | | (_) |    \ V  V / (_) | |  | | (_| |_| This is
        >>> |_| |_|\___|_|_|\___/      \_/\_/ \___/|_|  |_|\__,_(_) a test         
    """
    head_text = figlet(text)
    head_split = head_text.split("\n")
    max_chars = max(head_split, key=len)
    n_lines = len(head_split)
    n_delta = n_lines - len(args)
    for i in range(n_lines):
        line = head_split[i].rstrip().ljust(len(max_chars))
        if i >= n_delta:
            line += " " + args[i - n_delta]
        print(line)
