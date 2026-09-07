import re


def normalize_text(text: str | None) -> str:
    if text is None:
        return ""

    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)

    return text


def normalize_free_answer(text: str | None) -> str:
    text = normalize_text(text)

    # Remove harmless surrounding punctuation.
    text = text.strip(".,:;!?")

    return text


def canonicalize_mcq(
    output: str,
    options: list[str],
) -> str | None:

    normalized = normalize_text(output)

    # Direct option-letter answer.
    match = re.fullmatch(
        r"(?:option\s*)?([a-z])[\.\):]?",
        normalized,
    )

    if match:
        letter = match.group(1)
        index = ord(letter) - ord("a")

        if 0 <= index < len(options):
            return normalize_free_answer(options[index])

    # Look for an option's text in the answer.
    for option in options:
        option_norm = normalize_free_answer(option)

        if option_norm and option_norm in normalized:
            return option_norm

    return normalize_free_answer(output)


def canonicalize_output(
    output: str | None,
    task_type: str,
    options: list[str] | None = None,
) -> str:

    if not output:
        return ""

    if task_type == "mcq" and options:
        result = canonicalize_mcq(output, options)

        if result is not None:
            return result

    return normalize_free_answer(output)
