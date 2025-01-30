"""
modified from LTG-BERT --- https://github.com/ltgoslo/ltg-bert
"""

# ftfy fixes Unicode that’s broken in various ways.
import ftfy
from sacremoses import MosesPunctNormalizer

mpn = MosesPunctNormalizer(lang="en", penn=False)


def clean_text(text, minimal=False):
    if not minimal:
        text = add_whitespace(text)
        text = normalize_abbreviations(text)
        text = fix_double_quotes(text)
        text = mpn.normalize(text)

    text = ftfy.fix_text(text)
    text = text.strip()

    return text


def fix_double_quotes(text):
    # try to fix, but be gentle and try not to cause more harm
    n_quotes = text.count('"')
    if n_quotes == 0 or (n_quotes % 2) == 1 or '""' in text or '" "' in text:
        return text

    original_text = text

    i, i_quote, n_changes = 0, 0, 0
    while i < len(text):
        if text[i] != '"':
            i += 1
            continue

        if (i_quote % 2) == 0:
            if i > 0 and text[i - 1] != " ":
                text = text[:i] + " " + text[i:]
                i += 1
                n_changes += 1
            if i + 1 < len(text) and text[i + 1] == " ":
                text = text[: i + 1] + text[i + 2 :]
                n_changes += 1
        else:
            if i > 0 and text[i - 1] == " ":
                text = text[: i - 1] + text[i:]
                i -= 1
                n_changes += 1
            if i + 1 < len(text) and text[i + 1].isalnum():
                text = text[: i + 1] + " " + text[i + 1 :]
                n_changes += 1

        i_quote += 1
        i += 1

    # too much changes, let's return the original text to play it safe
    if n_changes > 2 and n_changes > n_quotes * 2 / 3:
        return original_text

    return text


def normalize_abbreviations(text):
    # Remove space from abbreviations

    text = text.replace(" n't ", "n't ")
    text = text.replace(" N'T ", "N'T ")
    text = text.replace(" 'll ", "'ll ")
    text = text.replace(" 'LL ", "'LL ")
    text = text.replace(" 're ", "'re ")
    text = text.replace(" 'RE ", "'RE ")
    text = text.replace(" 've ", "'ve ")
    text = text.replace(" 'VE ", "'VE ")
    text = text.replace(" 'm ", "'m ")
    text = text.replace(" 'M ", "'M ")
    text = text.replace(" 's ", "'s ")
    text = text.replace(" 'S ", "'S ")
    text = text.replace(" 'd ", "'d ")
    text = text.replace(" 'D ", "'D ")

    text = text.replace(" n't,", "n't,")
    text = text.replace(" N'T,", "N'T,")
    text = text.replace(" 'll,", "'ll,")
    text = text.replace(" 'LL,", "'LL,")
    text = text.replace(" 're,", "'re,")
    text = text.replace(" 'RE,", "'RE,")
    text = text.replace(" 've,", "'ve,")
    text = text.replace(" 'VE,", "'VE,")
    text = text.replace(" 'm,", "'m,")
    text = text.replace(" 'M,", "'M,")
    text = text.replace(" 's,", "'s,")
    text = text.replace(" 'S,", "'S,")
    text = text.replace(" 'd,", "'d,")
    text = text.replace(" 'D,", "'D,")

    text = text.replace(" n't.", "n't.")
    text = text.replace(" N'T.", "N'T.")
    text = text.replace(" 'll.", "'ll.")
    text = text.replace(" 'LL.", "'LL.")
    text = text.replace(" 're.", "'re.")
    text = text.replace(" 'RE.", "'RE.")
    text = text.replace(" 've.", "'ve.")
    text = text.replace(" 'VE.", "'VE.")
    text = text.replace(" 'm.", "'m.")
    text = text.replace(" 'M.", "'M.")
    text = text.replace(" 's.", "'s.")
    text = text.replace(" 'S.", "'S.")
    text = text.replace(" 'd.", "'d.")
    text = text.replace(" 'D.", "'D.")
    return text


def add_whitespace(text):
    # remove excess whitespace
    text = " ".join(text.replace("\n", "<<NEWLINE/>>").split()).replace(
        "<<NEWLINE/>>", "\n"
    )

    # iterate from str[-2] to str[0]
    for i in range(len(text) - 2, -1, -1):
        # separate period
        if text[i] == "." and (
            text[i + 1].isupper() or text[i + 1] in ["‘", "(", "[", "{"]
        ):
            text = text[: i + 1] + " " + text[i + 1 :]

        # separate other punctuation marks
        elif text[i] in ["?", "!", "…", "’"] and (
            text[i + 1].isalnum() or text[i + 1] in ["‘", "(", "[", "{"]
        ):
            text = text[: i + 1] + " " + text[i + 1 :]

        # separate '...'
        elif (
            i > 2
            and text[i] == "."
            and text[i - 1] == "."
            and text[i - 2] == "."
            and text[i + 1] != " "
        ):
            text = text[: i + 1] + " " + text[i + 1 :]

        # add the same rule a second time?
        elif (
            i > 2
            and text[i] == "."
            and text[i - 1] == "."
            and text[i - 2] == "."
            and text[i + 1] != " "
        ):
            text = text[: i + 1] + " " + text[i + 1 :]

        # seperate comma
        elif text[i] == "," and (
            text[i + 1].isalpha() or text[i + 1] in ["‘", "(", "[", "{"]
        ):
            text = text[: i + 1] + " " + text[i + 1 :]

        # seperate other special case symbols
        elif text[i] in [";", ")", "]", "}", "%"] and (
            text[i + 1].isalnum() or text[i + 1] in ["‘", "(", "[", "{"]
        ):
            text = text[: i + 1] + " " + text[i + 1 :]

        # handling some special case with ":"
        elif text[i] == ":" and (
            text[i + 1] in ["‘", "(", "[", "{"]
            or (
                text[i + 1].isalnum()
                and (
                    not text[i + 1].isnumeric()
                    or i - 1 < 0
                    or not text[i - 1].isnumeric()
                )
            )
        ):
            text = text[: i + 1] + " " + text[i + 1 :]

        # remove whitespace from parentheses
        elif text[i] in ["(", "[", "{"] and text[i + 1] == " ":
            text = text[: i + 1] + text[i + 2 :]

        # remove whitespace from ":"
        elif text[i] == " " and text[i + 1] in [
            ".",
            ";",
            ":",
            "?",
            "!",
            "…",
            ",",
            "’",
            ")",
            "]",
            "}",
        ]:
            text = text[:i] + text[i + 1 :]

        # remove whitespace from currency (although EU should be different)
        elif (
            i > 0
            and text[i] == " "
            and text[i - 1] in ["$", "£", "€"]
            and text[i + 1].isnumeric()
        ):
            text = text[:i] + text[i + 1 :]

        # remove whitespace from percentage
        elif (
            i > 0 and text[i] == " " and text[i - 1].isnumeric() and text[i + 1] == "%"
        ):
            text = text[:i] + text[i + 1 :]

    return text
