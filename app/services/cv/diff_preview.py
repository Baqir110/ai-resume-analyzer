import difflib
import html
from typing import Any


class DiffPreviewService:
    @staticmethod
    def generate_word_diff(original_text: str, optimized_text: str) -> dict[str, Any]:
        """
        Generates word-level diffs with HTML/Markdown formatting and change statistics.
        """
        if not isinstance(original_text, str) or not isinstance(optimized_text, str):
            raise ValueError("Diff inputs must be strings")
        if len(original_text) > 100_000 or len(optimized_text) > 100_000:
            raise ValueError("Diff input is too large")
        orig_words = original_text.split()
        opt_words = optimized_text.split()

        matcher = difflib.SequenceMatcher(None, orig_words, opt_words)

        diff_html_tokens = []
        additions = 0
        deletions = 0

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                diff_html_tokens.append(html.escape(" ".join(orig_words[i1:i2])))
            elif tag == "replace":
                deleted = html.escape(" ".join(orig_words[i1:i2]))
                added = html.escape(" ".join(opt_words[j1:j2]))
                diff_html_tokens.append(
                    f'<del style="color: #d9534f; background-color: #fdf7f7; text-decoration: line-through;">{deleted}</del>'
                )
                diff_html_tokens.append(
                    f'<ins style="color: #5cb85c; background-color: #f0fff0; text-decoration: none; font-weight: bold;">{added}</ins>'
                )
                deletions += i2 - i1
                additions += j2 - j1
            elif tag == "delete":
                deleted = html.escape(" ".join(orig_words[i1:i2]))
                diff_html_tokens.append(
                    f'<del style="color: #d9534f; background-color: #fdf7f7; text-decoration: line-through;">{deleted}</del>'
                )
                deletions += i2 - i1
            elif tag == "insert":
                added = html.escape(" ".join(opt_words[j1:j2]))
                diff_html_tokens.append(
                    f'<ins style="color: #5cb85c; background-color: #f0fff0; text-decoration: none; font-weight: bold;">{added}</ins>'
                )
                additions += j2 - j1

        formatted_diff = " ".join(diff_html_tokens)
        similarity_ratio = round(matcher.ratio() * 100, 2)

        return {
            "original": original_text,
            "optimized": optimized_text,
            "diff_html": formatted_diff,
            "similarity_ratio": similarity_ratio,
            "words_added": additions,
            "words_removed": deletions,
        }

    @classmethod
    def compare_bullet_lists(
        cls, original_bullets: list[str], optimized_bullets: list[str]
    ) -> list[dict[str, Any]]:
        """
        Pairs and compares lists of bullet points.
        """
        if len(original_bullets) > 500 or len(optimized_bullets) > 500:
            raise ValueError("Too many bullets")
        results = []
        max_len = max(len(original_bullets), len(optimized_bullets))

        for i in range(max_len):
            orig = original_bullets[i] if i < len(original_bullets) else ""
            opt = optimized_bullets[i] if i < len(optimized_bullets) else ""
            results.append(cls.generate_word_diff(orig, opt))

        return results
