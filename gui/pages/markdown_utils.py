"""
Markdown to HTML conversion utilities for the GUI
"""

import re


def markdown_to_html(text: str) -> str:
    """
    Convert markdown text to styled HTML for display in the GUI.

    Handles:
    - Headers (# ## ###)
    - Bold (**text**)
    - Italic (*text* or _text_)
    - Bullet lists (- item or * item)
    - Numbered lists (1. item)
    - Code blocks (```code```)
    - Inline code (`code`)
    - Paragraphs
    """
    if not text:
        return ""

    lines = text.split('\n')
    html_parts = []
    in_code_block = False
    code_block_content = []
    in_list = False
    list_type = None  # 'ul' or 'ol'

    def process_inline(line: str) -> str:
        """Process inline markdown (bold, italic, code)"""
        # Escape HTML special characters first (but preserve our tags)
        line = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')

        # Inline code `code` -> <code>code</code>
        line = re.sub(r'`([^`]+)`', r'<code style="background-color: #f0f0f0; padding: 2px 6px; border-radius: 4px; font-family: monospace; font-size: 0.9em;">\1</code>', line)

        # Bold **text** or __text__
        line = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', line)
        line = re.sub(r'__([^_]+)__', r'<strong>\1</strong>', line)

        # Italic *text* or _text_ (but not inside words)
        line = re.sub(r'(?<!\w)\*([^*]+)\*(?!\w)', r'<em>\1</em>', line)
        line = re.sub(r'(?<!\w)_([^_]+)_(?!\w)', r'<em>\1</em>', line)

        return line

    def close_list():
        """Close any open list"""
        nonlocal in_list, list_type
        if in_list:
            html_parts.append(f'</{list_type}>')
            in_list = False
            list_type = None

    i = 0
    while i < len(lines):
        line = lines[i]

        # Handle code blocks
        if line.strip().startswith('```'):
            if in_code_block:
                # End code block
                code_content = '\n'.join(code_block_content)
                code_content = code_content.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                html_parts.append(f'<pre style="background-color: #f5f5f5; padding: 12px; border-radius: 8px; overflow-x: auto; font-family: monospace; font-size: 0.9em; margin: 8px 0;"><code>{code_content}</code></pre>')
                code_block_content = []
                in_code_block = False
            else:
                # Start code block
                close_list()
                in_code_block = True
            i += 1
            continue

        if in_code_block:
            code_block_content.append(line)
            i += 1
            continue

        stripped = line.strip()

        # Empty line
        if not stripped:
            close_list()
            i += 1
            continue

        # Headers
        if stripped.startswith('### '):
            close_list()
            header_text = process_inline(stripped[4:])
            html_parts.append(f'<h4 style="font-size: 1.1em; font-weight: 600; margin: 16px 0 8px 0; color: #1d1d1f;">{header_text}</h4>')
            i += 1
            continue

        if stripped.startswith('## '):
            close_list()
            header_text = process_inline(stripped[3:])
            html_parts.append(f'<h3 style="font-size: 1.25em; font-weight: 600; margin: 20px 0 10px 0; color: #1d1d1f;">{header_text}</h3>')
            i += 1
            continue

        if stripped.startswith('# '):
            close_list()
            header_text = process_inline(stripped[2:])
            html_parts.append(f'<h2 style="font-size: 1.4em; font-weight: 600; margin: 24px 0 12px 0; color: #1d1d1f;">{header_text}</h2>')
            i += 1
            continue

        # Bullet lists (- or *)
        bullet_match = re.match(r'^[\-\*]\s+(.+)$', stripped)
        if bullet_match:
            if not in_list or list_type != 'ul':
                close_list()
                html_parts.append('<ul style="margin: 8px 0; padding-left: 24px;">')
                in_list = True
                list_type = 'ul'
            item_text = process_inline(bullet_match.group(1))
            html_parts.append(f'<li style="margin: 4px 0;">{item_text}</li>')
            i += 1
            continue

        # Numbered lists
        numbered_match = re.match(r'^(\d+)[\.\)]\s+(.+)$', stripped)
        if numbered_match:
            if not in_list or list_type != 'ol':
                close_list()
                html_parts.append('<ol style="margin: 8px 0; padding-left: 24px;">')
                in_list = True
                list_type = 'ol'
            item_text = process_inline(numbered_match.group(2))
            html_parts.append(f'<li style="margin: 4px 0;">{item_text}</li>')
            i += 1
            continue

        # Regular paragraph
        close_list()
        para_text = process_inline(stripped)
        html_parts.append(f'<p style="margin: 8px 0; line-height: 1.6;">{para_text}</p>')
        i += 1

    # Close any remaining open elements
    close_list()
    if in_code_block and code_block_content:
        code_content = '\n'.join(code_block_content)
        code_content = code_content.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        html_parts.append(f'<pre style="background-color: #f5f5f5; padding: 12px; border-radius: 8px; overflow-x: auto; font-family: monospace; font-size: 0.9em; margin: 8px 0;"><code>{code_content}</code></pre>')

    return ''.join(html_parts)


def generate_diff_html(text1: str, text2: str, side: str, similarity_threshold: float = 0.3) -> str:
    """
    Generate HTML with diff highlighting for substantially different sentences.
    Also converts markdown to proper HTML formatting.

    Args:
        text1: First text
        text2: Second text
        side: 'left' or 'right' - which side to highlight
        similarity_threshold: Only highlight sentences with similarity below this threshold (0-1)

    Returns:
        HTML string with highlighted differences and formatted markdown
    """
    import difflib

    # First convert both texts to HTML
    html1 = markdown_to_html(text1)
    html2 = markdown_to_html(text2)

    # For diff highlighting, we work with sentences from the original text
    # Split into sentences for comparison
    def split_sentences(text):
        # Split on sentence boundaries but keep the structure
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]

    sentences1 = split_sentences(text1)
    sentences2 = split_sentences(text2)

    # Use difflib to find differences
    matcher = difflib.SequenceMatcher(None, sentences1, sentences2)

    # Find which sentences are substantially different
    highlight_sentences = set()

    if side == 'left':
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'replace':
                for i in range(i1, i2):
                    best_ratio = 0
                    for j in range(j1, j2):
                        ratio = difflib.SequenceMatcher(None, sentences1[i], sentences2[j]).ratio()
                        best_ratio = max(best_ratio, ratio)
                    if best_ratio < similarity_threshold:
                        highlight_sentences.add(sentences1[i])
            elif tag == 'delete':
                for i in range(i1, i2):
                    highlight_sentences.add(sentences1[i])

        # Now apply highlighting to the HTML
        result_html = html1
        for sentence in highlight_sentences:
            # Escape the sentence for regex
            escaped = re.escape(sentence.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'))
            # Find and wrap with highlight span
            result_html = re.sub(
                f'({escaped})',
                r'<span style="background-color: #64b5f6; padding: 2px;">\1</span>',
                result_html,
                count=1
            )
        return result_html

    else:  # side == 'right'
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'replace':
                for j in range(j1, j2):
                    best_ratio = 0
                    for i in range(i1, i2):
                        ratio = difflib.SequenceMatcher(None, sentences1[i], sentences2[j]).ratio()
                        best_ratio = max(best_ratio, ratio)
                    if best_ratio < similarity_threshold:
                        highlight_sentences.add(sentences2[j])
            elif tag == 'insert':
                for j in range(j1, j2):
                    highlight_sentences.add(sentences2[j])

        # Now apply highlighting to the HTML
        result_html = html2
        for sentence in highlight_sentences:
            # Escape the sentence for regex
            escaped = re.escape(sentence.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'))
            # Find and wrap with highlight span
            result_html = re.sub(
                f'({escaped})',
                r'<span style="background-color: #64b5f6; padding: 2px;">\1</span>',
                result_html,
                count=1
            )
        return result_html


def apply_llm_highlights(text: str, phrases_to_highlight: list) -> str:
    """
    Apply LLM-specified highlights to text.

    This function takes the original text and a list of exact phrases identified
    by the LLM Judge as semantically significant differences, then highlights them.

    Args:
        text: The original answer text
        phrases_to_highlight: List of exact text phrases to highlight

    Returns:
        HTML string with highlighted phrases and markdown formatting applied
    """
    if not text:
        return ""

    if not phrases_to_highlight:
        # No phrases to highlight, just convert markdown to HTML
        return markdown_to_html(text)

    # First convert markdown to HTML
    html = markdown_to_html(text)

    # Apply highlights for each phrase
    for phrase in phrases_to_highlight:
        if not phrase or len(phrase) < 3:  # Skip very short phrases
            continue

        # Escape the phrase for regex matching
        # The phrase needs to be HTML-escaped since we're searching in HTML
        escaped_phrase = phrase.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        escaped_for_regex = re.escape(escaped_phrase)

        # Try to find and highlight the phrase (case-insensitive for flexibility)
        pattern = f'({escaped_for_regex})'

        html = re.sub(
            pattern,
            r'<span style="background-color: #64b5f6; padding: 2px; border-radius: 2px;" title="Key difference identified by LLM Judge">\1</span>',
            html,
            count=1,  # Only highlight first occurrence to avoid over-highlighting
            flags=re.IGNORECASE
        )

    return html
