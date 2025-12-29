# refrag/core/decoder.py
"""
REFRAG Mixed Context Decoder: Builds context with RAW + COMPRESSED chunks

This creates the final context to send to the LLM.
"""

from typing import List, Dict, Any


class MixedContextDecoder:
    """
    REFRAG Context Builder: Combines RAW and COMPRESSED chunks.

    Builds a mixed-format context that:
    - Preserves full detail for high-priority chunks [RAW]
    - Saves space with keywords for low-priority chunks [COMPRESSED]

    This enables fitting more relevant information in the context window.
    """

    def __init__(
        self,
        format_style: str = "tagged"  # "tagged", "separated", or "inline"
    ):
        """
        Initialize decoder.

        Args:
            format_style: How to format mixed context
                - "tagged": Use [RAW]...[/RAW] and [COMPRESSED]...[/COMPRESSED]
                - "separated": Separate sections for RAW and COMPRESSED
                - "inline": Mix them inline with markers
        """
        self.format_style = format_style

    def build_context(
        self,
        chunks: List[str],
        compressed_chunks: List[str],
        is_raw: List[bool],
        include_metadata: bool = False,
        scores: List[float] = None
    ) -> str:
        """
        Build mixed RAW + COMPRESSED context.

        Args:
            chunks: Original chunk texts
            compressed_chunks: Compressed versions
            is_raw: Boolean list (True = use RAW, False = use COMPRESSED)
            include_metadata: Include similarity scores as metadata
            scores: Similarity scores (optional, for metadata)

        Returns:
            Formatted context string ready for LLM
        """
        if self.format_style == "tagged":
            return self._build_tagged_context(
                chunks, compressed_chunks, is_raw, include_metadata, scores
            )
        elif self.format_style == "separated":
            return self._build_separated_context(
                chunks, compressed_chunks, is_raw, include_metadata, scores
            )
        else:  # inline
            return self._build_inline_context(
                chunks, compressed_chunks, is_raw, include_metadata, scores
            )

    def _build_tagged_context(
        self,
        chunks: List[str],
        compressed_chunks: List[str],
        is_raw: List[bool],
        include_metadata: bool,
        scores: List[float]
    ) -> str:
        """
        Build context with [RAW] and [COMPRESSED] tags.

        Example:
        [RAW]Python is a programming language created by Guido.[/RAW]
        [COMPRESSED]machine learning AI data[/COMPRESSED]
        [RAW]JavaScript runs in browsers.[/RAW]
        """
        context_parts = []

        for i, (chunk, compressed, raw) in enumerate(zip(chunks, compressed_chunks, is_raw)):
            if raw:
                # Use full chunk
                text = f"[RAW]{chunk}[/RAW]"
            else:
                # Use compressed version
                text = f"[COMPRESSED]{compressed}[/COMPRESSED]"

            # Add metadata if requested
            if include_metadata and scores:
                text = f"{text} (score: {scores[i]:.3f})"

            context_parts.append(text)

        return "\n".join(context_parts)

    def _build_separated_context(
        self,
        chunks: List[str],
        compressed_chunks: List[str],
        is_raw: List[bool],
        include_metadata: bool,
        scores: List[float]
    ) -> str:
        """
        Build context with separate RAW and COMPRESSED sections.

        Example:
        === HIGH PRIORITY CONTEXT ===
        - Python is a programming language created by Guido.
        - JavaScript runs in browsers.

        === SUPPORTING KEYWORDS ===
        - machine learning AI data
        - neural networks training
        """
        raw_parts = []
        compressed_parts = []

        for i, (chunk, compressed, raw) in enumerate(zip(chunks, compressed_chunks, is_raw)):
            if raw:
                raw_parts.append(f"- {chunk}")
            else:
                compressed_parts.append(f"- {compressed}")

        context = ""
        if raw_parts:
            context += "=== HIGH PRIORITY CONTEXT ===\n"
            context += "\n".join(raw_parts)

        if compressed_parts:
            if raw_parts:
                context += "\n\n"
            context += "=== SUPPORTING KEYWORDS ===\n"
            context += "\n".join(compressed_parts)

        return context

    def _build_inline_context(
        self,
        chunks: List[str],
        compressed_chunks: List[str],
        is_raw: List[bool],
        include_metadata: bool,
        scores: List[float]
    ) -> str:
        """
        Build inline context with minimal markers.

        Example:
        [FULL] Python is a programming language created by Guido.
        [KW] machine learning AI data
        [FULL] JavaScript runs in browsers.
        """
        context_parts = []

        for i, (chunk, compressed, raw) in enumerate(zip(chunks, compressed_chunks, is_raw)):
            if raw:
                prefix = "[FULL]"
                text = chunk
            else:
                prefix = "[KW]"
                text = compressed

            context_parts.append(f"{prefix} {text}")

        return "\n".join(context_parts)

    def build_simple_context(
        self,
        chunks: List[str],
        compressed_chunks: List[str],
        is_raw: List[bool]
    ) -> str:
        """
        Build simple context without any special formatting.
        Just concatenate RAW chunks fully and COMPRESSED as keywords.

        Args:
            chunks: Original chunks
            compressed_chunks: Compressed versions
            is_raw: RAW flags

        Returns:
            Simple concatenated context
        """
        parts = []
        for chunk, compressed, raw in zip(chunks, compressed_chunks, is_raw):
            if raw:
                parts.append(chunk)
            else:
                parts.append(compressed)

        return " ".join(parts)
