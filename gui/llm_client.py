"""
LLM Client Module
Handles API calls to OpenAI and Anthropic for paraphrase and answer generation
"""

from typing import List, Dict, Optional
import asyncio
from enum import Enum


class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


class LLMModel(Enum):
    """Supported LLM models"""
    # OpenAI models (2025)
    GPT5 = "gpt-5"
    GPT5_CODEX = "gpt-5-codex"
    GPT4_5 = "gpt-4.5"
    GPT4O = "gpt-4o"
    GPT4O_MINI = "gpt-4o-mini"
    O1_PREVIEW = "o1-preview"
    O1_MINI = "o1-mini"

    # Anthropic models (2025)
    CLAUDE_SONNET_4_5 = "claude-sonnet-4-5-20250929"
    CLAUDE_OPUS_4_1 = "claude-opus-4-1"
    CLAUDE_SONNET_4 = "claude-sonnet-4"
    CLAUDE_HAIKU_4_5 = "claude-haiku-4-5"
    CLAUDE_SONNET_3_5 = "claude-3-5-sonnet-20241022"

    # Google Gemini models (2025)
    GEMINI_3_PRO_IMAGE = "gemini-3-pro-image"
    GEMINI_2_5_PRO = "gemini-2.5-pro"
    GEMINI_2_5_FLASH = "gemini-2.5-flash"
    GEMINI_2_5_FLASH_LITE = "gemini-2.5-flash-lite"
    GEMINI_2_0_FLASH_EXP = "gemini-2.0-flash-exp"


class LLMClient:
    """Unified client for LLM API calls"""

    def __init__(self, provider: LLMProvider, model: LLMModel, api_key: str):
        """
        Initialize LLM client

        Args:
            provider: LLM provider (OpenAI or Anthropic)
            model: Model to use
            api_key: API key for the provider
        """
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self._client = None

    def _get_client(self):
        """Lazy initialization of API client"""
        if self._client is None:
            if self.provider == LLMProvider.OPENAI:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(api_key=self.api_key)
            elif self.provider == LLMProvider.ANTHROPIC:
                from anthropic import AsyncAnthropic
                self._client = AsyncAnthropic(api_key=self.api_key)
            elif self.provider == LLMProvider.GEMINI:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._client = genai.GenerativeModel(self.model.value)
        return self._client

    async def generate_paraphrases(
        self,
        original_prompt: str,
        context: str,
        num_paraphrases: int
    ) -> List[str]:
        """
        Generate paraphrases of the original prompt

        Args:
            original_prompt: The original question/prompt
            context: Context information
            num_paraphrases: Number of paraphrases to generate

        Returns:
            List of paraphrased prompts (including the original)
        """
        system_prompt = f"""You are a helpful assistant that generates paraphrases of questions while preserving their meaning.

Given a question and context, generate {num_paraphrases} different ways to ask the same question. Each paraphrase should:
- Preserve the exact meaning and intent of the original question
- Use different wording and sentence structure
- Be natural and grammatically correct
- Request the same information in different ways

Return ONLY the paraphrased questions, one per line, numbered 1-{num_paraphrases}."""

        user_prompt = f"""Context: {context}

Original Question: {original_prompt}

Generate {num_paraphrases} paraphrases:"""

        client = self._get_client()

        if self.provider == LLMProvider.OPENAI:
            response = await client.chat.completions.create(
                model=self.model.value,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.8,
                max_tokens=1000
            )
            content = response.choices[0].message.content

        elif self.provider == LLMProvider.ANTHROPIC:
            response = await client.messages.create(
                model=self.model.value,
                max_tokens=1000,
                temperature=0.8,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            content = response.content[0].text

        elif self.provider == LLMProvider.GEMINI:
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            response = await client.generate_content_async(
                full_prompt,
                generation_config={'temperature': 0.8, 'max_output_tokens': 1000}
            )
            content = response.text

        # Parse the numbered paraphrases
        paraphrases = [original_prompt]  # Include original as first item
        for line in content.strip().split('\n'):
            line = line.strip()
            # Remove numbering like "1.", "1)", etc.
            if line and (line[0].isdigit() or line.startswith('-')):
                # Find the first non-digit, non-punctuation character
                i = 0
                while i < len(line) and (line[i].isdigit() or line[i] in '.-) '):
                    i += 1
                paraphrase = line[i:].strip()
                if paraphrase:
                    paraphrases.append(paraphrase)

        return paraphrases[:num_paraphrases + 1]  # Original + N paraphrases

    async def generate_answer(
        self,
        question: str,
        context: str,
        temperature: float = 0.7
    ) -> str:
        """
        Generate an answer to a question given context

        Args:
            question: The question to answer
            context: Context information
            temperature: Sampling temperature (0-1)

        Returns:
            Generated answer
        """
        system_prompt = """You are a helpful AI assistant. Answer the question based on the provided context. Be comprehensive, accurate, and well-structured in your response."""

        user_prompt = f"""Context: {context}

Question: {question}

Please provide a detailed answer based on the context above."""

        client = self._get_client()

        if self.provider == LLMProvider.OPENAI:
            response = await client.chat.completions.create(
                model=self.model.value,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=2000
            )
            return response.choices[0].message.content.strip()

        elif self.provider == LLMProvider.ANTHROPIC:
            response = await client.messages.create(
                model=self.model.value,
                max_tokens=2000,
                temperature=temperature,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            return response.content[0].text.strip()

        elif self.provider == LLMProvider.GEMINI:
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            response = await client.generate_content_async(
                full_prompt,
                generation_config={'temperature': temperature, 'max_output_tokens': 2000}
            )
            return response.text.strip()

    async def generate_all_answers(
        self,
        questions: List[str],
        context: str,
        temperature: float = 0.7,
        progress_callback: Optional[callable] = None
    ) -> List[str]:
        """
        Generate answers for multiple questions

        Args:
            questions: List of questions
            context: Context information
            temperature: Sampling temperature
            progress_callback: Optional callback function(current, total)

        Returns:
            List of generated answers
        """
        answers = []
        total = len(questions)

        for i, question in enumerate(questions):
            answer = await self.generate_answer(question, context, temperature)
            answers.append(answer)

            if progress_callback:
                await progress_callback(i + 1, total)

        return answers

    async def judge_answers(
        self,
        question: str,
        context: str,
        answer_a: str,
        answer_b: str,
        answer_a_label: str = "Answer A",
        answer_b_label: str = "Answer B"
    ) -> Dict:
        """
        Use LLM-as-a-Judge to compare two answers and determine which is better.

        Based on the methodology from Zheng et al. (2023) "Judging LLM-as-a-Judge
        with MT-Bench and Chatbot Arena."

        Args:
            question: The original question
            context: The context used to generate the answers
            answer_a: First answer to compare
            answer_b: Second answer to compare
            answer_a_label: Label for first answer (e.g., "Initial Answer")
            answer_b_label: Label for second answer (e.g., "Best F_S Answer")

        Returns:
            Dict containing:
                - winner: 'A', 'B', or 'TIE'
                - explanation: Detailed reasoning
                - scores: Dict with scores for each answer (1-10)
                - criteria_breakdown: Dict with scores per criterion
        """
        system_prompt = """You are an expert evaluator assessing the quality of answers to questions based on provided context.

Your task is to compare two answers and determine which one is better. Evaluate based on these criteria:

1. **Faithfulness to Context**: Does the answer accurately reflect information from the context without hallucinations?
2. **Completeness**: Does the answer address all aspects of the question?
3. **Coherence**: Is the answer well-organized and logically structured?
4. **Relevance**: Does the answer focus on what was asked without unnecessary tangents?

IMPORTANT: Base your evaluation ONLY on how well each answer represents the information in the context. Do not prefer answers simply because they are longer or more detailed if that additional detail is not supported by the context.

Respond in the following JSON format ONLY (no other text):
{
    "winner": "A" or "B" or "TIE",
    "scores": {
        "A": <1-10>,
        "B": <1-10>
    },
    "criteria_breakdown": {
        "faithfulness": {"A": <1-10>, "B": <1-10>},
        "completeness": {"A": <1-10>, "B": <1-10>},
        "coherence": {"A": <1-10>, "B": <1-10>},
        "relevance": {"A": <1-10>, "B": <1-10>}
    },
    "explanation": "<detailed explanation of your judgment>"
}"""

        # Truncate context if too long
        max_context_chars = 8000
        truncated_context = context[:max_context_chars]
        if len(context) > max_context_chars:
            truncated_context += "\n\n[Context truncated for evaluation...]"

        user_prompt = f"""## Question
{question}

## Context
{truncated_context}

## {answer_a_label} (Answer A)
{answer_a}

## {answer_b_label} (Answer B)
{answer_b}

Please evaluate which answer better represents the information from the context."""

        client = self._get_client()

        try:
            if self.provider == LLMProvider.OPENAI:
                response = await client.chat.completions.create(
                    model=self.model.value,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,  # Low temperature for consistent evaluation
                    max_tokens=1500
                )
                content = response.choices[0].message.content

            elif self.provider == LLMProvider.ANTHROPIC:
                response = await client.messages.create(
                    model=self.model.value,
                    max_tokens=1500,
                    temperature=0.1,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": user_prompt}
                    ]
                )
                content = response.content[0].text

            elif self.provider == LLMProvider.GEMINI:
                full_prompt = f"{system_prompt}\n\n{user_prompt}"
                response = await client.generate_content_async(
                    full_prompt,
                    generation_config={'temperature': 0.1, 'max_output_tokens': 1500}
                )
                content = response.text

            # Parse JSON response
            import json
            import re

            # Try to extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                result = json.loads(json_match.group())
                return {
                    'winner': result.get('winner', 'TIE'),
                    'explanation': result.get('explanation', 'No explanation provided'),
                    'scores': result.get('scores', {'A': 5, 'B': 5}),
                    'criteria_breakdown': result.get('criteria_breakdown', {})
                }
            else:
                # Fallback if JSON parsing fails
                return {
                    'winner': 'TIE',
                    'explanation': f'Could not parse evaluation. Raw response: {content[:500]}',
                    'scores': {'A': 5, 'B': 5},
                    'criteria_breakdown': {}
                }

        except Exception as e:
            return {
                'winner': 'ERROR',
                'explanation': f'Evaluation failed: {str(e)}',
                'scores': {'A': 0, 'B': 0},
                'criteria_breakdown': {}
            }


def get_available_models(provider: LLMProvider) -> List[LLMModel]:
    """Get list of available models for a provider"""
    if provider == LLMProvider.OPENAI:
        return [
            LLMModel.GPT5, LLMModel.GPT5_CODEX, LLMModel.GPT4_5,
            LLMModel.GPT4O, LLMModel.GPT4O_MINI,
            LLMModel.O1_PREVIEW, LLMModel.O1_MINI
        ]
    elif provider == LLMProvider.ANTHROPIC:
        return [
            LLMModel.CLAUDE_SONNET_4_5, LLMModel.CLAUDE_OPUS_4_1,
            LLMModel.CLAUDE_SONNET_4, LLMModel.CLAUDE_HAIKU_4_5,
            LLMModel.CLAUDE_SONNET_3_5
        ]
    elif provider == LLMProvider.GEMINI:
        return [
            LLMModel.GEMINI_3_PRO_IMAGE, LLMModel.GEMINI_2_5_PRO,
            LLMModel.GEMINI_2_5_FLASH, LLMModel.GEMINI_2_5_FLASH_LITE,
            LLMModel.GEMINI_2_0_FLASH_EXP
        ]
    return []
