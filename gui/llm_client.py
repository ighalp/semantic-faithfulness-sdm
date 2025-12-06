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
    CLAUDE_SONNET_4 = "claude-sonnet-4-20250514"
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
                temperature=0.8
            )
            content = response.choices[0].message.content

        elif self.provider == LLMProvider.ANTHROPIC:
            response = await client.messages.create(
                model=self.model.value,
                max_tokens=8192,
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
                generation_config={'temperature': 0.8, 'max_output_tokens': 8192}
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
                temperature=temperature
            )
            return response.choices[0].message.content.strip()

        elif self.provider == LLMProvider.ANTHROPIC:
            response = await client.messages.create(
                model=self.model.value,
                max_tokens=8192,
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
                generation_config={'temperature': temperature, 'max_output_tokens': 8192}
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
        answer_b_label: str = "Answer B",
        custom_system_prompt: str = None
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
        default_system_prompt = """You are an expert evaluator assessing the quality of answers to questions based on provided context.

Your task is to compare two answers and determine which one is better. Evaluate based on these criteria:

1. **Faithfulness to Context**: Does the answer accurately reflect information from the context without hallucinations?
2. **Completeness**: Does the answer address all aspects of the question?
3. **Coherence**: Is the answer well-organized and logically structured?
4. **Relevance**: Does the answer focus on what was asked without unnecessary tangents?

IMPORTANT: Base your evaluation ONLY on how well each answer represents the information in the context. Do not prefer answers simply because they are longer or more detailed if that additional detail is not supported by the context.

CRITICAL - Extracting phrases for highlighting:
When you write your detailed explanation, you will discuss specific strengths, weaknesses, and differences. For each answer, you MUST extract the EXACT phrases from the answer text that you are discussing in your explanation.

For example, if in your explanation you write: "Answer A explicitly states the 53% international revenue figure..."
Then key_differences_a MUST include the exact phrase from Answer A like: "53% of NVIDIA's revenue was generated from sales outside the United States"

The phrases you extract should:
- Be EXACT quotes from the answer text (copy-paste, not paraphrased)
- Be the specific content you reference in your detailed explanation
- Include enough context to be meaningful (typically 10-150 characters)
- Cover the main points you discuss for each answer's strengths/weaknesses
- NOT be single words or short fragments like "Executive Summary" - extract the full meaningful statement

Provide scores from 1-10 for each answer on each criterion, overall scores, determine the winner (A, B, or TIE), the key phrases you discuss, and provide a detailed explanation of your judgment."""

        system_prompt = custom_system_prompt if custom_system_prompt else default_system_prompt

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

        # Define JSON schema for structured output
        json_schema = {
            "type": "object",
            "properties": {
                "winner": {
                    "type": "string",
                    "enum": ["A", "B", "TIE"],
                    "description": "The winning answer: A, B, or TIE"
                },
                "score_a": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "description": "Score for Answer A (1-10)"
                },
                "score_b": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "description": "Score for Answer B (1-10)"
                },
                "faithfulness_a": {"type": "integer", "minimum": 1, "maximum": 10},
                "faithfulness_b": {"type": "integer", "minimum": 1, "maximum": 10},
                "completeness_a": {"type": "integer", "minimum": 1, "maximum": 10},
                "completeness_b": {"type": "integer", "minimum": 1, "maximum": 10},
                "coherence_a": {"type": "integer", "minimum": 1, "maximum": 10},
                "coherence_b": {"type": "integer", "minimum": 1, "maximum": 10},
                "relevance_a": {"type": "integer", "minimum": 1, "maximum": 10},
                "relevance_b": {"type": "integer", "minimum": 1, "maximum": 10},
                "key_differences_a": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "EXACT quotes from Answer A that you discuss in your explanation. Copy-paste the actual phrases you reference when describing A's strengths/weaknesses. 3-6 phrases, each 10-150 chars."
                },
                "key_differences_b": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "EXACT quotes from Answer B that you discuss in your explanation. Copy-paste the actual phrases you reference when describing B's strengths/weaknesses. 3-6 phrases, each 10-150 chars."
                },
                "explanation": {
                    "type": "string",
                    "description": "Detailed explanation of the judgment"
                }
            },
            "required": ["winner", "score_a", "score_b", "key_differences_a", "key_differences_b", "explanation"]
        }

        import json

        try:
            result = None

            if self.provider == LLMProvider.OPENAI:
                response = await client.chat.completions.create(
                    model=self.model.value,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "judge_evaluation",
                            "strict": True,
                            "schema": json_schema
                        }
                    }
                )
                result = json.loads(response.choices[0].message.content)

            elif self.provider == LLMProvider.ANTHROPIC:
                # Anthropic uses tool_use for structured output
                response = await client.messages.create(
                    model=self.model.value,
                    max_tokens=8192,
                    temperature=0.1,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": user_prompt}
                    ],
                    tools=[{
                        "name": "submit_evaluation",
                        "description": "Submit the evaluation results",
                        "input_schema": json_schema
                    }],
                    tool_choice={"type": "tool", "name": "submit_evaluation"}
                )
                # Extract from tool use response
                for block in response.content:
                    if block.type == "tool_use" and block.name == "submit_evaluation":
                        result = block.input
                        break

            elif self.provider == LLMProvider.GEMINI:
                # Gemini uses response_schema for structured output
                import google.generativeai as genai

                # Recreate client with JSON mode
                generation_config = genai.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=8192,
                    response_mime_type="application/json",
                    response_schema={
                        "type": "object",
                        "properties": {
                            "winner": {"type": "string", "enum": ["A", "B", "TIE"]},
                            "score_a": {"type": "integer"},
                            "score_b": {"type": "integer"},
                            "faithfulness_a": {"type": "integer"},
                            "faithfulness_b": {"type": "integer"},
                            "completeness_a": {"type": "integer"},
                            "completeness_b": {"type": "integer"},
                            "coherence_a": {"type": "integer"},
                            "coherence_b": {"type": "integer"},
                            "relevance_a": {"type": "integer"},
                            "relevance_b": {"type": "integer"},
                            "key_differences_a": {"type": "array", "items": {"type": "string"}},
                            "key_differences_b": {"type": "array", "items": {"type": "string"}},
                            "explanation": {"type": "string"}
                        },
                        "required": ["winner", "score_a", "score_b", "key_differences_a", "key_differences_b", "explanation"]
                    }
                )
                full_prompt = f"{system_prompt}\n\n{user_prompt}"
                response = await client.generate_content_async(
                    full_prompt,
                    generation_config=generation_config
                )
                result = json.loads(response.text)

            if result:
                # Convert flat schema to nested structure expected by UI
                return {
                    'winner': result.get('winner', 'TIE'),
                    'explanation': result.get('explanation', 'No explanation provided'),
                    'scores': {
                        'A': result.get('score_a', 5),
                        'B': result.get('score_b', 5)
                    },
                    'criteria_breakdown': {
                        'faithfulness': {
                            'A': result.get('faithfulness_a', 5),
                            'B': result.get('faithfulness_b', 5)
                        },
                        'completeness': {
                            'A': result.get('completeness_a', 5),
                            'B': result.get('completeness_b', 5)
                        },
                        'coherence': {
                            'A': result.get('coherence_a', 5),
                            'B': result.get('coherence_b', 5)
                        },
                        'relevance': {
                            'A': result.get('relevance_a', 5),
                            'B': result.get('relevance_b', 5)
                        }
                    },
                    'key_differences': {
                        'A': result.get('key_differences_a', []),
                        'B': result.get('key_differences_b', [])
                    }
                }
            else:
                return {
                    'winner': 'TIE',
                    'explanation': 'Could not parse evaluation response',
                    'scores': {'A': 5, 'B': 5},
                    'key_differences': {'A': [], 'B': []},
                    'criteria_breakdown': {}
                }

        except Exception as e:
            return {
                'winner': 'ERROR',
                'explanation': f'Evaluation failed: {str(e)}',
                'scores': {'A': 0, 'B': 0},
                'criteria_breakdown': {},
                'key_differences': {'A': [], 'B': []}
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
