from typing import Union


class OpenAITokenPricing:
    """Utility class for calculating OpenAI API costs."""

    # Token estimation ratio (characters to tokens)
    CHAR_TO_TOKEN_RATIO = 0.25

    # Cost per 1K tokens in USD
    GPT4_INPUT_COST = 0.15e-3  # $0.15 per 1K tokens
    GPT4_OUTPUT_COST = 0.60e-3  # $0.60 per 1K tokens

    @classmethod
    def estimate_tokens(cls, text: Union[str, bytes]) -> float:
        """Estimate the number of tokens from text using character ratio."""
        if isinstance(text, bytes):
            text = text.decode("utf-8")
        return len(text) * cls.CHAR_TO_TOKEN_RATIO

    @classmethod
    def calculate_cost(cls, prompt: str, response: str) -> float:
        """Calculate the approximate cost for GPT-4 API usage."""
        input_tokens = cls.estimate_tokens(prompt)
        output_tokens = cls.estimate_tokens(response)

        return input_tokens * cls.GPT4_INPUT_COST + output_tokens * cls.GPT4_OUTPUT_COST
