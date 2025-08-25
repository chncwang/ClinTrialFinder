from typing import Dict, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class AITokenPricing:
    """Utility class for calculating OpenAI API costs."""

    # Token estimation ratio (characters to tokens)
    CHAR_TO_TOKEN_RATIO = 0.25

    # Cost per 1K tokens in USD for different models
    MODEL_COSTS: Dict[str, Tuple[float, float]] = {
        "gpt-5": (1.25e-3, 10.0e-3),  # $1.25 input, $10.00 output per 1K tokens
        "gpt-4o-mini": (0.15e-3, 0.60e-3),  # $0.15 input, $0.60 output per 1K tokens
        "gpt-4o": (2.5e-3, 10e-3),  # $2.50 input, $10.00 output per 1K tokens
        "gpt-4.1-mini": (0.4e-3, 1.6e-3),  # $0.40 input, $1.60 output per 1K tokens
        "gpt-4.1": (2.0e-3, 8.0e-3),  # $2.00 input, $8.00 output per 1K tokens
        "sonar-pro": (3e-3, 15e-3),  # $0.30 input, $1.50 output per 1K tokens
        "sonar": (1e-3, 1e-3),  # $0.001 input, $0.001 output per 1K tokens
    }

    @classmethod
    def estimate_tokens(cls, text: Union[str, bytes]) -> float:
        """Estimate the number of tokens from text using character ratio."""
        if isinstance(text, bytes):
            text = text.decode("utf-8")
        return len(text) * cls.CHAR_TO_TOKEN_RATIO

    @classmethod
    def calculate_cost(
        cls, prompt: str, response: str, model: str = "gpt-4.1-mini"
    ) -> float:
        """Calculate the approximate cost for API usage.

        Args:
            prompt: Input text
            response: Output text
            model: Model name (defaults to gpt-4.1-mini)

        Returns:
            float: Estimated cost in USD

        Raises:
            ValueError: If model is not found in MODEL_COSTS
        """
        if model not in cls.MODEL_COSTS:
            raise ValueError(
                f"Unknown model: {model}. Available models: {list(cls.MODEL_COSTS.keys())}"
            )

        input_tokens = cls.estimate_tokens(prompt)
        logger.debug(f"calculate_cost: Estimated input tokens: {input_tokens:.1f}")
        output_tokens = cls.estimate_tokens(response)
        logger.debug(f"calculate_cost: Estimated output tokens: {output_tokens:.1f}")

        input_cost, output_cost = cls.MODEL_COSTS[model]
        return (input_tokens * input_cost + output_tokens * output_cost) * 1e-3
