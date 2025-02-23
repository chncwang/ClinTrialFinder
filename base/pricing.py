from typing import Dict, Tuple, Union


class AITokenPricing:
    """Utility class for calculating OpenAI API costs."""

    # Token estimation ratio (characters to tokens)
    CHAR_TO_TOKEN_RATIO = 0.25

    # Cost per 1K tokens in USD for different models
    MODEL_COSTS: Dict[str, Tuple[float, float]] = {
        "gpt-4o-mini": (0.15e-3, 0.60e-3),  # $0.15 input, $0.60 output per 1K tokens
        "sonar-pro": (0.3e-3, 1.5e-3),  # $0.30 input, $1.50 output per 1K tokens
    }

    @classmethod
    def estimate_tokens(cls, text: Union[str, bytes]) -> float:
        """Estimate the number of tokens from text using character ratio."""
        if isinstance(text, bytes):
            text = text.decode("utf-8")
        return len(text) * cls.CHAR_TO_TOKEN_RATIO

    @classmethod
    def calculate_cost(
        cls, prompt: str, response: str, model: str = "gpt-4o-mini"
    ) -> float:
        """Calculate the approximate cost for API usage.

        Args:
            prompt: Input text
            response: Output text
            model: Model name (defaults to gpt-4o-mini)

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
        output_tokens = cls.estimate_tokens(response)

        input_cost, output_cost = cls.MODEL_COSTS[model]
        return input_tokens * input_cost + output_tokens * output_cost
