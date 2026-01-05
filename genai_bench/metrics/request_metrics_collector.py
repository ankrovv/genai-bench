from genai_bench.logging import init_logger
from genai_bench.metrics.metrics import RequestLevelMetrics
from genai_bench.protocol import (
    UserChatResponse,
    UserImageGenerationResponse,
    UserResponse,
)

logger = init_logger(__name__)


class RequestMetricsCollector:
    """
    A class to collect and calculate metrics for individual requests.

    Attributes:
        metrics (RequestLevelMetrics): An instance to store metrics related
            to a single request.
    """

    def __init__(self):
        self.metrics = RequestLevelMetrics()

    def calculate_metrics(
        self,
        response: UserResponse,
    ):
        """
        Calculates various metrics from the response of a request.

        Args:
            response (UserResponse): The customized UserResponse object
                containing the response data needed to calculate metrics.
        """
        assert (
            response.num_prefill_tokens is not None
        ), "response.num_prefill_tokens is None"
        assert (
            response.time_at_first_token is not None
        ), "response.time_at_first_token is None"
        assert response.start_time is not None, "response.start_time is None"
        assert response.end_time is not None, "response.end_time is None"

        # Safely calculate common metrics
        self.metrics.num_input_tokens = response.num_prefill_tokens
        self.metrics.ttft = response.time_at_first_token - response.start_time
        self.metrics.e2e_latency = response.end_time - response.start_time
        self.metrics.total_tokens = self.metrics.num_input_tokens

        # Calculate prefill throughput
        self.metrics.input_throughput = (
            self.metrics.num_input_tokens / self.metrics.ttft
            if self.metrics.ttft
            else 0
        )

        # Check if the response is a UserChatResponse for output metrics
        if isinstance(response, UserChatResponse):
            self._calculate_output_metrics(response)
        elif isinstance(response, UserImageGenerationResponse):
            self._calculate_image_generation_metrics(response)
        else:
            # For non-chat responses, reset output metrics to avoid NoneType
            # Error in AggregatedMetricsCollector
            self._reset_output_metrics()

    def _calculate_output_metrics(self, response: UserChatResponse):
        """
        Helper function to calculate output metrics from a UserChatResponse.
        """
        assert response.tokens_received is not None, "response.tokens_received is None"
        self.metrics.num_output_tokens = response.tokens_received
        self.metrics.total_tokens += self.metrics.num_output_tokens
        self.metrics.output_latency = self.metrics.e2e_latency - self.metrics.ttft

        # Avoid divide by zero for tokens
        if self.metrics.num_output_tokens > 1:
            self.metrics.tpot = self.metrics.output_latency / (
                self.metrics.num_output_tokens - 1
            )
            self.metrics.output_inference_speed = 1 / self.metrics.tpot
            self.metrics.output_throughput = (
                (self.metrics.num_output_tokens - 1) / self.metrics.output_latency
                if self.metrics.output_latency
                else 0
            )
        else:
            logger.warning(
                f"‼️ num_output_tokens:"
                f"{self.metrics.num_output_tokens} is <= 1. Please check"
                f" your server and request!"
            )

    def _calculate_image_generation_metrics(
        self, response: UserImageGenerationResponse
    ):
        """
        Helper function to calculate metrics from a UserImageGenerationResponse.

        For non-streaming image generation tasks:
        - num_output_tokens represents the number of images generated
        - output_latency is 0 (entire image arrives at once)
        - tpot and related speed metrics don't make sense and will be filtered out
        """
        assert response.tokens_received is not None, "response.tokens_received is None"
        self.metrics.num_output_tokens = response.tokens_received
        self.metrics.total_tokens += self.metrics.num_output_tokens

        # For non-streaming image generation, output arrives all at once
        # so output_latency is 0. The filter_metrics() function will
        # automatically set tpot and output_inference_speed to None
        # since output_latency < 0.001s
        self.metrics.output_latency = 0.0
        self.metrics.tpot = 0.0
        self.metrics.output_inference_speed = 0.0
        self.metrics.output_throughput = 0.0

    def _reset_output_metrics(self):
        """Helper function to reset all output-related metrics to 0."""
        for field in RequestLevelMetrics.OUTPUT_METRICS_FIELDS:
            setattr(self.metrics, field, 0)
