import logging
from dataclasses import dataclass, field
from litellm import completion_cost

logger = logging.getLogger(__name__)

@dataclass
class StepCost:
    step_number: int
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    is_tool_call: bool = False

@dataclass
class QueryCost:
    query: str
    steps: list[StepCost] = field(default_factory=list)
    total_cost_usd: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    def add_step(self, step: StepCost):
        self.steps.append(step)
        self.total_cost_usd += step.cost_usd
        self.total_input_tokens += step.input_tokens
        self.total_output_tokens += step.output_tokens

class CostTracker:
    """
    Tracks costs across agent executions.
    """
    def __init__(self):
        self.queries: list[QueryCost] = []
        self._current_query: QueryCost | None = None

    def start_query(self, query: str):
        self._current_query = QueryCost(query=query)

    def log_completion(self, step_number: int, response, is_tool_call: bool = False):
        """
        Log a completion response's cost.
        """
        if not self._current_query:
            return

        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        try:
            cost = completion_cost(completion_response=response)
        except Exception:
            cost = 0.0

        model = response.model or "unknown"
        step = StepCost(
            step_number=step_number,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            is_tool_call=is_tool_call,
        )
        self._current_query.add_step(step)

    def end_query(self):
        if self._current_query:
            self.queries.append(self._current_query)
            self._current_query = None

    def print_cost_breakdown(self):
        for qc in self.queries:
            print(f"\n{'='*50}")
            print(f"Query: {qc.query[:80]}")
            print(f"{'='*50}")
            for step in qc.steps:
                tool_tag = " [tool_call]" if step.is_tool_call else ""
                print(f"  Step {step.step_number}{tool_tag}: "
                      f"in={step.input_tokens} out={step.output_tokens} "
                      f"cost=${step.cost_usd:.4f} ({step.model})")
            print(f"  Total: in={qc.total_input_tokens} out={qc.total_output_tokens} "
                  f"cost=${qc.total_cost_usd:.4f}")

