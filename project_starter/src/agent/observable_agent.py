import asyncio
import json
import os
import time

import structlog
from litellm import acompletion

from src.observability.cost_tracker import CostTracker
from src.observability.loop_detector import AdvancedLoopDetector
from src.observability.tracer import AgentStep, AgentTracer, ToolCallRecord
from src.tools.registry import registry

logger = structlog.get_logger()


class ObservableAgent:
    """
    Production-grade agent with full observability.

    This agent implements the ReAct pattern (Reasoning + Acting) but enhances it
    with "Observability" - the ability to track, trace, and debug the agent's
    internal state and actions.
    """
    def __init__(
        self,
        model: str = None,
        max_steps: int = 10,
        agent_name: str = "ObservableAgent",
        verbose: bool = True,
        system_prompt: str = None,
        tools: list = None,
    ):
        self.model = model or os.getenv("MODEL_NAME", "ollama/qwen3:8b")
        self.max_steps = max_steps
        self.agent_name = agent_name
        self.system_prompt = system_prompt
        self.tools = tools if tools is not None else registry.get_all_tools()

        # Observability components
        self.tracer = AgentTracer(verbose=verbose)
        self.loop_detector = AdvancedLoopDetector()
        self.cost_tracker = CostTracker()

    def _build_tools_schema(self) -> list[dict]:
        return [tool.to_openai_schema() for tool in self.tools]

    async def _execute_tool_call(self, tool_call) -> ToolCallRecord:
        """Execute a single tool call and return a record."""
        func_name = tool_call.function.name
        args_str = tool_call.function.arguments
        start = time.time()

        try:
            args = json.loads(args_str)
        except json.JSONDecodeError:
            args = {}

        # Check for loops before executing
        loop_result = self.loop_detector.check_tool_call(func_name, args_str)
        if loop_result.is_looping:
            output = f"[LOOP DETECTED] {loop_result.message}"
        else:
            tool = registry.get_tool(func_name)
            if tool is None:
                output = f"Error: Tool '{func_name}' not found."
            else:
                try:
                    result = tool.execute(**args)
                    output = json.dumps(result) if not isinstance(result, str) else result
                except Exception as e:
                    output = f"Error executing {func_name}: {e}"

        duration = (time.time() - start) * 1000
        return ToolCallRecord(
            tool_name=func_name,
            tool_input=args,
            tool_output=output[:2000],
            duration_ms=duration,
        )

    async def run(self, user_query: str) -> dict:
        """Execute the agent loop with full observability."""
        trace_id = self.tracer.start_trace(self.agent_name, user_query, self.model)
        self.cost_tracker.start_query(user_query)
        self.loop_detector.reset()

        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_query})

        tools_schema = self._build_tools_schema() if self.tools else None

        try:
            for step_num in range(1, self.max_steps + 1):
                step_start = time.time()
                logger.info("agent_step", agent=self.agent_name, step=step_num)

                # Call LLM
                kwargs = {"model": self.model, "messages": messages}
                if tools_schema:
                    kwargs["tools"] = tools_schema
                    kwargs["tool_choice"] = "auto"

                response = await acompletion(**kwargs)
                choice = response.choices[0]
                message = choice.message

                # Log cost
                self.cost_tracker.log_completion(step_num, response)

                tool_records = []

                # If the model wants to call tools
                if message.tool_calls:
                    # Add assistant message with tool calls
                    messages.append(message.model_dump())

                    # Execute tool calls
                    for tc in message.tool_calls:
                        record = await self._execute_tool_call(tc)
                        tool_records.append(record)

                        # Add tool result to messages
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": record.tool_output,
                        })

                    # Log step
                    step_duration = (time.time() - step_start) * 1000
                    usage = response.usage
                    agent_step = AgentStep(
                        step_number=step_num,
                        reasoning=message.content,
                        tool_calls=tool_records,
                        input_tokens=usage.prompt_tokens if usage else 0,
                        output_tokens=usage.completion_tokens if usage else 0,
                        duration_ms=step_duration,
                    )
                    self.tracer.log_step(trace_id, agent_step)

                    # Check output stagnation
                    stagnation = self.loop_detector.check_output_stagnation(
                        message.content or ""
                    )
                    if stagnation.is_looping:
                        logger.warning("stagnation_detected", message=stagnation.message)
                        final = f"[Agent stopped: {stagnation.message}]"
                        self.tracer.end_trace(trace_id, final, status="stagnation")
                        self.cost_tracker.end_query()
                        return {"answer": final, "trace_id": trace_id}

                else:
                    # No tool calls — model produced a final answer
                    final_answer = message.content or ""
                    step_duration = (time.time() - step_start) * 1000
                    usage = response.usage
                    agent_step = AgentStep(
                        step_number=step_num,
                        reasoning=final_answer,
                        input_tokens=usage.prompt_tokens if usage else 0,
                        output_tokens=usage.completion_tokens if usage else 0,
                        duration_ms=step_duration,
                    )
                    self.tracer.log_step(trace_id, agent_step)
                    self.tracer.end_trace(trace_id, final_answer)
                    self.cost_tracker.end_query()
                    return {"answer": final_answer, "trace_id": trace_id}

            # Max steps reached
            final = "Max steps reached without a final answer."
            self.tracer.end_trace(trace_id, final, status="max_steps")
            self.cost_tracker.end_query()
            return {"answer": final, "trace_id": trace_id}

        except Exception as e:
            logger.error("agent_error", error=str(e))
            self.tracer.end_trace(trace_id, "", status="error", error=str(e))
            self.cost_tracker.end_query()
            return {"answer": f"Error: {e}", "trace_id": trace_id}
