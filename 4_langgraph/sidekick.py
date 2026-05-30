from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from typing import List, Any, Optional, Dict
from pydantic import BaseModel, Field
from werkzeug.exceptions import BadGateway

from IPython.display import Image, display
from langchain_core.runnables.graph import MermaidDrawMethod
from langchain_core.runnables.graph_mermaid import draw_mermaid_png

from sidekick_tools import playwright_tools, other_tools
import uuid
import asyncio
from datetime import datetime

DEFAULT_GPT_MODEL = "gpt-4o-mini"

load_dotenv(override=True)


class State(TypedDict):
    messages: Annotated[List[Any], add_messages]
    success_criteria: str
    clarifying_questions: str
    clarifying_answers: str
    feedback_on_work: Optional[str]
    success_criteria_met: bool
    user_input_needed: bool


class EvaluatorOutput(BaseModel):
    feedback: str = Field(description="Feedback on the assistant's response")
    success_criteria_met: bool = Field(description="Whether the success criteria have been met")
    user_input_needed: bool = Field(
        description="True if more input is needed from the user, or clarifications, or the assistant is stuck"
    )

class ClarifierOutput(BaseModel):
    clarifying_questions: str = Field(description="The clarifying questions to ask the user.")


class Sidekick:
    def __init__(self):
        self.worker_llm_with_tools = None
        self.evaluator_llm_with_output = None
        self.clarifier_llm_with_output = None
        self.tools = None
        self.llm_with_tools = None
        self.graph = None
        self.sidekick_id = str(uuid.uuid4())
        self.memory = MemorySaver()
        self.browser = None
        self.playwright = None

    async def setup(self):
        self.tools, self.browser, self.playwright = await playwright_tools()
        self.tools += await other_tools()

        self.worker_llm_with_tools = self._connect_model(tool_bind=True)

        self.evaluator_llm_with_output = self._connect_model(structured_output=EvaluatorOutput)

        self.clarifier_llm_with_output = self._connect_model(structured_output=ClarifierOutput)

        await self.build_graph()

    def _connect_model(self, tool_bind: bool = False, structured_output: Optional[Any] = None):
        """Attempt to connect to an LLM model."""
        try:
            llm = ChatOpenAI(model=DEFAULT_GPT_MODEL)
        except Exception as ex:
            raise BadGateway(f'Unable to connect to model: {DEFAULT_GPT_MODEL}. Error: {ex}')

        return (
            llm.bind_tools(self.tools) if tool_bind
            else llm.with_structured_output(structured_output)
        )

    def worker(self, state: State) -> Dict[str, Any]:
        system_message = f"""You are a helpful assistant that can use tools to complete tasks.
    You keep working on a task until either you have a question or clarification for the user, or the success criteria is met.
    You have many tools to help you, including tools to browse the internet, navigating and retrieving web pages.
    You have a tool to run python code, but note that you would need to include a print() statement if you wanted to receive output.
    The current date and time is {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

    This is the success criteria:
    {state["success_criteria"]}
    You should reply either with a question for the user about this assignment, or with your final response.

    If you have a question for the user, you need to reply by clearly stating your question. An example might be:

    Question: please clarify whether you want a summary or a detailed answer

    If you've finished, reply with the final answer, and don't ask a question; simply reply with the answer.
    """

        if state.get("clarifying_questions"):
            system_message += f"""
    These clarifying questions were asked to the user: {state["clarifying_questions"]}.
    """

        if state.get("clarifying_answers"):
            system_message += f"""
    Here is what the user responded with to clarifying questions which were asked: {state["clarifying_answers"]}
    """

        if state.get("feedback_on_work"):
            system_message += f"""
    Previously you thought you completed the assignment, but your reply was rejected because the success criteria was not met.
    Here is the feedback on why this was rejected:
    {state["feedback_on_work"]}
    With this feedback, please continue the assignment, ensuring that you meet the success criteria or have a question for the user."""

        # Add in the system message

        found_system_message = False
        messages = state["messages"]
        for message in messages:
            if isinstance(message, SystemMessage):
                message.content = system_message
                found_system_message = True

        if not found_system_message:
            messages = [SystemMessage(content=system_message)] + messages

        # Invoke the LLM with tools
        response = self.worker_llm_with_tools.invoke(messages)

        # Return updated state
        return {
            "messages": [response],
        }
    
    def clarifier(self, state: State) -> Dict[str, Any]:
        system_message = f"""You are a clarifier that asks the user for clarifying questions in order to help an Assistant complete a task.
        Based on the conversation history, identify what information is missing.

        The entire conversation with the assistant, with the user's original request and all replies, is:
        {self.format_conversation(state["messages"])}
        """
        
        clarifier_messages = [
            SystemMessage(content=system_message),
            HumanMessage(content="What clarifying questions should I ask the user?")
        ]

        print ("Sending clarifier messages to LLM to generate questions")
        clarifier_response: ClarifierOutput = self.clarifier_llm_with_output.invoke(clarifier_messages)

        return {
            "messages": [
                AIMessage(content=f"CLARIFICATION_NEEDED: {clarifier_response.clarifying_questions}")
            ],
            "clarifying_questions": clarifier_response.clarifying_questions
        }

    def worker_router(self, state: State) -> str:
        last_message = state["messages"][-1]

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            print("Routing to tools")
            return "tools"

        # Check if we have already asked clarifying questions in this thread.
        # This covers the "up front" requirement.
        has_clarified = any(
            isinstance(m, AIMessage) and "CLARIFICATION_NEEDED" in m.content 
            for m in state["messages"]
        )
        
        if not has_clarified:
            print("Routing to clarifier (Initial clarification)")
            return "clarifier"

        print("Routing to evaluator")
        return "evaluator"

    def clarifier_router(self, state: State) -> str:
        # After clarifier runs and we hit the breakpoint, the next step after resumption
        # should always be the worker so it can use the new information.
        print("Clarifier routing to worker to refine answer")
        return "worker"

    def format_conversation(self, messages: List[Any]) -> str:
        conversation = "Conversation history:\n\n"
        for message in messages:
            if isinstance(message, HumanMessage):
                conversation += f"User: {message.content}\n"
            elif isinstance(message, AIMessage):
                text = message.content or "[Tools use]"
                conversation += f"Assistant: {text}\n"
        return conversation

    def evaluator(self, state: State) -> State:
        last_response = state["messages"][-1].content

        system_message = """You are an evaluator that determines if a task has been completed successfully by an Assistant.
    Assess the Assistant's last response based on the given criteria. Respond with your feedback, and with your decision on whether the success criteria has been met,
    and whether more input is needed from the user."""

        user_message = f"""You are evaluating a conversation between the User and Assistant. You decide what action to take based on the last response from the Assistant.

    The entire conversation with the assistant, with the user's original request and all replies, is:
    {self.format_conversation(state["messages"])}

    The success criteria for this assignment is:
    {state["success_criteria"]}

    And the final response from the Assistant that you are evaluating is:
    {last_response}

    Respond with your feedback, and decide if the success criteria is met by this response.
    Also, decide if more user input is required, either because the assistant has a question, needs clarification, or seems to be stuck and unable to answer without help.

    The Assistant has access to a tool to write files. If the Assistant says they have written a file, then you can assume they have done so.
    Overall you should give the Assistant the benefit of the doubt if they say they've done something. But you should reject if you feel that more work should go into this.

    """
        if state["feedback_on_work"]:
            user_message += f"Also, note that in a prior attempt from the Assistant, you provided this feedback: {state['feedback_on_work']}\n"
            user_message += "If you're seeing the Assistant repeating the same mistakes, then consider responding that user input is required."

        evaluator_messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=user_message),
        ]

        eval_result = self.evaluator_llm_with_output.invoke(evaluator_messages)
        new_state = {
            "messages": [
                {
                    "role": "assistant",
                    "content": f"Evaluator Feedback on this answer: {eval_result.feedback}",
                }
            ],
            "feedback_on_work": eval_result.feedback,
            "success_criteria_met": eval_result.success_criteria_met,
            "user_input_needed": eval_result.user_input_needed,
        }
        return new_state

    def route_based_on_evaluation(self, state: State) -> str:
        if state["success_criteria_met"]:
            return "END"
        
        if state["user_input_needed"]:
            print("Routing to clarifier (evaluator requested input)")
            return "clarifier"
        
        return "worker"

    async def build_graph(self):
        # Set up Graph Builder with State
        graph_builder = StateGraph(State)

        # Add nodes
        graph_builder.add_node("worker", self.worker)
        graph_builder.add_node("tools", ToolNode(tools=self.tools))
        # Responsible for asking the user for clarifying question.
        graph_builder.add_node("clarifier", self.clarifier)
        graph_builder.add_node("evaluator", self.evaluator)

        # Add Edge to trigger asking the user for clarifying questions.
        graph_builder.add_conditional_edges(
            "clarifier", self.clarifier_router,
            {"worker": "worker"}
        )

        # Add edges
        graph_builder.add_conditional_edges(
            "worker", self.worker_router, {"tools": "tools", "clarifier": "clarifier", "evaluator": "evaluator"}
        )
        graph_builder.add_edge("tools", "worker")
        graph_builder.add_conditional_edges(
            "evaluator", self.route_based_on_evaluation, {"worker": "worker", "clarifier": "clarifier", "END": END}
        )
        graph_builder.add_edge(START, "worker")

        # Compile the graph with a breakpoint after clarifier
        self.graph = graph_builder.compile(
            checkpointer=self.memory,
            interrupt_after=["clarifier"]
        )

        render_mermaid(self.graph)

    async def run_superstep(self, message, success_criteria, history):
        config = {"configurable": {"thread_id": self.sidekick_id}}
        
        # Check if we are resumed from a breakpoint
        state = await self.graph.aget_state(config)
        
        if state.next:
            # We are interrupted, 'message' is the answer to clarifying questions
            print(f"Resuming from breakpoint {state.next}. User message: {message}")
            
            # Update state with the user answers
            await self.graph.aupdate_state(
                config, 
                {"messages": [HumanMessage(content=f"User Answers: {message}")]}
            )
            
            # Resume execution. result will contain all messages from the beginning of the thread
            result = await self.graph.ainvoke(None, config=config)
        else:
            # Initial run or fresh start
            initial_state = {
                "messages": [HumanMessage(content=message)],
                "success_criteria": success_criteria or "The answer should be clear and accurate",
                "clarifying_questions": None,
                "clarifying_answers": None,
                "feedback_on_work": None,
                "success_criteria_met": False,
                "user_input_needed": True,
            }
            result = await self.graph.ainvoke(initial_state, config=config)

        # Get updated state to check if we are interrupted again
        final_state = await self.graph.aget_state(config)
        
        # When resuming, 'result' is the full state dictionary
        messages = result["messages"]
        last_message = messages[-1]
        
        user_entry = {"role": "user", "content": message}
        
        if final_state.next:
            # We hit a breakpoint (clarifier)
            clarification = last_message.content.replace("CLARIFICATION_NEEDED: ", "")
            assistant_entry = {"role": "assistant", "content": clarification}
            return history + [user_entry, assistant_entry]
        else:
            # Graph finished. 
            # Search for the last worker and evaluator messages in the current run's results.
            # When resuming, result['messages'] contains the entire thread.
            
            worker_reply = "No response from worker"
            evaluator_feedback = "No feedback from evaluator"
            
            # The very last message should be the Evaluator's feedback
            if len(messages) > 0:
                evaluator_feedback = messages[-1].content
            
            # Look for the worker's reply before the evaluator's
            for msg in reversed(messages[:-1]):
                if isinstance(msg, AIMessage) and "Evaluator Feedback" not in msg.content and "CLARIFICATION_NEEDED" not in msg.content:
                    worker_reply = msg.content
                    break

            reply = {"role": "assistant", "content": worker_reply}
            feedback = {"role": "assistant", "content": evaluator_feedback}
            return history + [user_entry, reply, feedback]

    def cleanup(self):
        if self.browser:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.browser.close())
                if self.playwright:
                    loop.create_task(self.playwright.stop())
            except RuntimeError:
                # If no loop is running, do a direct run
                asyncio.run(self.browser.close())
                if self.playwright:
                    asyncio.run(self.playwright.stop())


def render_mermaid(graph):
    # Extract the graph structure and generate the png bytes
    png_bytes = graph.get_graph().draw_mermaid_png()

    # Display the image inline using IPython
    display(Image(png_bytes))

    # Save the bytes to a PNG file
    with open("sidekick_graph.png", "wb") as f:
        f.write(png_bytes)
    f.close()
