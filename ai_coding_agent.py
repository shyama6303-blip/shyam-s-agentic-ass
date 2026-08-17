import os
import sys
import io
import traceback
from typing import TypedDict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END


# ============================================================
# 1. FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="AI Coding Agent",
    description="AI-powered coding, testing and execution pipeline",
    version="2.0.0",
)


# ============================================================
# 2. GEMINI CONFIGURATION
# ============================================================

# Render should have GEMINI_API_KEY configured in:
# Dashboard -> Environment -> Add Environment Variable
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

# Stable Gemini model suitable for coding/agentic tasks.
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

llm = None

if api_key:
    try:
        # Do not pass temperature here. Current Gemini 3.6 guidance
        # removes deprecated sampling parameters.
        llm = ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            google_api_key=api_key,
        )
    except Exception as exc:
        print(f"Gemini initialization error: {exc}")
        llm = None
else:
    print("WARNING: GEMINI_API_KEY is not configured.")


# ============================================================
# 3. STATE
# ============================================================

class CrewState(TypedDict, total=False):
    messages: List[BaseMessage]
    code: Optional[str]
    report: Optional[str]


# ============================================================
# 4. REQUEST MODEL
# ============================================================

class TaskRequest(BaseModel):
    task: str


# ============================================================
# 5. HELPERS
# ============================================================

def extract_content(response) -> str:
    """Safely convert a LangChain Gemini response to plain text."""
    content = getattr(response, "content", response)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))
            else:
                parts.append(str(item))

        return "\n".join(parts)

    return str(content)


def clean_python_code(code: str) -> str:
    """Remove accidental Markdown code fences."""
    code = str(code).strip()

    if code.startswith("```python"):
        code = code[len("```python"):]

    elif code.startswith("```Python"):
        code = code[len("```Python"):]

    elif code.startswith("```"):
        code = code[3:]

    if code.endswith("```"):
        code = code[:-3]

    return code.strip()


# ============================================================
# 6. PYTHON EXECUTION
# ============================================================

def run_python_code(code: str) -> str:
    """
    Execute generated Python code in the current application process.

    NOTE:
    This is intended for a controlled workshop/demo environment.
    Do not execute untrusted code this way in a production system.
    """

    if not isinstance(code, str):
        code = str(code)

    clean_code = clean_python_code(code)

    old_stdout = sys.stdout
    new_stdout = io.StringIO()

    sys.stdout = new_stdout

    try:
        local_scope = {}
        exec(clean_code, {"__name__": "__main__"}, local_scope)

        result = new_stdout.getvalue()

    except Exception:
        result = (
            "Execution Error:\n"
            + traceback.format_exc()
        )

    finally:
        sys.stdout = old_stdout

    result = result.strip()

    return result if result else "Success (no terminal output)"


# ============================================================
# 7. GENERATE TEST CASES
# ============================================================

def generate_test_cases(task_description: str) -> str:
    """Ask Gemini to generate QA test scenarios."""

    if llm is None:
        return (
            "Test generation unavailable because Gemini is not configured."
        )

    prompt = f"""
You are a Senior QA Engineer.

Generate 3 to 5 highly specific test scenarios for this coding task:

{task_description}

Include:
1. Normal cases
2. Boundary cases
3. Edge cases
4. Invalid input cases where appropriate

Return only a numbered list.
"""

    response = llm.invoke(prompt)
    return extract_content(response)


# ============================================================
# 8. DEVELOPER NODE
# ============================================================

def real_time_developer(state: CrewState):
    print("[Developer] Generating Python code...")

    if llm is None:
        raise ValueError(
            "Gemini API is not configured. "
            "Set GEMINI_API_KEY in the Render Environment variables."
        )

    messages = state.get("messages", [])

    if not messages:
        raise ValueError("No coding task was provided.")

    task = messages[-1].content

    developer_prompt = f"""
You are an expert Python developer.

Write a complete, executable Python program for the following task:

{task}

Requirements:
- Return ONLY Python source code.
- Do NOT use Markdown.
- Do NOT include ```python.
- The program must be executable directly with Python.
- Use clear variable names.
- Handle reasonable edge cases.
- If the task asks for output, make sure the program prints the result.
- Do not explain the code outside the Python source.
"""

    response = llm.invoke(developer_prompt)
    code_str = clean_python_code(extract_content(response))

    if not code_str:
        raise ValueError("Gemini returned empty Python code.")

    print("\nGenerated Code:")
    print(code_str)

    return {
        "code": code_str
    }


# ============================================================
# 9. TESTER NODE
# ============================================================

def real_time_tester(state: CrewState):
    print("[Tester] Generating tests and executing code...")

    task = state["messages"][-1].content
    generated_code = state.get("code", "")

    if not generated_code:
        raise ValueError("No generated code available for testing.")

    # Generate QA scenarios
    test_cases = generate_test_cases(task)

    # Execute generated Python
    execution_result = run_python_code(generated_code)

    report = (
        "### EXECUTION OUTPUT\n\n"
        f"{execution_result}\n\n"
        "### TEST SCENARIOS\n\n"
        f"{test_cases}"
    )

    print("\nTest Report:")
    print(report)

    return {
        "report": report
    }


# ============================================================
# 10. LANGGRAPH WORKFLOW
# ============================================================

workflow = StateGraph(CrewState)

workflow.add_node("developer", real_time_developer)
workflow.add_node("tester", real_time_tester)

workflow.add_edge(START, "developer")
workflow.add_edge("developer", "tester")
workflow.add_edge("tester", END)

rt_app = workflow.compile()


# ============================================================
# 11. FRONTEND
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Coding Agent</title>

    <style>
        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            font-family: Arial, sans-serif;
            background: #0f172a;
            color: white;
        }

        .container {
            max-width: 1000px;
            margin: auto;
            padding: 40px 20px;
        }

        h1 {
            text-align: center;
            font-size: 42px;
            margin-bottom: 10px;
        }

        .subtitle {
            text-align: center;
            color: #94a3b8;
            font-size: 18px;
            margin-bottom: 35px;
        }

        .card {
            background: #1e293b;
            padding: 25px;
            border-radius: 16px;
            margin-bottom: 25px;
        }

        textarea {
            width: 100%;
            height: 150px;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #475569;
            background: #0f172a;
            color: white;
            font-size: 16px;
            resize: vertical;
        }

        textarea:focus {
            outline: 2px solid #6366f1;
        }

        button {
            width: 100%;
            margin-top: 15px;
            padding: 15px;
            border: none;
            border-radius: 10px;
            background: #6366f1;
            color: white;
            font-size: 17px;
            font-weight: bold;
            cursor: pointer;
        }

        button:hover {
            background: #4f46e5;
        }

        button:disabled {
            background: #475569;
            cursor: not-allowed;
        }

        .status {
            margin-top: 15px;
            text-align: center;
            font-weight: bold;
        }

        .success {
            color: #4ade80;
        }

        .error {
            color: #f87171;
        }

        pre {
            background: #020617;
            padding: 20px;
            border-radius: 10px;
            overflow-x: auto;
            white-space: pre-wrap;
            color: #e2e8f0;
            min-height: 100px;
        }
    </style>
</head>

<body>
    <div class="container">

        <h1>🤖 AI Coding Agent</h1>

        <p class="subtitle">
            Generate • Test • Execute Python Code using AI
        </p>

        <div class="card">

            <h2>📝 Enter Coding Task</h2>

            <textarea
                id="task"
                placeholder="Example: Write a Python program to check whether a number is prime."
            ></textarea>

            <button id="runButton" onclick="runAgent()">
                🚀 Run AI Agent
            </button>

            <div id="status" class="status"></div>

        </div>

        <div class="card">

            <h2>💻 Generated Code</h2>

            <pre id="code">Your generated code will appear here.</pre>

        </div>

        <div class="card">

            <h2>🧪 Test & Execution Report</h2>

            <pre id="report">Your test report will appear here.</pre>

        </div>

    </div>

    <script>
        async function runAgent() {

            const task =
                document.getElementById("task").value.trim();

            const status =
                document.getElementById("status");

            const code =
                document.getElementById("code");

            const report =
                document.getElementById("report");

            const button =
                document.getElementById("runButton");

            if (!task) {
                status.innerText =
                    "⚠️ Please enter a coding task.";

                status.className =
                    "status error";

                return;
            }

            button.disabled = true;
            button.innerText =
                "⏳ AI Agent Running...";

            status.innerText =
                "⏳ Generating code and running tests...";

            status.className = "status";

            code.innerText =
                "Generating code...";

            report.innerText =
                "Running tests...";

            try {

                const response = await fetch("/run", {
                    method: "POST",

                    headers: {
                        "Content-Type": "application/json"
                    },

                    body: JSON.stringify({
                        task: task
                    })
                });

                let data;

                try {
                    data = await response.json();
                } catch {
                    throw new Error(
                        "Server returned an invalid response."
                    );
                }

                if (!response.ok) {
                    throw new Error(
                        data.detail || "Server error"
                    );
                }

                code.innerText =
                    data.code ||
                    "No code generated.";

                report.innerText =
                    data.report ||
                    "No report generated.";

                status.innerText =
                    "✅ AI Agent completed successfully.";

                status.className =
                    "status success";

            } catch (error) {

                status.innerText =
                    "❌ " + error.message;

                status.className =
                    "status error";

                code.innerText = "";
                report.innerText = "";

            } finally {

                button.disabled = false;

                button.innerText =
                    "🚀 Run AI Agent";
            }
        }
    </script>

</body>
</html>
"""


# ============================================================
# 12. HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "gemini_configured": bool(api_key),
        "model": MODEL_NAME,
    }


# ============================================================
# 13. RUN AI CODING AGENT
# ============================================================

@app.post("/run")
def run_agent(request: TaskRequest):

    task = request.task.strip()

    if not task:
        raise HTTPException(
            status_code=400,
            detail="Task cannot be empty."
        )

    if not api_key:
        raise HTTPException(
            status_code=500,
            detail=(
                "GEMINI_API_KEY is not configured on the Render server. "
                "Add GEMINI_API_KEY under Render → Environment Variables "
                "and redeploy."
            )
        )

    if llm is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "Gemini could not be initialized. "
                "Check GEMINI_API_KEY and the installed package versions."
            )
        )

    try:

        initial_state: CrewState = {
            "messages": [
                HumanMessage(content=task)
            ],
            "code": None,
            "report": None,
        }

        result = rt_app.invoke(
            initial_state,
            config={
                "recursion_limit": 20
            }
        )

        return {
            "success": True,
            "task": task,
            "code": result.get("code", ""),
            "report": result.get("report", ""),
        }

    except Exception as exc:

        print(traceback.format_exc())

        raise HTTPException(
            status_code=500,
            detail=f"AI Agent Error: {str(exc)}"
        )


# ============================================================
# 14. LOCAL RUN
# ============================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        reload=False,
    )