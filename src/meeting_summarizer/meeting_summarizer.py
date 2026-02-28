import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.model_provider import ModelProvider


class MeetingSummarizer:
    def __init__(
        self,
        model: str | None = None,
        llm_model: str | None = None,
        embedding_model: str | None = None,
        exec_env: str | None = None,
        summary_system_prompt: str = "",
        action_item_system_prompt: str = "",
    ):
        selected_llm_model = llm_model or model
        self.model_provider = ModelProvider(
            exec_env=exec_env,
            llm_model=selected_llm_model,
            embedding_model=embedding_model,
        )
        self.exec_env = self.model_provider.exec_env
        self.model = self.model_provider.llm_model
        self.embedding_model_name = self.model_provider.embedding_model
        self.embedding_model = self.model_provider.get_langchain_embeddings()
        self.summary_system_prompt = summary_system_prompt or (
            "You are an expert meeting summarizer. Your task is to read the provided meeting transcript and generate a concise summary that captures the key points discussed, decisions made, and action items assigned. The summary should be clear, well-structured, and easy to understand."
        )
        self.action_item_system_prompt = action_item_system_prompt or (
            """Extract all action items from the following meeting transcript. Identify individual 
and team-wide action items in the following format:

{
  "individual_actions": {
    "Alice": ["Task 1", "Task 2"],
    "Bob": ["Task 1"]
  },
  "team_actions": ["Task 1", "Task 2"],
  "entities": ["Alice", "Bob"]
}

Only include what is explicitly mentioned. Do not infer. You must respond strictly in 
valid JSON format — no extra text or commentary."""
        )

    def get_summary(self, transcript: str) -> str:
        try:
            return self.model_provider.chat_completion(
                system_prompt=self.summary_system_prompt,
                user_prompt=transcript,
                model_name=self.model,
            )
        except Exception as e:
            print(f"Error generating summary: {e}")
            return f"Error: Could not generate summary due to API issue: {e}"

    def get_action_items(self, transcript: str) -> dict:
        try:
            action_items = self.model_provider.chat_completion(
                system_prompt=self.action_item_system_prompt,
                user_prompt=transcript,
                model_name=self.model,
            )
            try:
                return json.loads(action_items)
            except json.JSONDecodeError:
                return {
                    "error": "Invalid JSON returned from model",
                    "raw_output": action_items,
                }
        except Exception as e:
            print(f"Error generating action items: {e}")
            return {"error": f"API call failed: {e}", "raw_output": ""}

    def summarize(self, transcript: str) -> tuple[str, dict]:
        summary = self.get_summary(transcript)
        action_items = self.get_action_items(transcript)

        return summary, action_items


if __name__ == "__main__":
    transcript_path = Path(__file__).parent / "transcripts" / "meeting_transcript.txt"
    with open(transcript_path, "r", encoding="utf-8") as file:
        transcript = file.read().strip()

    summarizer = MeetingSummarizer()

    summary, action_items = summarizer.summarize(transcript)
    print("*" * 40)
    print(summary)
    print("*" * 40)
    print("JSON:")
    print(json.dumps(action_items, indent=2))
    print("*" * 40)
