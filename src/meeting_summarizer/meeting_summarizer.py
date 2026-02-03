import os
from dotenv import load_dotenv
from openai import OpenAI
import ollama
import json

load_dotenv("../.env")


class MeetingSummarizer:
    def __init__(
        self,
        model: str = "gpt-4",
        summary_system_prompt: str = "",
        action_item_system_prompt: str = "",
    ):
        self.exec_env = os.getenv("EXECUTION_ENV", "local").lower()
        self.model = (
            model if self.exec_env == "cloud" else os.getenv("LOCAL_MODEL_NAME")
        )
        self.client = (
            OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            if self.exec_env == "cloud"
            else None
        )
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
        self.base_url = (
            os.getenv("LOCAL_MODEL_BASE_URL") if self.exec_env == "local" else None
        )

    def get_summary(self, transcript: str) -> str:
        try:
            if self.exec_env == "local":
                response = ollama.chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.summary_system_prompt},
                        {"role": "user", "content": transcript},
                    ],
                )
                summary = response["message"]["content"].strip()
                return summary
            elif self.exec_env == "cloud":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.summary_system_prompt},
                        {"role": "user", "content": transcript},
                    ],
                )

                summary = response.choices[0].message.content.strip()
                return summary
            else:
                raise ValueError(f"Unsupported execution environment: {self.exec_env}")
        except Exception as e:
            print(f"Error generating summary: {e}")
            return f"Error: Could not generate summary due to API issue: {e}"

    def get_action_items(self, transcript: str) -> dict:
        try:
            if self.exec_env == "local":
                response = ollama.chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.action_item_system_prompt},
                        {"role": "user", "content": transcript},
                    ],
                )

                action_items = response["message"]["content"].strip()
                try:
                    return json.loads(action_items)
                except json.JSONDecodeError:
                    return {
                        "error": "Invalid JSON returned from model",
                        "raw_output": action_items,
                    }
            elif self.exec_env == "cloud":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.action_item_system_prompt},
                        {"role": "user", "content": transcript},
                    ],
                )

                action_items = response.choices[0].message.content.strip()
            else:
                raise ValueError(f"Unsupported execution environment: {self.exec_env}")
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
    with open("transcripts/meeting_transcript.txt", "r") as file:
        transcript = file.read().strip()

    summarizer = MeetingSummarizer()

    summary, action_items = summarizer.summarize(transcript)
    print("*" * 40)
    print(summary)
    print("*" * 40)
    print("JSON:")
    print(json.dumps(action_items, indent=2))
    print("*" * 40)
