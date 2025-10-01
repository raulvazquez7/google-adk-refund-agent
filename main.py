"""
Main entry point for the Barefoot Zénit Refund Agent CLI.

This script initializes and runs a command-line interface for interacting with the
refund agent. It handles loading environment variables, setting up logging,
and managing the conversation loop with Langfuse observability integration.
"""
import os
from dotenv import load_dotenv
import asyncio
import logging
import uuid

# Load .env FIRST
load_dotenv()

# Now import ADK and Langfuse
from google.adk.runners import InMemoryRunner
from google.genai import types
from src.agent import refund_agent
from langfuse import get_client

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
langfuse = get_client()

# --- Application Constants ---
APP_NAME = "barefoot_refund_cli"
USER_ID = "local_test_user"
SESSION_ID = f"session_{uuid.uuid4()}"

async def main():
    """
    Main function to run a chat session with the agent in the terminal,
    with tracing enabled for Langfuse Cloud.
    """
    print("--- Barefoot Zénit Refund Agent (with Langfuse Cloud) ---")
    print(f"Starting session... (Session ID: {SESSION_ID})")
    print("Type 'exit' to end the conversation.")
    print("-" * 50)

    runner = InMemoryRunner(agent=refund_agent, app_name=APP_NAME)
    
    session_service = runner.session_service
    await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
    )

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                print("--- Session ended ---")
                break

            # Create span using correct pattern from documentation
            with langfuse.start_as_current_span(name="chat-interaction"):
                # Update current trace with metadata
                langfuse.update_current_trace(
                    user_id=USER_ID,
                    session_id=SESSION_ID,
                    input={"message": user_input},
                    tags=["adk-agent", "barefoot-refund"]
                )

                final_response_text = ""
                async for event in runner.run_async(
                    user_id=USER_ID,
                    session_id=SESSION_ID,
                    new_message=types.Content(role="user", parts=[types.Part(text=user_input)])
                ):
                    logging.info(f"AGENT EVENT: {event}")

                    if event.is_final_response() and event.content:
                        parts = event.content.parts or []
                        text = parts[0].text if parts and hasattr(parts[0], "text") else ""
                        if text:
                            final_response_text = text.strip()
                            print(f"Agent: {final_response_text}")
                
                # Update trace with final response
                langfuse.update_current_trace(output={"response": final_response_text})
                
                # Get trace URL for debugging
                trace_url = langfuse.get_trace_url()
                print(f"\n🔍 View trace: {trace_url}")

            # Force immediate sending
            langfuse.flush()

        except KeyboardInterrupt:
            print("\n--- Session ended by user ---")
            break
        except Exception as e:
            logging.error(f"An error occurred: {e}", exc_info=True)
            break

if __name__ == "__main__":
    asyncio.run(main())