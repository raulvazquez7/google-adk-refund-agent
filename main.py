import asyncio
import logging
import os
import uuid
from dotenv import load_dotenv

from google.adk.runners import InMemoryRunner
from google.genai import types
from src.agent import refund_agent

# --- Tracing Configuration (NUEVO) ---
from opentelemetry import trace
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
from opentelemetry.sdk.trace import TracerProvider, export

# --- Configuration ---
load_dotenv()

# --- GCP Configuration ---
PROJECT_ID = os.getenv("GCP_PROJECT_ID")

# --- Initialize Cloud Tracing (NUEVO) ---
# Esto configura el sistema de telemetría global.
# El ADK detectará automáticamente esta configuración y comenzará a exportar trazas.
provider = TracerProvider()
processor = export.BatchSpanProcessor(CloudTraceSpanExporter(project_id=PROJECT_ID))
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

APP_NAME = "barefoot_refund_cli"
USER_ID = "local_test_user"
SESSION_ID = f"session_{uuid.uuid4()}"


async def main():
    """
    Main function to run a chat session with the agent in the terminal.
    """
    print("--- Agente de Reembolsos de Barefoot Zénit ---")
    print(f"Iniciando sesión... (ID de sesión: {SESSION_ID})")
    print("Escribe 'salir' para terminar.")
    print("-" * 50)

    runner = InMemoryRunner(agent=refund_agent, app_name=APP_NAME)
    
    session_service = runner.session_service
    await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
    )

    while True:
        try:
            user_input = input("Tú: ")
            if user_input.lower() == 'salir':
                print("--- Sesión finalizada ---")
                break

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
                        print(f"Agente: {text.strip()}")

        except KeyboardInterrupt:
            print("\n--- Sesión finalizada por el usuario ---")
            break
        except Exception as e:
            logging.error(f"Ha ocurrido un error: {e}", exc_info=True)
            break

if __name__ == "__main__":
    asyncio.run(main())
