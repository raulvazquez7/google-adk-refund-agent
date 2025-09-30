import os
from dotenv import load_dotenv
import asyncio
import logging
import uuid

# Carga .env PRIMERO
load_dotenv()

# Ahora sí, importar ADK y Langfuse
from google.adk.runners import InMemoryRunner
from google.genai import types
from src.agent import refund_agent
from langfuse import get_client

# --- Configuración de Logging y Langfuse ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
langfuse = get_client()

# --- Constantes de la Aplicación ---
APP_NAME = "barefoot_refund_cli"
USER_ID = "local_test_user"
SESSION_ID = f"session_{uuid.uuid4()}"

async def main():
    """
    Función principal para ejecutar una sesión de chat con el agente en la terminal,
    con tracing habilitado para Langfuse Cloud.
    """
    print("--- Agente de Reembolsos de Barefoot Zénit (con Langfuse Cloud) ---")
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

            # Crear span usando el patrón correcto de la documentación
            with langfuse.start_as_current_span(name="chat-interaction"):
                # Actualizar la traza actual con metadatos
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
                            print(f"Agente: {final_response_text}")
                
                # Actualizar la traza con la respuesta final
                langfuse.update_current_trace(output={"response": final_response_text})
                
                # Obtener URL de la traza para debug
                trace_url = langfuse.get_trace_url()
                print(f"\n🔍 Ver traza: {trace_url}")

            # Forzar el envío inmediato
            langfuse.flush()

        except KeyboardInterrupt:
            print("\n--- Sesión finalizada por el usuario ---")
            break
        except Exception as e:
            logging.error(f"Ha ocurrido un error: {e}", exc_info=True)
            break

if __name__ == "__main__":
    asyncio.run(main())