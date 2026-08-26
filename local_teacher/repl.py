import sys

def iniciar_repl(app) -> None:
    """Inicia el bucle interactivo REPL para el chat."""
    print("\n")
    while True:
        try:
            entrada = input(
                "Haz una pregunta de seguimiento (o 'salir' para terminar, '/web' para opciones): "
            ).strip()

            if not entrada:
                continue

            if entrada.lower() in ("salir", "exit", "quit", "q"):
                return

            if entrada.startswith("/web"):
                comando = entrada[4:].strip()
                if comando.lower() in ("off", "false"):
                    app.busqueda_web_alternativa = False
                    app.web_filter = ""
                    print("[*] Búsqueda web deshabilitada.")
                elif comando.lower() in ("on", "true", ""):
                    app.busqueda_web_alternativa = True
                    app.web_filter = ""
                    print("[*] Búsqueda web habilitada (sin filtros).")
                elif comando.lower().startswith("filter "):
                    app.busqueda_web_alternativa = True
                    app.web_filter = comando[7:].strip()
                    print(f"[*] Búsqueda web habilitada con filtro: {app.web_filter}")
                else:
                    print("[-] Uso incorrecto de /web. Opciones:")
                    print("    /web on          -> Activar búsqueda web")
                    print("    /web off         -> Desactivar búsqueda web")
                    print("    /web filter <x>  -> Activar con filtro (ej: /web filter site:edu)")
                continue

            app.responder_consulta(entrada)
            print("\n")
        except (KeyboardInterrupt, EOFError):
            break
