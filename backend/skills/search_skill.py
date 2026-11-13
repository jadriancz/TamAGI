"""Search files across the workspace by name or glob pattern.

This skill fills a gap: while Tama can read/write specific files, it had no way
to discover where files are located. Now she can ask questions like "find me
run.zig" or "find all .py files modified today" without needing to know paths.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from backend.skills.base import Skill, SkillResult
from backend.config import load_config


class SearchFileSkill(Skill):
    name = "search_file"
    description = (
        "Busca archivos en el sistema de archivos por nombre o patrón (glob). "
        "Devuelve ruta(s) ABSOLUTAS del primer hit o lista vacía si no encuentra. "
        "Usa glob como: '*.py', 'run.zig', '**/config.yaml'. "
        "El search está limitado a los directorios permitidos por guardrails. "
        "IMPORTANTE: Para leer un archivo encontrado, usa la RUTA ABSOLUTA exacta que te devuelva este skill en el skill 'read'."
    )
    parameters = {
        "pattern": {
            "type": "string",
            "description": "Patrón de búsqueda (ej: '*.py', 'run.zig', '**/notes/*.md')",
            "required": True,
        },
        "path": {
            "type": "string",
            "description": "Ruta raíz para buscar (por defecto: workspace actual)",
            "required": False,
        },
        "max_results": {
            "type": "number",
            "description": "Máximo número de resultados a devolver (por defecto: 50)",
            "required": False,
        },
    }

    async def execute(self, **kwargs: Any) -> SkillResult:
        import os
        
        pattern: str = kwargs.get("pattern", "")
        if not pattern:
            return SkillResult(success=False, error="El patrón de búsqueda es requerido")

        guardrails = load_config().guardrails
        allowed_paths = [Path(p).resolve() for p in guardrails.allowed_read_paths]

        # Determinar la raíz de búsqueda
        search_path = kwargs.get("path")
        if search_path:
            root = Path(search_path).resolve()
            # Verificar que esté dentro de allowed_paths
            if not any(str(root).startswith(str(ap)) for ap in allowed_paths):
                return SkillResult(
                    success=False, 
                    error=f"Ruta fuera de allowed_paths. Busca dentro de: {[str(p) for p in allowed_paths]}"
                )
        else:
            # Usar PATH de entorno como raíz predeterminada si está en allowed
            workspace_root = Path.cwd()
            for ap in allowed_paths:
                if str(ap) in str(workspace_root):
                    root = ap
                    break
            else:
                root = allowed_paths[0] if allowed_paths else Path.cwd()

        max_results = int(kwargs.get("max_results", 50))
        results: list[str] = []

        timeout = time.time() + 10  # timeout de seguridad
        try:
            for p in root.rglob(pattern):
                if time.time() > timeout:  # timeout alcanzado
                    break
                if p.is_file():
                    results.append(str(p))
                    if len(results) >= max_results:
                        break
        except PermissionError as e:
            return SkillResult(success=False, error=f"Permiso denegado: {e}")
        except Exception as e:
            return SkillResult(success=False, error=f"Error buscando: {e}")

        if not results:
            return SkillResult(success=True, output=f"No encontré '{pattern}'")

        lines = "\n".join(f"- {r}" for r in results[:max_results])
        return SkillResult(
            success=True,
            output=f"Encontré {len(results)} archivo(s):\n{lines}",
            data={"matches": results[:max_results], "count": len(results)},
        )