"""Narzędzia dla modelu: zwykłe funkcje Pythona, schemat z sygnatury, walidacja pydantic.

Odrzucona operacja (ToolError) wraca do modelu jako tekst "BŁĄD: ...", więc model
może poprawić się w następnej rundzie zamiast przerywać turę.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import typing
from dataclasses import dataclass
from typing import Any, Callable, get_args, get_origin

from pydantic import BaseModel, Field, ValidationError, create_model
from pydantic.fields import FieldInfo


class ToolError(Exception):
    """Błąd merytoryczny narzędzia, który ma wrócić do modelu jako komunikat."""


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    fn: Callable[..., Any]
    model: type[BaseModel]

    def schema(self) -> dict[str, Any]:
        params = self.model.model_json_schema()
        params.pop("title", None)
        for prop in params.get("properties", {}).values():
            prop.pop("title", None)
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": params},
        }

    async def call(self, raw_args: dict[str, Any] | str | None) -> str:
        if isinstance(raw_args, str):
            try:
                raw_args = json.loads(raw_args) if raw_args.strip() else {}
            except json.JSONDecodeError:
                return f"BŁĄD: argumenty narzędzia {self.name} nie są poprawnym JSON."
        raw_args = dict(raw_args or {})
        try:
            args = self.model(**raw_args)
        except ValidationError as e:
            problems = "; ".join(f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors()[:4])
            return f"BŁĄD argumentów ({self.name}): {problems}. Oczekiwane pola: {list(self.model.model_fields)}"
        try:
            result = self.fn(**args.model_dump())
            if inspect.isawaitable(result):
                result = await result
        except ToolError as e:
            return f"BŁĄD: {e}"
        except Exception as e:  # noqa: BLE001 - błąd narzędzia nie może zabić tury agenta
            return f"BŁĄD wewnętrzny narzędzia {self.name}: {type(e).__name__}: {e}"
        if result is None:
            return "OK"
        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=False, default=str)


def _unwrap_annotation(annotation: Any) -> tuple[Any, FieldInfo | None]:
    if get_origin(annotation) is typing.Annotated:
        base, *extras = get_args(annotation)
        for extra in extras:
            if isinstance(extra, FieldInfo):
                return base, extra
        return base, None
    return annotation, None


def build_spec(fn: Callable[..., Any], *, name: str | None = None, description: str | None = None) -> ToolSpec:
    sig = inspect.signature(fn)
    hints = typing.get_type_hints(fn, include_extras=True)
    fields: dict[str, Any] = {}
    for pname, param in sig.parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        annotation = hints.get(pname, str)
        base, info = _unwrap_annotation(annotation)
        has_default = param.default is not inspect._empty
        # Osobna nazwa: wcześniej ta zmienna nadpisywała parametr `description` funkcji,
        # przez co opisem narzędzia stawał się opis jego ostatniego pola zamiast docstringu.
        field_desc = info.description if info is not None else None
        if has_default:
            info = Field(default=param.default, description=field_desc)
        else:
            info = Field(description=field_desc)
        fields[pname] = (base, info)
    model = create_model(f"{fn.__name__}_args", **fields)  # type: ignore[call-overload]
    doc = (description or inspect.getdoc(fn) or fn.__name__).strip()
    return ToolSpec(name=name or fn.__name__, description=doc, fn=fn, model=model)


def tool(fn: Callable[..., Any] | None = None, *, name: str | None = None, description: str | None = None):
    """Dekorator: zamienia funkcję w ToolSpec. Opis bierze z docstringa."""

    def wrap(f: Callable[..., Any]) -> ToolSpec:
        return build_spec(f, name=name, description=description)

    return wrap(fn) if fn is not None else wrap


class Toolbox:
    """Zbiór narzędzi jednego agenta z dispatchem po nazwie."""

    def __init__(self, specs: list[ToolSpec] | None = None):
        self._specs: dict[str, ToolSpec] = {}
        for s in specs or []:
            self.add(s)

    def add(self, spec: ToolSpec) -> None:
        self._specs[spec.name] = spec

    def extend(self, specs: list[ToolSpec]) -> None:
        for s in specs:
            self.add(s)

    def names(self) -> list[str]:
        return list(self._specs)

    def schemas(self) -> list[dict[str, Any]]:
        return [s.schema() for s in self._specs.values()]

    def has(self, name: str) -> bool:
        return name in self._specs

    async def call(self, name: str, raw_args: dict[str, Any] | str | None) -> str:
        spec = self._specs.get(name)
        if spec is None:
            return f"BŁĄD: nie ma narzędzia {name!r}. Dostępne: {', '.join(self.names())}"
        return await spec.call(raw_args)


async def run_sync(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    return await asyncio.to_thread(fn, *args, **kwargs)
