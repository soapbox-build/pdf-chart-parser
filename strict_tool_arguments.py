"""A tool argument no signature declares is REFUSED by name, not silently dropped.

COPIED, NOT SHARED: byte-identical in overture-mcp and xlsx-templater-mcp, the two servers
on the `mcp` SDK's FastMCP (servers may not import across directories). The TypeScript
servers' equivalent is servers/_shared/strict-tool-arguments.ts; the fastmcp 4.x servers
refuse by themselves.

WHY. mcp 1.29's FastMCP validates arguments with a pydantic model per tool whose config
leaves `extra` at pydantic's default, 'ignore', so an undeclared or misspelled argument
vanished before the function ran. Measured by scripts/refuse-contract (2026-09-28): all
eight overture-mcp tools and all three xlsx-templater-mcp tools. A missing-required error
only LOOKED like a refusal, because pydantic echoes the whole input in input_value.

WHAT. strict_tool_arguments(mcp), called once after every tool is registered, swaps each
tool's argument model for a subclass with extra='forbid'. pydantic then refuses the call
naming the key ('Extra inputs are not permitted', loc = the key). Only the top level is
made strict, as in the TypeScript helper. It raises rather than returning quietly when it
finds nothing to make strict, so an SDK change that moves the tools cannot turn it into a
no-op.
"""

from __future__ import annotations

from pydantic import ConfigDict


def strict_tool_arguments(mcp) -> int:
    tools = mcp._tool_manager.list_tools()
    if not tools:
        raise RuntimeError("strict_tool_arguments: no tools registered; call it after registering them")
    for tool in tools:
        base = tool.fn_metadata.arg_model
        strict = type(base.__name__, (base,), {
            "__module__": base.__module__,
            "model_config": ConfigDict(**base.model_config, extra="forbid"),
        })
        tool.fn_metadata.arg_model = strict
    return len(tools)
