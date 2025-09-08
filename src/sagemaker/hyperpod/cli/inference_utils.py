import json
import click
from typing import Callable, Mapping, Type
from sagemaker.hyperpod.cli.common_utils import extract_version_from_args, get_latest_version, load_schema_for_version


def generate_click_command(
    *,
    schema_registry: Mapping[str, Type] = None,
    template_name: str = None,
    # Keep backward compatibility with old parameters
    schema_pkg: str = None,
    registry: Mapping[str, Type] = None,
) -> Callable:
    
    # Handle backward compatibility
    if registry is not None and schema_registry is None:
        schema_registry = registry
    if schema_pkg is not None and template_name is None:
        template_name = schema_pkg
    elif template_name is not None and schema_pkg is None:
        # Map template names to actual package names for schema loading
        if template_name == "hyp-jumpstart-endpoint":
            schema_pkg = "hyperpod_jumpstart_inference_template"
        elif template_name == "hyp-custom-endpoint":
            schema_pkg = "hyperpod_custom_inference_template"
        else:
            schema_pkg = template_name
            
    if schema_registry is None:
        raise ValueError("You must pass a schema_registry mapping version→Model")
    if template_name is None:
        raise ValueError("You must pass a template_name")

    # Defer registry access and version extraction until runtime
    def get_runtime_info():
        default_version = get_latest_version(schema_registry)
        version = extract_version_from_args(schema_registry, template_name, default_version)
        return default_version, version

    def decorator(func: Callable) -> Callable:
        # Parser for the single JSON‐dict env var flag
        def _parse_json_flag(ctx, param, value):
            if value is None:
                return None
            try:
                return json.loads(value)
            except json.JSONDecodeError as e:
                raise click.BadParameter(f"{param.name!r} must be valid JSON: {e}")

        # 1) the wrapper click actually invokes
        def wrapped_func(*args, **kwargs):
            # Get runtime info (this is when heavy operations happen)
            default_version, current_version = get_runtime_info()
            
            namespace = kwargs.pop("namespace", None)
            name = kwargs.pop("metadata_name", None)
            kwargs.pop("version", "1.0")  # Remove unused version

            Model = schema_registry.get(current_version)
            if Model is None:
                raise click.ClickException(f"Unsupported schema version: {current_version}")

            flat = Model(**kwargs)
            domain = flat.to_domain()
            return func(name, namespace, current_version, domain)

        # 2) inject click options from JSON Schema - HYBRID APPROACH
        # Load schema only when needed (for help or execution), but defer heavy operations
        def get_schema_and_options():
            default_version, version = get_runtime_info()
            schema = load_schema_for_version(version, schema_pkg)
            return schema

        # Add static JSON flags first (these don't require schema loading)
        json_flags = {
            "env": ("JSON object of environment variables, e.g. " '\'{"VAR1":"foo","VAR2":"bar"}\''),
            "dimensions": ("JSON object of dimensions, e.g. " '\'{"VAR1":"foo","VAR2":"bar"}\''),
            "resources_limits": ('JSON object of resource limits, e.g. \'{"cpu":"2","memory":"4Gi"}\''),
            "resources_requests": ('JSON object of resource requests, e.g. \'{"cpu":"1","memory":"2Gi"}\''),
        }

        for flag_name, help_text in json_flags.items():
            wrapped_func = click.option(
                f"--{flag_name.replace('_', '-')}",
                callback=_parse_json_flag,
                type=str,
                default=None,
                help=help_text,
                metavar="JSON",
            )(wrapped_func)

        excluded_props = set([
            "version",
            "env", 
            "dimensions",
            "resources_limits",
            "resources_requests",
        ])

        # HYBRID APPROACH: Load schema dynamically when options are needed
        # This happens during help generation or command execution, not at import time
        try:
            schema = get_schema_and_options()
            props = schema.get("properties", {})
            reqs = set(schema.get("required", []))

            # reverse so flags appear in the same order as in schema.json
            for name, spec in reversed(list(props.items())):
                if name in excluded_props:
                    continue

                # type inference
                if "enum" in spec:
                    ctype = click.Choice(spec["enum"])
                elif spec.get("type") == "integer":
                    ctype = int
                elif spec.get("type") == "number":
                    ctype = float
                elif spec.get("type") == "boolean":
                    ctype = bool
                else:
                    ctype = str

                wrapped_func = click.option(
                    f"--{name.replace('_','-')}",
                    required=(name in reqs),
                    default=spec.get("default", None),
                    show_default=("default" in spec),
                    type=ctype,
                    help=spec.get("description", ""),
                )(wrapped_func)
        except Exception:
            # If schema loading fails during import (e.g., missing dependencies),
            # still return the wrapped function with basic options
            # Schema will be loaded again during execution when it's actually needed
            pass

        return wrapped_func

    return decorator
