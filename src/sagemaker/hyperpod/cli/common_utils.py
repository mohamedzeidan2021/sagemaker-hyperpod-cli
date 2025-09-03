import sys
from typing import Mapping, Type
import click
import pkgutil
import json

JUMPSTART_SCHEMA = "hyperpod_jumpstart_inference_template"
CUSTOM_SCHEMA = "hyperpod_custom_inference_template"
JUMPSTART_COMMAND = "hyp-jumpstart-endpoint"
CUSTOM_COMMAND = "hyp-custom-endpoint"
PYTORCH_SCHEMA="hyperpod_pytorch_job_template"
PYTORCH_COMMAND="hyp-pytorch-job"

# Schema cache: (base_package, version) -> loaded schema dict
_SCHEMA_CACHE = {}


def extract_version_from_args(registry: Mapping[str, Type], schema_pkg: str, default: str) -> str:
    if "--version" not in sys.argv:
        return default

    idx = sys.argv.index("--version")
    if idx + 1 >= len(sys.argv):
        return default

    requested_version = sys.argv[idx + 1]
    invoked_command = next(
        (arg for arg in sys.argv if arg.startswith('hyp-')),
        None
    )

    # Check if schema validation is needed
    needs_validation = (
        (schema_pkg == JUMPSTART_SCHEMA and invoked_command == JUMPSTART_COMMAND) or
        (schema_pkg == CUSTOM_SCHEMA and invoked_command == CUSTOM_COMMAND) or
        (schema_pkg == PYTORCH_SCHEMA and invoked_command == PYTORCH_COMMAND)
    )

    if registry is not None and requested_version not in registry:
        if needs_validation:
                raise click.ClickException(f"Unsupported schema version: {requested_version}")
        else:
            return default

    return requested_version


def get_latest_version(registry: Mapping[str, Type]) -> str:
    """
    Get the latest version from the schema registry.
    """
    if not registry:
        raise ValueError("Schema registry is empty")

    # Sort versions and return the last (highest) one
    sorted_versions = sorted(registry.keys(), key=lambda v: [int(x) for x in v.split('.')])
    return sorted_versions[-1]


def load_schema_for_version(
    version: str,
    base_package: str,
) -> dict:
    """
    Load schema.json from the top-level <base_package>.vX_Y_Z package.
    Uses caching to avoid expensive re-imports: first load ~400ms, subsequent loads instant.
    """
    # Create cache key
    cache_key = (base_package, version)
    
    # Check if schema is already cached
    if cache_key in _SCHEMA_CACHE:
        return _SCHEMA_CACHE[cache_key]
    
    # Schema not cached, load it (this is the expensive operation)
    ver_pkg = f"{base_package}.v{version.replace('.', '_')}"
    raw = pkgutil.get_data(ver_pkg, "schema.json")
    if raw is None:
        raise click.ClickException(
            f"Could not load schema.json for version {version} "
            f"(looked in package {ver_pkg})"
        )
    
    # Parse JSON and cache the result
    schema = json.loads(raw)
    _SCHEMA_CACHE[cache_key] = schema
    
    return schema


def get_cached_schema(schema_registry: Mapping[str, Type], template_name: str, version: str) -> dict:
    """
    Get cached schema for the new unified API.
    Maps template names to base packages and uses existing caching mechanism.
    
    Args:
        schema_registry: Registry mapping versions to model classes
        template_name: Template name (e.g., "hyp-pytorch-job", "hyp-jumpstart-endpoint")
        version: Schema version
        
    Returns:
        Parsed schema dict
    """
    # Map template names to base packages
    template_to_package = {
        "hyp-pytorch-job": PYTORCH_SCHEMA,
        "hyp-jumpstart-endpoint": JUMPSTART_SCHEMA,
        "hyp-custom-endpoint": CUSTOM_SCHEMA,
    }
    
    base_package = template_to_package.get(template_name)
    if base_package is None:
        raise ValueError(f"Unknown template name: {template_name}")
    
    # Use existing caching mechanism
    return load_schema_for_version(version, base_package)
