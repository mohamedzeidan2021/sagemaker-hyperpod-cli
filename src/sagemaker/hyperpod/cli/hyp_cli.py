import click
import yaml
import json
import os
import subprocess
from functools import partial
from pydantic import BaseModel, ValidationError, Field
from typing import Optional, Union
from importlib.metadata import version, PackageNotFoundError

# Lazy loading infrastructure
from sagemaker.hyperpod.common.lazy_loading import (
    create_lazy_group, 
    create_lazy_cli_command,
    create_lazy_top_level_cli
)

# ALL command imports removed - now using complete lazy loading


@click.group(context_settings={'max_content_width': 200})
def get_package_version(package_name):
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "Not installed"

def print_version(ctx, param, value):
    if not value or ctx.resilient_parsing:
        return

    hyp_version = get_package_version("sagemaker-hyperpod")
    pytorch_template_version = get_package_version("hyperpod-pytorch-job-template")
    custom_inference_version = get_package_version("hyperpod-custom-inference-template")
    jumpstart_inference_version = get_package_version("hyperpod-jumpstart-inference-template")

    click.echo(f"hyp version: {hyp_version}")
    click.echo(f"hyperpod-pytorch-job-template version: {pytorch_template_version}")
    click.echo(f"hyperpod-custom-inference-template version: {custom_inference_version}")
    click.echo(f"hyperpod-jumpstart-inference-template version: {jumpstart_inference_version}")
    ctx.exit()


# Main CLI with complete lazy loading - handles both subgroups and top-level commands
@click.group(
    cls=create_lazy_top_level_cli,
    context_settings={'max_content_width': 200}
)
@click.option('--version', is_flag=True, callback=print_version, expose_value=False, is_eager=True, help='Show version information')
def cli():
    pass


class CLICommand(click.Group):
    def __init__(self, *args, default_cmd: Union[str, None] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_cmd = default_cmd

    def parse_args(self, ctx, args):
        # Only inject default subcommand when:
        #  - user didn't name a subcommand, and
        #  - user didn't ask for help
        if self.default_cmd:
            # any non-flag token that is a known subcommand?
            has_subcmd = any((not a.startswith("-")) and (a in self.commands) for a in args)
            asked_for_help = any(a in ("-h", "--help") for a in args)
            if (not has_subcmd) and (not asked_for_help):
                args = [self.default_cmd] + args
        return super().parse_args(ctx, args)


# Lazy loading create group - imports commands only when executed, not for help
# Uses LazyCLICommand to preserve default_cmd='_default_create' behavior
LazyCreateGroup = partial(create_lazy_cli_command, 'create')

@cli.group(cls=LazyCreateGroup)
def create():
    """
    Create endpoints, pytorch jobs or cluster stacks.

    If only used as 'hyp create' without [OPTIONS] COMMAND [ARGS] during init experience,
    then it will validate configuration and render template files for deployment.
    The generated files in the run directory can be used for actual deployment
    to SageMaker HyperPod clusters or CloudFormation stacks.

    Prerequisites for directly calling 'hyp create':
    - Must be run in a directory initialized with 'hyp init'
    - config.yaml and the appropriate template file must exist
    """
    pass


# Lazy loading list group - imports commands only when executed, not for help
LazyListGroup = partial(create_lazy_group, 'list')

@cli.group(cls=LazyListGroup)
def list():
    """List endpoints, pytorch jobs or cluster stacks."""
    pass


# Lazy loading describe group - imports commands only when executed, not for help
LazyDescribeGroup = partial(create_lazy_group, 'describe')

@cli.group(cls=LazyDescribeGroup)
def describe():
    """Describe endpoints, pytorch jobs or cluster stacks."""
    pass


# Lazy loading delete group - imports commands only when executed, not for help
LazyDeleteGroup = partial(create_lazy_group, 'delete')

@cli.group(cls=LazyDeleteGroup)
def delete():
    """Delete endpoints or pytorch jobs."""
    pass


# Lazy loading update group - imports commands only when executed, not for help
LazyUpdateGroup = partial(create_lazy_group, 'update')

@cli.group(cls=LazyUpdateGroup)
def update():
    """Update an existing HyperPod cluster configuration."""
    pass


# Lazy loading list_pods group - imports commands only when executed, not for help
LazyListPodsGroup = partial(create_lazy_group, 'list_pods')

@cli.group(cls=LazyListPodsGroup)
def list_pods():
    """List pods for endpoints or pytorch jobs."""
    pass


# Lazy loading get_logs group - imports commands only when executed, not for help
LazyGetLogsGroup = partial(create_lazy_group, 'get_logs')

@cli.group(cls=LazyGetLogsGroup)
def get_logs():
    """Get pod logs for endpoints or pytorch jobs."""
    pass


# Lazy loading invoke group - imports commands only when executed, not for help
LazyInvokeGroup = partial(create_lazy_group, 'invoke')

@cli.group(cls=LazyInvokeGroup)
def invoke():
    """Invoke model endpoints."""
    pass


# Lazy loading get_operator_logs group - imports commands only when executed, not for help
LazyGetOperatorLogsGroup = partial(create_lazy_group, 'get_operator_logs')

@cli.group(cls=LazyGetOperatorLogsGroup)
def get_operator_logs():
    """Get operator logs for endpoints."""
    pass


# Lazy loading exec group - imports commands only when executed, not for help
LazyExecGroup = partial(create_lazy_group, 'exec')

@cli.group(cls=LazyExecGroup)
def exec():
    """Execute commands in pods for endpoints or pytorch jobs."""
    pass


# ALL COMMANDS NOW USE COMPLETE LAZY LOADING!
# No manual add_command calls needed - everything auto-discovered from registry:
# 
# Top-level commands (lazy loaded from registry):
# - init, reset, configure, validate
# - list-cluster, set-cluster-context, get-cluster-context, get-monitoring
#
# Groups (lazy loaded from registry):
# - create, list, describe, delete, update 
# - list-pods, get-logs, invoke, get-operator-logs, exec
#
# All 36+ commands across all groups now use lazy loading!


if __name__ == "__main__":
    cli()
