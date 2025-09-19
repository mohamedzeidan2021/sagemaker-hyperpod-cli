import click
from typing import Union

# Import LazyCommand for deferred command loading
from sagemaker.hyperpod.cli.lazy_command import LazyCommand


@click.group(context_settings={'max_content_width': 200})
def get_package_version(package_name):
    # Defer heavy import until actually needed
    from importlib.metadata import version, PackageNotFoundError
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


@click.group(context_settings={'max_content_width': 200})
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


# Create command groups using standard Click groups (LazyGroup removed for simplicity)
@cli.group(cls=CLICommand, default_cmd='_default_create')
def create():
    """Create endpoints, pytorch jobs or cluster stacks.

If only used as 'hyp create' without [OPTIONS] COMMAND [ARGS] during init experience,
then it will validate configuration and render template files for deployment.
The generated files in the run directory can be used for actual deployment
to SageMaker HyperPod clusters or CloudFormation stacks.

Prerequisites for directly calling 'hyp create':
- Must be run in a directory initialized with 'hyp init'
- config.yaml and the appropriate template file must exist"""
    pass

@cli.group()
def list():
    """List endpoints, pytorch jobs or cluster stacks."""
    pass

@cli.group()
def describe():
    """Describe endpoints, pytorch jobs or cluster stacks."""
    pass

@cli.group()
def update():
    """Update an existing HyperPod cluster configuration."""
    pass

@cli.group()
def delete():
    """Delete endpoints or pytorch jobs."""
    pass

@cli.group(name='list-pods')
def list_pods():
    """List pods for endpoints or pytorch jobs."""
    pass

@cli.group(name='get-logs')
def get_logs():
    """Get pod logs for endpoints or pytorch jobs."""
    pass

@cli.group()
def invoke():
    """Invoke model endpoints."""
    pass

@cli.group(name='get-operator-logs')
def get_operator_logs():
    """Get operator logs for endpoints."""
    pass

@cli.group(name='exec')
def exec():
    """Execute commands in pods for endpoints or pytorch jobs."""
    pass


# Add initialization commands using lazy loading
cli.add_command(LazyCommand(
    import_name='sagemaker.hyperpod.cli.commands.init:init',
    name='init',
    help='Initialize a new HyperPod project directory.'
))

cli.add_command(LazyCommand(
    import_name='sagemaker.hyperpod.cli.commands.init:reset',
    name='reset',
    help='Reset the current HyperPod project configuration.'
))

cli.add_command(LazyCommand(
    import_name='sagemaker.hyperpod.cli.commands.init:configure',
    name='configure',
    help='Configure HyperPod CLI settings.'
))

cli.add_command(LazyCommand(
    import_name='sagemaker.hyperpod.cli.commands.init:validate',
    name='validate',
    help='Validate the current HyperPod project configuration.'
))

# Add cluster management commands using lazy loading
cli.add_command(LazyCommand(
    import_name='sagemaker.hyperpod.cli.commands.cluster:list_cluster',
    name='list-cluster',
    help='List available HyperPod clusters.'
))

cli.add_command(LazyCommand(
    import_name='sagemaker.hyperpod.cli.commands.cluster:set_cluster_context',
    name='set-cluster-context',
    help='Set the current cluster context.'
))

cli.add_command(LazyCommand(
    import_name='sagemaker.hyperpod.cli.commands.cluster:get_cluster_context',
    name='get-cluster-context',
    help='Get the current cluster context.'
))

cli.add_command(LazyCommand(
    import_name='sagemaker.hyperpod.cli.commands.cluster:get_monitoring',
    name='get-monitoring',
    help='Get cluster monitoring information.'
))

# Add training commands to create group using lazy loading
create.add_command(LazyCommand(
    import_name='sagemaker.hyperpod.cli.commands.training:pytorch_create',
    name='hyp-pytorch-job'
))

# Add inference commands to create group using lazy loading
create.add_command(LazyCommand(
    import_name='sagemaker.hyperpod.cli.commands.inference:js_create',
    name='hyp-jumpstart-endpoint'
))

create.add_command(LazyCommand(
    import_name='sagemaker.hyperpod.cli.commands.inference:custom_create',
    name='hyp-custom-endpoint'
))

# Add default create command for init experience
_default_create_cmd = LazyCommand(
    import_name='sagemaker.hyperpod.cli.commands.init:_default_create',
    name='_default_create'
)
_default_create_cmd.hidden = True
create.add_command(_default_create_cmd)

# Add training commands to list group using lazy loading
list.add_command(LazyCommand(
    import_name='sagemaker.hyperpod.cli.commands.training:list_jobs',
    name='hyp-pytorch-job'
))

# Add inference commands to list group using lazy loading
list.add_command(LazyCommand(
    import_name='sagemaker.hyperpod.cli.commands.inference:js_list',
    name='hyp-jumpstart-endpoint'
))

list.add_command(LazyCommand(
    import_name='sagemaker.hyperpod.cli.commands.inference:custom_list',
    name='hyp-custom-endpoint'
))

# Add cluster stack commands to list group using lazy loading
list.add_command(LazyCommand(
    import_name='sagemaker.hyperpod.cli.commands.cluster_stack:list_cluster_stacks',
    name='cluster-stacks'
))

# Add training commands to describe group using lazy loading
describe.add_command(LazyCommand(
    import_name='sagemaker.hyperpod.cli.commands.training:pytorch_describe',
    name='hyp-pytorch-job'
))

# Add inference commands to describe group using lazy loading
describe.add_command(LazyCommand(
    import_name='sagemaker.hyperpod.cli.commands.inference:js_describe',
    name='hyp-jumpstart-endpoint'
))

describe.add_command(LazyCommand(
    import_name='sagemaker.hyperpod.cli.commands.inference:custom_describe',
    name='hyp-custom-endpoint'
))

# Add cluster stack commands to describe group using lazy loading
describe.add_command(LazyCommand(
    import_name='sagemaker.hyperpod.cli.commands.cluster_stack:describe_cluster_stack',
    name='cluster-stack'
))

# Add cluster update command using lazy loading
update.add_command(LazyCommand(
    import_name='sagemaker.hyperpod.cli.commands.cluster_stack:update_cluster',
    name='cluster'
))

# Add training commands to delete group using lazy loading
delete.add_command(LazyCommand(
    import_name='sagemaker.hyperpod.cli.commands.training:pytorch_delete',
    name='hyp-pytorch-job'
))

# Add inference commands to delete group using lazy loading
delete.add_command(LazyCommand(
    import_name='sagemaker.hyperpod.cli.commands.inference:js_delete',
    name='hyp-jumpstart-endpoint'
))

delete.add_command(LazyCommand(
    import_name='sagemaker.hyperpod.cli.commands.inference:custom_delete',
    name='hyp-custom-endpoint'
))

# Add cluster stack commands to delete group using lazy loading
delete.add_command(LazyCommand(
    import_name='sagemaker.hyperpod.cli.commands.cluster_stack:delete_cluster_stack',
    name='cluster-stack'
))

# Add training commands to list_pods group using lazy loading
list_pods.add_command(LazyCommand(
    import_name='sagemaker.hyperpod.cli.commands.training:pytorch_list_pods',
    name='hyp-pytorch-job'
))

# Add inference commands to list_pods group using lazy loading
list_pods.add_command(LazyCommand(
    import_name='sagemaker.hyperpod.cli.commands.inference:js_list_pods',
    name='hyp-jumpstart-endpoint'
))

list_pods.add_command(LazyCommand(
    import_name='sagemaker.hyperpod.cli.commands.inference:custom_list_pods',
    name='hyp-custom-endpoint'
))

# Add training commands to get_logs group using lazy loading
get_logs.add_command(LazyCommand(
    import_name='sagemaker.hyperpod.cli.commands.training:pytorch_get_logs',
    name='hyp-pytorch-job'
))

# Add inference commands to get_logs group using lazy loading
get_logs.add_command(LazyCommand(
    import_name='sagemaker.hyperpod.cli.commands.inference:js_get_logs',
    name='hyp-jumpstart-endpoint'
))

get_logs.add_command(LazyCommand(
    import_name='sagemaker.hyperpod.cli.commands.inference:custom_get_logs',
    name='hyp-custom-endpoint'
))

# Add training commands to get_operator_logs group using lazy loading
get_operator_logs.add_command(LazyCommand(
    import_name='sagemaker.hyperpod.cli.commands.training:pytorch_get_operator_logs',
    name='hyp-pytorch-job'
))

# Add inference commands to get_operator_logs group using lazy loading
get_operator_logs.add_command(LazyCommand(
    import_name='sagemaker.hyperpod.cli.commands.inference:js_get_operator_logs',
    name='hyp-jumpstart-endpoint'
))

get_operator_logs.add_command(LazyCommand(
    import_name='sagemaker.hyperpod.cli.commands.inference:custom_get_operator_logs',
    name='hyp-custom-endpoint'
))

# Add inference commands to invoke group using lazy loading
invoke.add_command(LazyCommand(
    import_name='sagemaker.hyperpod.cli.commands.inference:custom_invoke',
    name='hyp-custom-endpoint'
))

invoke.add_command(LazyCommand(
    import_name='sagemaker.hyperpod.cli.commands.inference:custom_invoke',
    name='hyp-jumpstart-endpoint'
))

# Add training commands to exec group using lazy loading
exec.add_command(LazyCommand(
    import_name='sagemaker.hyperpod.cli.commands.training:pytorch_exec',
    name='hyp-pytorch-job'
))

if __name__ == "__main__":
    cli()
