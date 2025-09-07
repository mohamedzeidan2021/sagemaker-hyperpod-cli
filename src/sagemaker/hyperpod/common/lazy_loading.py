"""
Lazy loading infrastructure for SageMaker HyperPod CLI commands.

This module provides the core infrastructure for deferred command imports,
enabling faster CLI startup times by only loading command modules when
they are actually executed (not when generating help text).
"""

import importlib
import logging
from typing import Dict, Any, Optional, List
import click

logger = logging.getLogger(__name__)


# Command Registry - Static definitions with no imports
COMMAND_REGISTRY = {
    # Subgroup definitions - help text for main CLI groups
    'subgroups': {
        'create': 'Create endpoints, pytorch jobs or cluster stacks.',
        'list': 'List endpoints, pytorch jobs or cluster stacks.',
        'describe': 'Describe endpoints, pytorch jobs or cluster stacks.',
        'delete': 'Delete endpoints or pytorch jobs.',
        'update': 'Update an existing HyperPod cluster configuration.',
        'list-pods': 'List pods for endpoints or pytorch jobs.',
        'get-logs': 'Get pod logs for endpoints or pytorch jobs.',
        'invoke': 'Invoke model endpoints.',
        'get-operator-logs': 'Get operator logs for endpoints.',
        'exec': 'Execute commands in pods for endpoints or pytorch jobs.'
    },
    
    # List subcommands
    'list': {
        'hyp-pytorch-job': {
            'module': 'sagemaker.hyperpod.cli.commands.training',
            'function': 'list_jobs',
            'help': 'List PyTorch training jobs'
        },
        'hyp-jumpstart-endpoint': {
            'module': 'sagemaker.hyperpod.cli.commands.inference',
            'function': 'js_list',
            'help': 'List JumpStart endpoints'
        },
        'hyp-custom-endpoint': {
            'module': 'sagemaker.hyperpod.cli.commands.inference',
            'function': 'custom_list',
            'help': 'List custom endpoints'
        },
        'cluster-stack': {
            'module': 'sagemaker.hyperpod.cli.commands.cluster_stack',
            'function': 'list_cluster_stacks',
            'help': 'List cluster stacks'
        }
    },
    
    # Create subcommands
    'create': {
        'hyp-pytorch-job': {
            'module': 'sagemaker.hyperpod.cli.commands.training',
            'function': 'pytorch_create',
            'help': 'Create a PyTorch training job'
        },
        'hyp-jumpstart-endpoint': {
            'module': 'sagemaker.hyperpod.cli.commands.inference',
            'function': 'js_create',
            'help': 'Create a JumpStart inference endpoint'
        },
        'hyp-custom-endpoint': {
            'module': 'sagemaker.hyperpod.cli.commands.inference',
            'function': 'custom_create',
            'help': 'Create a custom inference endpoint'
        },
        'cluster-stack': {
            'module': 'sagemaker.hyperpod.cli.commands.cluster_stack',
            'function': 'create_cluster_stack',
            'help': 'Create a new HyperPod cluster stack'
        },
        '_default_create': {
            'module': 'sagemaker.hyperpod.cli.commands.init',
            'function': '_default_create',
            'help': 'Default create command',
            'hidden': True
        }
    },
    
    # Describe subcommands
    'describe': {
        'hyp-pytorch-job': {
            'module': 'sagemaker.hyperpod.cli.commands.training',
            'function': 'pytorch_describe',
            'help': 'Describe a PyTorch training job'
        },
        'hyp-jumpstart-endpoint': {
            'module': 'sagemaker.hyperpod.cli.commands.inference',
            'function': 'js_describe',
            'help': 'Describe a JumpStart inference endpoint'
        },
        'hyp-custom-endpoint': {
            'module': 'sagemaker.hyperpod.cli.commands.inference',
            'function': 'custom_describe',
            'help': 'Describe a custom inference endpoint'
        },
        'cluster-stack': {
            'module': 'sagemaker.hyperpod.cli.commands.cluster_stack',
            'function': 'describe_cluster_stack',
            'help': 'Describe a HyperPod cluster stack'
        }
    },
    
    # Delete subcommands
    'delete': {
        'hyp-pytorch-job': {
            'module': 'sagemaker.hyperpod.cli.commands.training',
            'function': 'pytorch_delete',
            'help': 'Delete a PyTorch training job'
        },
        'hyp-jumpstart-endpoint': {
            'module': 'sagemaker.hyperpod.cli.commands.inference',
            'function': 'js_delete',
            'help': 'Delete a JumpStart inference endpoint'
        },
        'hyp-custom-endpoint': {
            'module': 'sagemaker.hyperpod.cli.commands.inference',
            'function': 'custom_delete',
            'help': 'Delete a custom inference endpoint'
        }
    },
    
    # Update subcommands
    'update': {
        'cluster': {
            'module': 'sagemaker.hyperpod.cli.commands.cluster_stack',
            'function': 'update_cluster',
            'help': 'Update an existing HyperPod cluster configuration'
        }
    },
    
    # List-pods subcommands
    'list_pods': {
        'hyp-pytorch-job': {
            'module': 'sagemaker.hyperpod.cli.commands.training',
            'function': 'pytorch_list_pods',
            'help': 'List pods for PyTorch training jobs'
        },
        'hyp-jumpstart-endpoint': {
            'module': 'sagemaker.hyperpod.cli.commands.inference',
            'function': 'js_list_pods',
            'help': 'List pods for JumpStart inference endpoints'
        },
        'hyp-custom-endpoint': {
            'module': 'sagemaker.hyperpod.cli.commands.inference',
            'function': 'custom_list_pods',
            'help': 'List pods for custom inference endpoints'
        }
    },
    
    # Get-logs subcommands
    'get_logs': {
        'hyp-pytorch-job': {
            'module': 'sagemaker.hyperpod.cli.commands.training',
            'function': 'pytorch_get_logs',
            'help': 'Get logs for PyTorch training job pods'
        },
        'hyp-jumpstart-endpoint': {
            'module': 'sagemaker.hyperpod.cli.commands.inference',
            'function': 'js_get_logs',
            'help': 'Get logs for JumpStart inference endpoint pods'
        },
        'hyp-custom-endpoint': {
            'module': 'sagemaker.hyperpod.cli.commands.inference',
            'function': 'custom_get_logs',
            'help': 'Get logs for custom inference endpoint pods'
        }
    },
    
    # Get-operator-logs subcommands
    'get_operator_logs': {
        'hyp-pytorch-job': {
            'module': 'sagemaker.hyperpod.cli.commands.training',
            'function': 'pytorch_get_operator_logs',
            'help': 'Get operator logs for PyTorch training jobs'
        },
        'hyp-jumpstart-endpoint': {
            'module': 'sagemaker.hyperpod.cli.commands.inference',
            'function': 'js_get_operator_logs',
            'help': 'Get operator logs for JumpStart inference endpoints'
        },
        'hyp-custom-endpoint': {
            'module': 'sagemaker.hyperpod.cli.commands.inference',
            'function': 'custom_get_operator_logs',
            'help': 'Get operator logs for custom inference endpoints'
        }
    },
    
    # Invoke subcommands
    'invoke': {
        'hyp-custom-endpoint': {
            'module': 'sagemaker.hyperpod.cli.commands.inference',
            'function': 'custom_invoke',
            'help': 'Invoke custom inference endpoints'
        },
        'hyp-jumpstart-endpoint': {
            'module': 'sagemaker.hyperpod.cli.commands.inference',
            'function': 'custom_invoke',
            'help': 'Invoke JumpStart inference endpoints'
        }
    },
    
    # Exec subcommands
    'exec': {
        'hyp-pytorch-job': {
            'module': 'sagemaker.hyperpod.cli.commands.training',
            'function': 'pytorch_exec',
            'help': 'Execute commands in PyTorch training job pods'
        }
    },
    
    # Top-level commands - now fully implemented with lazy loading
    'top_level': {
        'list-cluster': {
            'module': 'sagemaker.hyperpod.cli.commands.cluster',
            'function': 'list_cluster',
            'help': 'List available SageMaker HyperPod clusters'
        },
        'set-cluster-context': {
            'module': 'sagemaker.hyperpod.cli.commands.cluster',
            'function': 'set_cluster_context',
            'help': 'Configure kubectl for HyperPod cluster'
        },
        'get-cluster-context': {
            'module': 'sagemaker.hyperpod.cli.commands.cluster',
            'function': 'get_cluster_context',
            'help': 'Get current cluster context'
        },
        'get-monitoring': {
            'module': 'sagemaker.hyperpod.cli.commands.cluster',
            'function': 'get_monitoring',
            'help': 'Get monitoring configuration'
        },
        'init': {
            'module': 'sagemaker.hyperpod.cli.commands.init',
            'function': 'init',
            'help': 'Initialize HyperPod configuration'
        },
        'reset': {
            'module': 'sagemaker.hyperpod.cli.commands.init',
            'function': 'reset',
            'help': 'Reset HyperPod configuration'
        },
        'configure': {
            'module': 'sagemaker.hyperpod.cli.commands.init',
            'function': 'configure',
            'help': 'Configure HyperPod settings'
        },
        'validate': {
            'module': 'sagemaker.hyperpod.cli.commands.init',
            'function': 'validate',
            'help': 'Validate HyperPod configuration'
        }
    }
}

# Group settings for special CLI behaviors
GROUP_SETTINGS = {
    'create': {'default_cmd': '_default_create'},
    # Other groups don't have default commands
}


def _load_command(module_path: str, function_name: str) -> Optional[click.Command]:
    """
    Import and return command function only when needed.
    
    Args:
        module_path: Python module path (e.g., 'sagemaker.hyperpod.cli.commands.training')
        function_name: Function name within the module (e.g., 'list_jobs')
        
    Returns:
        Click command object or None if import fails
        
    Raises:
        click.ClickException: If module import fails with user-friendly error message
    """
    try:
        logger.debug(f"Lazy loading command: {module_path}.{function_name}")
        module = importlib.import_module(module_path)
        command_function = getattr(module, function_name)
        logger.debug(f"Successfully loaded command: {function_name}")
        return command_function
    except ImportError as e:
        error_msg = f"Failed to import module '{module_path}': {e}"
        logger.error(error_msg)
        raise click.ClickException(f"Command unavailable: {error_msg}")
    except AttributeError as e:
        error_msg = f"Function '{function_name}' not found in module '{module_path}': {e}"
        logger.error(error_msg)
        raise click.ClickException(f"Command unavailable: {error_msg}")
    except Exception as e:
        error_msg = f"Unexpected error loading command '{function_name}': {e}"
        logger.error(error_msg)
        raise click.ClickException(f"Command unavailable: {error_msg}")


class LazyGroup(click.Group):
    """
    Lazy loading Click Group that only imports command modules when commands are executed.
    
    This class provides the core lazy loading functionality:
    1. list_commands() returns command names without importing modules
    2. get_command() imports modules only when commands are actually invoked
    3. format_commands() shows help using static text without imports
    """
    
    def __init__(self, registry_key: str, *args, **kwargs):
        """
        Initialize LazyGroup with a registry key.
        
        Args:
            registry_key: Key in COMMAND_REGISTRY (e.g., 'list', 'create')
        """
        super().__init__(*args, **kwargs)
        self.registry_key = registry_key
    
    def list_commands(self, ctx: click.Context) -> List[str]:
        """
        Return command names without importing modules.
        
        Args:
            ctx: Click context
            
        Returns:
            Sorted list of command names
        """
        commands = COMMAND_REGISTRY.get(self.registry_key, {})
        # Filter out hidden commands
        visible_commands = [
            name for name, info in commands.items()
            if not info.get('hidden', False)
        ]
        return sorted(visible_commands)
    
    def get_command(self, ctx: click.Context, name: str) -> Optional[click.Command]:
        """
        Import command only when actually invoked (not for help).
        
        Args:
            ctx: Click context
            name: Command name to load
            
        Returns:
            Click command object or None if not found
        """
        commands = COMMAND_REGISTRY.get(self.registry_key, {})
        if name in commands:
            cmd_info = commands[name]
            return _load_command(cmd_info['module'], cmd_info['function'])
        return None
    
    def format_commands(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        """
        Show help using static text - no imports needed.
        
        Args:
            ctx: Click context
            formatter: Click help formatter
        """
        commands = []
        registry_commands = COMMAND_REGISTRY.get(self.registry_key, {})
        
        for name, info in registry_commands.items():
            # Skip hidden commands in help display
            if not info.get('hidden', False):
                commands.append((name, info['help']))
        
        if commands:
            with formatter.section('Commands'):
                formatter.write_dl(commands)


class LazyCLICommand(LazyGroup):
    """
    Lazy loading version of CLICommand that combines lazy loading with default command support.
    
    This class extends LazyGroup to support the existing CLICommand functionality
    like default_cmd parameter while maintaining lazy loading benefits.
    """
    
    def __init__(self, registry_key: str, default_cmd: Optional[str] = None, *args, **kwargs):
        """
        Initialize LazyCLICommand with lazy loading and default command support.
        
        Args:
            registry_key: Key in COMMAND_REGISTRY
            default_cmd: Default command to use when no subcommand is specified
        """
        super().__init__(registry_key, *args, **kwargs)
        self.default_cmd = default_cmd or GROUP_SETTINGS.get(registry_key, {}).get('default_cmd')
    
    def parse_args(self, ctx: click.Context, args: List[str]) -> List[str]:
        """
        Parse args and inject default subcommand when needed (preserves CLICommand behavior).
        
        Args:
            ctx: Click context
            args: Command line arguments
            
        Returns:
            Processed arguments list
        """
        # Only inject default subcommand when:
        #  - user didn't name a subcommand, and
        #  - user didn't ask for help
        if self.default_cmd:
            # Check if any non-flag token is a known subcommand
            has_subcmd = any(
                (not a.startswith("-")) and (a in self.list_commands(ctx)) 
                for a in args
            )
            asked_for_help = any(a in ("-h", "--help") for a in args)
            
            if (not has_subcmd) and (not asked_for_help):
                args = [self.default_cmd] + args
        
        return super().parse_args(ctx, args)


def create_lazy_group(registry_key: str, **kwargs) -> LazyGroup:
    """
    Factory function to create LazyGroup instances.
    
    Args:
        registry_key: Key in COMMAND_REGISTRY
        **kwargs: Additional arguments passed to LazyGroup
        
    Returns:
        LazyGroup instance configured for the specified registry key
    """
    return LazyGroup(registry_key, **kwargs)


class LazyTopLevelCLI(click.Group):
    """
    Special lazy group for top-level CLI that handles both subgroups and individual commands.
    """
    
    def list_commands(self, ctx: click.Context) -> List[str]:
        """Return both subgroups and individual top-level commands."""
        # Subgroups defined in CLI
        subgroups = ['create', 'list', 'describe', 'delete', 'update', 
                     'list-pods', 'get-logs', 'invoke', 'get-operator-logs', 'exec']
        
        # Individual top-level commands from registry
        top_level_commands = list(COMMAND_REGISTRY.get('top_level', {}).keys())
        
        return sorted(subgroups + top_level_commands)
    
    def get_command(self, ctx: click.Context, name: str) -> Optional[click.Command]:
        """Get command from top_level registry or delegate to subgroups."""
        # Check if it's a top-level command
        top_level_commands = COMMAND_REGISTRY.get('top_level', {})
        if name in top_level_commands:
            cmd_info = top_level_commands[name]
            return _load_command(cmd_info['module'], cmd_info['function'])
        
        # Otherwise, let Click handle the subgroups normally
        return super().get_command(ctx, name)
    
    def format_commands(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        """
        Show help using static text from registry - no imports needed.
        
        This is the key fix for the help performance issue. Instead of calling
        get_command() for every command (which triggers lazy imports), we use
        static help text from the COMMAND_REGISTRY.
        
        Args:
            ctx: Click context
            formatter: Click help formatter
        """
        commands = []
        
        # Add subgroups from registry
        subgroups = ['create', 'list', 'describe', 'delete', 'update', 
                     'list-pods', 'get-logs', 'invoke', 'get-operator-logs', 'exec']
        subgroup_help = COMMAND_REGISTRY.get('subgroups', {})
        
        for subgroup in subgroups:
            help_text = subgroup_help.get(subgroup, f'{subgroup.title()} commands')
            commands.append((subgroup, help_text))
        
        # Add individual top-level commands from registry
        top_level_commands = COMMAND_REGISTRY.get('top_level', {})
        for name, info in top_level_commands.items():
            commands.append((name, info['help']))
        
        # Sort commands alphabetically
        commands.sort(key=lambda x: x[0])
        
        if commands:
            with formatter.section('Commands'):
                formatter.write_dl(commands)


def create_lazy_cli_command(registry_key: str, **kwargs) -> LazyCLICommand:
    """
    Factory function to create LazyCLICommand instances.
    
    Args:
        registry_key: Key in COMMAND_REGISTRY  
        **kwargs: Additional arguments passed to LazyCLICommand
        
    Returns:
        LazyCLICommand instance configured for the specified registry key
    """
    return LazyCLICommand(registry_key, **kwargs)


def create_lazy_top_level_cli(**kwargs) -> LazyTopLevelCLI:
    """
    Factory function to create LazyTopLevelCLI instances.
    
    Returns:
        LazyTopLevelCLI instance for the main CLI
    """
    return LazyTopLevelCLI(**kwargs)
