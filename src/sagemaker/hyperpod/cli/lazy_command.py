"""
LazyCommand implementation for deferred command loading in Click CLI.

This module provides a LazyCommand class that defers the import of command
implementations until they are actually needed, significantly improving
CLI startup performance.

Note: LazyGroup was removed as it provided minimal performance benefit
while adding unnecessary complexity. Standard Click groups are used instead.
"""

import click
from functools import cached_property
from importlib import import_module


class LazyCommand(click.Command):
    """
    A click Command that imports the actual implementation only when needed.
    
    Similar to LazyGroup but for individual commands.
    """

    def __init__(self, import_name, **kwargs):
        """
        Initialize LazyCommand with import specification.
        
        Args:
            import_name: String in format 'module.path:function_name'
                        specifying where to import the actual command function
            **kwargs: Additional arguments passed to click.Command
        """
        self._import_name = import_name
        # Don't set callback yet - we'll delegate everything to the real command
        super().__init__(**kwargs)

    @cached_property
    def _impl(self):
        """
        Lazily import and cache the actual command implementation.
        
        Note: @cached_property provides no benefit for CLI commands since each
        CLI invocation starts a new Python process. The performance improvement
        comes from DEFERRED IMPORTS - avoiding heavy dependency imports entirely
        for help commands, not from caching imports.
        
        Returns:
            The imported Click command object
        """
        module, name = self._import_name.split(':', 1)
        return getattr(import_module(module), name)

    def invoke(self, ctx):
        """Invoke the lazily-loaded command implementation."""
        return self._impl.invoke(ctx)

    def get_params(self, ctx):
        """Get parameters from the lazily-loaded implementation."""
        return self._impl.get_params(ctx)

    def get_usage(self, ctx):
        """Get usage from the lazily-loaded implementation."""
        return self._impl.get_usage(ctx)

    def get_help(self, ctx):
        """Get help from the lazily-loaded implementation."""
        return self._impl.get_help(ctx)

    def format_usage(self, ctx, formatter):
        """Format usage from the lazily-loaded implementation."""
        return self._impl.format_usage(ctx, formatter)

    def format_help(self, ctx, formatter):
        """Format help from the lazily-loaded implementation."""
        return self._impl.format_help(ctx, formatter)

    def parse_args(self, ctx, args):
        """Parse args using the lazily-loaded implementation."""
        return self._impl.parse_args(ctx, args)

    def make_context(self, info_name, args, parent=None, **extra):
        """Make context using the lazily-loaded implementation."""
        return self._impl.make_context(info_name, args, parent, **extra)
