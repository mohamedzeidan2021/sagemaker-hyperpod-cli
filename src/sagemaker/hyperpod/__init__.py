# Selective imports to avoid heavy dependencies during CLI startup
# Only import lightweight utility functions that are actually needed
from .common.utils import (
    create_boto3_client,
    region_to_az_ids, 
    get_aws_default_region,
    display_formatted_logs,
    parse_client_kubernetes_version,
    is_kubernetes_version_compatible,
    verify_kubernetes_version_compatibility,
)

# Import MonitoringConfig without triggering heavy observability imports  
from .observability.MonitoringConfig import MonitoringConfig
