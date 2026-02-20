#!/usr/bin/env python3
"""
Check if the MCP Memory Service HTTP server is running.

This script checks if the HTTP server is accessible (via Unix socket or TCP)
and provides helpful feedback to users about how to start it if not running.
"""

import sys
import os
import socket
import http.client
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
import json
import ssl


def _check_via_socket(socket_path: str, verbose: bool) -> bool:
    """
    Check server health via Unix domain socket.

    Args:
        socket_path: Path to the Unix domain socket file.
        verbose: If True, print detailed status messages.

    Returns:
        bool: True if server is reachable and healthy.
    """
    if not os.path.exists(socket_path):
        if verbose:
            print(f"[ERROR] Socket file not found: {socket_path}")
            print(f"\nTo start the server, run:")
            print(f"   uv run python scripts/server/run_http_server.py")
        return False

    class _UnixHTTPConnection(http.client.HTTPConnection):
        """HTTPConnection that connects through a Unix domain socket."""

        def __init__(self, socket_path: str):
            """Initialise with socket path instead of host:port."""
            super().__init__("localhost")
            self._socket_path = socket_path

        def connect(self):
            """Override connect to use Unix socket."""
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(3)
            sock.connect(self._socket_path)
            self.sock = sock

    try:
        conn = _UnixHTTPConnection(socket_path)
        conn.request("GET", "/api/health")
        resp = conn.getresponse()
        if resp.status == 200:
            data = json.loads(resp.read().decode("utf-8"))
            if verbose:
                print(f"[OK] Memory service is running (Unix socket)")
                print(f"   Socket: {socket_path}")
                print(f"   Version: {data.get('version', 'unknown')}")
                print(f"   Status: {data.get('status', 'unknown')}")
            return True
        else:
            if verbose:
                print(f"[WARN] Server responded with status {resp.status}")
            return False
    except (OSError, json.JSONDecodeError, http.client.HTTPException) as e:
        if verbose:
            print(f"[ERROR] Memory service is NOT running (socket check failed)")
            print(f"\nTo start the server, run:")
            print(f"   uv run python scripts/server/run_http_server.py")
            print(f"\nError: {str(e)}")
        return False


def check_http_server(verbose: bool = False) -> bool:
    """
    Check if the HTTP server is running (via Unix socket or TCP).

    Prefers Unix domain socket when MEMORY_SOCKET_ENABLED=true (default).
    Falls back to TCP check when HTTPS is enabled or socket is disabled.

    Args:
        verbose: If True, print detailed status messages

    Returns:
        bool: True if server is running, False otherwise
    """
    https_enabled = os.getenv('MCP_HTTPS_ENABLED', 'false').lower() == 'true'
    socket_enabled = os.getenv('MEMORY_SOCKET_ENABLED', 'true').lower() == 'true'
    socket_path = os.getenv('MEMORY_SOCKET_PATH', '/tmp/workflow/memory.sock')

    # Prefer Unix socket when not using HTTPS
    if socket_enabled and not https_enabled:
        return _check_via_socket(socket_path, verbose)

    # TCP fallback
    http_port = int(os.getenv('MCP_HTTP_PORT', '8000'))
    https_port = int(os.getenv('MCP_HTTPS_PORT', '8443'))
    endpoint = (
        f"https://localhost:{https_port}/api/health"
        if https_enabled
        else f"http://localhost:{http_port}/api/health"
    )

    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        req = Request(endpoint)
        with urlopen(req, timeout=3, context=ctx) as response:
            if response.status == 200:
                data = json.loads(response.read().decode('utf-8'))
                if verbose:
                    print("[OK] HTTP server is running")
                    print(f"   Version: {data.get('version', 'unknown')}")
                    print(f"   Endpoint: {endpoint}")
                    print(f"   Status: {data.get('status', 'unknown')}")
                return True
            else:
                if verbose:
                    print(f"[WARN] HTTP server responded with status {response.status}")
                return False
    except (URLError, HTTPError, json.JSONDecodeError) as e:
        if verbose:
            print("[ERROR] HTTP server is NOT running")
            print(f"\nTo start the HTTP server, run:")
            print(f"   uv run python scripts/server/run_http_server.py")
            print(f"\n   Or for HTTPS:")
            print(f"   MCP_HTTPS_ENABLED=true uv run python scripts/server/run_http_server.py")
            print(f"\nError: {str(e)}")
        return False


def main():
    """Main entry point for CLI usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Check if MCP Memory Service HTTP server is running"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Only return exit code (0=running, 1=not running), no output."
    )

    args = parser.parse_args()

    is_running = check_http_server(verbose=not args.quiet)
    sys.exit(0 if is_running else 1)


if __name__ == "__main__":
    main()
