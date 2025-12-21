# Copyright 2024 Heinrich Krupp
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
import json
from typing import Any, Dict, List, Optional

def generate_content_hash(
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None
) -> str:
    """
    Generate a unique hash for content, metadata, and project scope.

    This improved version ensures consistent hashing by:
    1. Normalizing content (strip whitespace, lowercase)
    2. Including project tag for cross-project uniqueness
    3. Sorting metadata keys
    4. Using a consistent JSON serialization

    The project tag (project:X) is included in the hash to allow identical
    content to exist in different projects without UNIQUE constraint conflicts.
    """
    # Normalize content
    normalized_content = content.strip().lower()

    # Create hash content with normalized content
    hash_content = normalized_content

    # Add scope tag if present (allows same content in different projects/governance)
    # Priority: governance:core > project:X
    if tags:
        governance_tags = [t for t in tags if t.startswith('governance:')]
        project_tags = sorted([t for t in tags if t.startswith('project:')])

        if governance_tags:
            # Governance files use governance tag for scoping
            hash_content = f"{sorted(governance_tags)[0]}:{hash_content}"
        elif project_tags:
            # Project files use project tag for scoping
            hash_content = f"{project_tags[0]}:{hash_content}"

    # Add metadata if present
    if metadata:
        # Filter out timestamp and dynamic fields
        static_metadata = {
            k: v for k, v in metadata.items()
            if k not in ['timestamp', 'content_hash', 'embedding']
        }
        if static_metadata:
            # Sort keys and use consistent JSON serialization
            hash_content += json.dumps(static_metadata, sort_keys=True, ensure_ascii=True)

    # Generate hash
    return hashlib.sha256(hash_content.encode('utf-8')).hexdigest()