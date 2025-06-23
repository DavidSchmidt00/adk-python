# Copyright 2025 Google LLC
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

"""Utility functions for structured A2A request and response logging."""

from __future__ import annotations

import json
import sys

try:
  from a2a.types import DataPart as A2ADataPart
  from a2a.types import Message as A2AMessage
  from a2a.types import Part as A2APart
  from a2a.types import SendMessageRequest
  from a2a.types import SendMessageResponse
  from a2a.types import Task as A2ATask
  from a2a.types import TextPart as A2ATextPart
except ImportError as e:
  if sys.version_info < (3, 10):
    raise ImportError(
        "A2A Tool requires Python 3.10 or above. Please upgrade your Python"
        " version."
    ) from e
  else:
    raise e


# Constants
_NEW_LINE = "\n"
_EXCLUDED_PART_FIELD = {"file": {"bytes"}}


def build_message_part_log(part: A2APart) -> str:
  """Builds a log representation of an A2A message part.

  Args:
    part: The A2A message part to log.

  Returns:
    A string representation of the part.
  """
  if isinstance(part.root, A2ATextPart):
    return f"TextPart: {part.root.text[:100]}" + (
        "..." if len(part.root.text) > 100 else ""
    )
  elif isinstance(part.root, A2ADataPart):
    # For data parts, show the data keys but exclude large values
    data_summary = {
        k: (
            f"<{type(v).__name__}>"
            if isinstance(v, (dict, list)) and len(str(v)) > 100
            else v
        )
        for k, v in part.root.data.items()
    }
    return f"DataPart: {json.dumps(data_summary, indent=2)}"
  else:
    return (
        f"{type(part.root).__name__}:"
        f" {part.model_dump_json(exclude_none=True, exclude=_EXCLUDED_PART_FIELD)}"
    )


def build_a2a_request_log(req: SendMessageRequest) -> str:
  """Builds a structured log representation of an A2A request.

  Args:
    req: The A2A SendMessageRequest to log.

  Returns:
    A formatted string representation of the request.
  """
  # Message parts logs
  message_parts_logs = []
  if req.params.message.parts:
    for i, part in enumerate(req.params.message.parts):
      message_parts_logs.append(f"Part {i}: {build_message_part_log(part)}")

  # Configuration logs
  config_log = "None"
  if req.params.configuration:
    config_data = {
        "acceptedOutputModes": req.params.configuration.acceptedOutputModes,
        "blocking": req.params.configuration.blocking,
        "historyLength": req.params.configuration.historyLength,
        "pushNotificationConfig": bool(
            req.params.configuration.pushNotificationConfig
        ),
    }
    config_log = json.dumps(config_data, indent=2)

  return f"""
A2A Request:
-----------------------------------------------------------
Request ID: {req.id}
Method: {req.method}
JSON-RPC: {req.jsonrpc}
-----------------------------------------------------------
Message:
  ID: {req.params.message.messageId}
  Role: {req.params.message.role}
  Task ID: {req.params.message.taskId}
  Context ID: {req.params.message.contextId}
-----------------------------------------------------------
Message Parts:
{_NEW_LINE.join(message_parts_logs) if message_parts_logs else "No parts"}
-----------------------------------------------------------
Configuration:
{config_log}
-----------------------------------------------------------
Metadata:
{json.dumps(req.params.metadata, indent=2) if req.params.metadata else "None"}
-----------------------------------------------------------
"""


def build_a2a_response_log(resp: SendMessageResponse) -> str:
  """Builds a structured log representation of an A2A response.

  Args:
    resp: The A2A SendMessageResponse to log.

  Returns:
    A formatted string representation of the response.
  """
  # Handle error responses
  if hasattr(resp.root, "error"):
    return f"""
A2A Response:
-----------------------------------------------------------
Type: ERROR
Error Code: {resp.root.error.code}
Error Message: {resp.root.error.message}
Error Data: {json.dumps(resp.root.error.data, indent=2) if resp.root.error.data else "None"}
-----------------------------------------------------------
Response ID: {resp.root.id}
JSON-RPC: {resp.root.jsonrpc}
-----------------------------------------------------------
"""

  # Handle success responses
  result = resp.root.result
  result_type = type(result).__name__

  # Build result details based on type
  result_details = []

  if isinstance(result, A2ATask):
    result_details.extend([
        f"Task ID: {result.id}",
        f"Context ID: {result.contextId}",
        f"Status State: {result.status.state}",
        f"Status Timestamp: {result.status.timestamp}",
        f"History Length: {len(result.history) if result.history else 0}",
        f"Artifacts Count: {len(result.artifacts) if result.artifacts else 0}",
    ])

    # Add status message if present
    if result.status.message:
      status_parts_logs = []
      if result.status.message.parts:
        for i, part in enumerate(result.status.message.parts):
          status_parts_logs.append(
              f"  Part {i}: {build_message_part_log(part)}"
          )
      result_details.extend([
          f"Status Message ID: {result.status.message.messageId}",
          f"Status Message Role: {result.status.message.role}",
          "Status Message Parts:",
          *status_parts_logs,
      ])

  elif isinstance(result, A2AMessage):
    result_details.extend([
        f"Message ID: {result.messageId}",
        f"Role: {result.role}",
        f"Task ID: {result.taskId}",
        f"Context ID: {result.contextId}",
    ])

    # Add message parts
    if result.parts:
      result_details.append("Message Parts:")
      for i, part in enumerate(result.parts):
        result_details.append(f"  Part {i}: {build_message_part_log(part)}")

  return f"""
A2A Response:
-----------------------------------------------------------
Type: SUCCESS
Result Type: {result_type}
-----------------------------------------------------------
Result Details:
{_NEW_LINE.join(result_details)}
-----------------------------------------------------------
Response ID: {resp.root.id}
JSON-RPC: {resp.root.jsonrpc}
-----------------------------------------------------------
Raw response summary:
{result.model_dump_json(exclude_none=True, exclude=_EXCLUDED_PART_FIELD) if hasattr(result, 'model_dump_json') else str(result)}
-----------------------------------------------------------
"""
