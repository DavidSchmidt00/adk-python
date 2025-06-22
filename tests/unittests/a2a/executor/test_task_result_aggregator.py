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

import sys
from unittest.mock import Mock

import pytest

# Skip all tests in this module if Python version is less than 3.10
pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 10), reason="A2A requires Python 3.10+"
)

# Import dependencies with version checking
try:
  from a2a.types import TaskState
  from a2a.types import TaskStatus
  from a2a.types import TaskStatusUpdateEvent
  from google.adk.a2a.executor.task_result_aggregator import TaskResultAggregator
except ImportError as e:
  if sys.version_info < (3, 10):
    # Create dummy classes to prevent NameError during test collection
    # Tests will be skipped anyway due to pytestmark
    class DummyTypes:
      pass

    TaskState = DummyTypes()
    TaskStatus = DummyTypes()
    TaskStatusUpdateEvent = DummyTypes()
    TaskResultAggregator = DummyTypes()
  else:
    raise e


class TestTaskResultAggregator:
  """Test suite for TaskResultAggregator class."""

  def setup_method(self):
    """Set up test fixtures."""
    self.aggregator = TaskResultAggregator()

  def test_initial_state(self):
    """Test the initial state of the aggregator."""
    assert self.aggregator.task_state == TaskState.working

  def test_process_failed_event(self):
    """Test processing a failed task status event."""
    mock_event = Mock(spec=TaskStatusUpdateEvent)
    mock_event.status = Mock(spec=TaskStatus)
    mock_event.status.state = TaskState.failed

    self.aggregator.process_event(mock_event)

    assert self.aggregator.task_state == TaskState.failed

  def test_process_auth_required_event(self):
    """Test processing an auth_required task status event."""
    mock_event = Mock(spec=TaskStatusUpdateEvent)
    mock_event.status = Mock(spec=TaskStatus)
    mock_event.status.state = TaskState.auth_required

    self.aggregator.process_event(mock_event)

    assert self.aggregator.task_state == TaskState.auth_required

  def test_process_input_required_event(self):
    """Test processing an input_required task status event."""
    mock_event = Mock(spec=TaskStatusUpdateEvent)
    mock_event.status = Mock(spec=TaskStatus)
    mock_event.status.state = TaskState.input_required

    self.aggregator.process_event(mock_event)

    assert self.aggregator.task_state == TaskState.input_required

  def test_failed_state_priority(self):
    """Test that failed state takes priority over other states."""
    # First set to failed
    failed_event = Mock(spec=TaskStatusUpdateEvent)
    failed_event.status = Mock(spec=TaskStatus)
    failed_event.status.state = TaskState.failed

    self.aggregator.process_event(failed_event)

    # Then try to set to auth_required - should remain failed
    auth_event = Mock(spec=TaskStatusUpdateEvent)
    auth_event.status = Mock(spec=TaskStatus)
    auth_event.status.state = TaskState.auth_required

    self.aggregator.process_event(auth_event)

    assert self.aggregator.task_state == TaskState.failed

  def test_auth_required_priority_over_input_required(self):
    """Test that auth_required state takes priority over input_required."""
    # First set to auth_required
    auth_event = Mock(spec=TaskStatusUpdateEvent)
    auth_event.status = Mock(spec=TaskStatus)
    auth_event.status.state = TaskState.auth_required

    self.aggregator.process_event(auth_event)

    # Then try to set to input_required - should remain auth_required
    input_event = Mock(spec=TaskStatusUpdateEvent)
    input_event.status = Mock(spec=TaskStatus)
    input_event.status.state = TaskState.input_required

    self.aggregator.process_event(input_event)

    assert self.aggregator.task_state == TaskState.auth_required

  def test_non_task_status_event_ignored(self):
    """Test that non-TaskStatusUpdateEvent events are ignored."""
    mock_event = Mock()  # Not a TaskStatusUpdateEvent
    original_state = self.aggregator.task_state

    self.aggregator.process_event(mock_event)

    assert self.aggregator.task_state == original_state

  def test_event_not_modified(self):
    """Test that the input event state is set to working during processing."""
    mock_event = Mock(spec=TaskStatusUpdateEvent)
    mock_event.status = Mock(spec=TaskStatus)
    original_state = TaskState.auth_required
    mock_event.status.state = original_state

    self.aggregator.process_event(mock_event)

    # Event state is modified to working as part of the processing
    # (this ensures intermediate state is always working for a2a request handler)
    assert mock_event.status.state == TaskState.working

  def test_state_transitions_sequence(self):
    """Test a sequence of state transitions."""
    events = [
        (TaskState.working, TaskState.working),
        (TaskState.input_required, TaskState.input_required),
        (TaskState.auth_required, TaskState.auth_required),
        (TaskState.failed, TaskState.failed),
        (TaskState.working, TaskState.failed),  # Should remain failed
    ]

    for event_state, expected_aggregator_state in events:
      mock_event = Mock(spec=TaskStatusUpdateEvent)
      mock_event.status = Mock(spec=TaskStatus)
      mock_event.status.state = event_state

      self.aggregator.process_event(mock_event)

      assert self.aggregator.task_state == expected_aggregator_state, (
          f"Expected {expected_aggregator_state}, got"
          f" {self.aggregator.task_state}"
      )
