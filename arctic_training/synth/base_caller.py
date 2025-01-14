from abc import ABC
from abc import abstractmethod


class BatchProcessor(ABC):
    @abstractmethod
    def add_chat_to_batch_task(self, task_name, **kwargs):
        """
        Add a chat completion request to the batch task.
        """
        pass

    @abstractmethod
    def execute_batch_task(self, task_name):
        """
        A synchronous method to execute the batch task. This method will block until the task is completed.
        """
        pass

    @abstractmethod
    def extract_messages_from_responses(responses):
        """
        Extract the response from the response object.
        """
        pass
