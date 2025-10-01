"""
Observer pattern for event-driven architecture.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Callable
from utils.logging_config import get_logger

logger = get_logger(__name__)


class Observer(ABC):
    """Abstract observer base class."""
    
    @abstractmethod
    def update(self, subject: 'Subject', event: str, data: Any):
        """Called when subject state changes."""
        pass


class Subject:
    """Subject that notifies observers of changes."""
    
    def __init__(self):
        self._observers: Dict[str, List[Observer]] = {}
        self.logger = get_logger(self.__class__.__name__)
    
    def attach(self, observer: Observer, event: str = '*'):
        """Attach an observer for specific event or all events."""
        if event not in self._observers:
            self._observers[event] = []
        
        if observer not in self._observers[event]:
            self._observers[event].append(observer)
            self.logger.debug(f"Attached observer {observer.__class__.__name__} for event '{event}'")
    
    def detach(self, observer: Observer, event: str = '*'):
        """Detach an observer from specific event or all events."""
        if event in self._observers:
            if observer in self._observers[event]:
                self._observers[event].remove(observer)
                self.logger.debug(f"Detached observer {observer.__class__.__name__} from event '{event}'")
    
    def notify(self, event: str, data: Any = None):
        """Notify all observers of an event."""
        self.logger.debug(f"Notifying observers of event '{event}'")
        
        # Notify specific event observers
        if event in self._observers:
            for observer in self._observers[event]:
                try:
                    observer.update(self, event, data)
                except Exception as e:
                    self.logger.error(f"Error notifying observer: {e}", exc_info=True)
        
        # Notify wildcard observers
        if '*' in self._observers:
            for observer in self._observers['*']:
                try:
                    observer.update(self, event, data)
                except Exception as e:
                    self.logger.error(f"Error notifying observer: {e}", exc_info=True)


class CallbackObserver(Observer):
    """Observer that calls a callback function."""
    
    def __init__(self, callback: Callable):
        self.callback = callback
    
    def update(self, subject: Subject, event: str, data: Any):
        """Call the callback function."""
        self.callback(subject, event, data)


class TrainingObserver(Observer):
    """Observer for training events."""
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.metrics_history = []
    
    def update(self, subject: Subject, event: str, data: Any):
        """Handle training events."""
        if event == 'epoch_end':
            self.metrics_history.append(data)
            self.logger.info(f"Epoch {data.get('epoch')}: {data.get('metrics')}")
        elif event == 'training_start':
            self.logger.info("Training started")
            self.metrics_history = []
        elif event == 'training_end':
            self.logger.info("Training completed")
