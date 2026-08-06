import queue


class EventBus:

    def __init__(self):

        self.events = queue.Queue()


    def emit(
        self,
        event,
        data=None
    ):

        self.events.put(
            {
                "event": event,
                "data": data
            }
        )


    def wait(self):

        return self.events.get()


bus = EventBus()