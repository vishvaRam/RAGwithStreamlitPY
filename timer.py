import time

class Timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = time.time()
        print("Timer started....")

    def stop(self):
        if self.start_time is None:
            return
        self.end_time = time.time()
        elapsed_time = self.end_time - self.start_time
        formatted_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        print(f"\ntime: {formatted_time}")
        return formatted_time