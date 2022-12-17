
class Logger:

    LOGS_LOCATION = './output'

    def __init__(self, log_file_name: str) -> None:
        self._file_name = log_file_name
        self._file = open(f"{self.LOGS_LOCATION}/{self._file_name}", 'a')
        self._file.truncate(0)

    def write(self, content: str) -> None:
        print(content)
        self._file.write(f"{content}\n")
