class Logger:
    def __init__(self, whites_log_path, blacks_log_path) -> None:
        self.whites_log_path = whites_log_path
        self.blacks_log_path = blacks_log_path

        self.__clear(self.whites_log_path)
        self.__clear(self.blacks_log_path)


    def __clear(self, file):
        open(file, "w").close()


    def write(self, target, text):
        if target == "whites":
            with open(self.whites_log_path, "w") as f:
                f.write(text)
        elif target == "blacks":
            with open(self.blacks_log_path, "w") as f:
                f.write(text)


    def clear(self, target):
        if target == "whites":
            self.__clear(self.whites_log_path)
        elif target == "blacks":
            self.__clear(self.blacks_log_path)


    def update(self, target, population, epoch):
        # self.__clear(target)
        self.write(target, f"<<<<<<<<<< Epoch {epoch} - {target} >>>>>>>>>>\n{population}\n\n")
