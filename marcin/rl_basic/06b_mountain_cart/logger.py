


class Log():
    def __init__(self, name, description):
        self.name = name
        self.description = description
        
        self.params = {}
        self.params_info = {}

        self.steps = []
        
        self.data = {}
        self.data_info = {}

        self._recording_started = False


    def add_param(self, name, value, info):
        self.params[name] = value
        self.params_info[name] = info

    def add_data_item(self, name, info):
        if not self._recording_started:
            self.data[name] = []
            self.data_info[name] = info
        else:
            raise ValueError('Cant add data items after calling append')


    def append(self, step, **data):

        self._recording_started = True

        if len(self.steps) != 0:
            assert step != self.steps[-1]

        assert set(self.data.keys()) == set(data.keys())
        for key, val in data.items():
            assert len(self.data[key]) == len(self.steps)

        self.steps.append(step)
        for key, val in data.items():
            self.data[key].append(val)

class Logger():
    def __init__(self):
        pass

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump( self.__dict__, f )

    def load(self, filename):
        with open(filename, 'rb') as f:
            tmp_dict = pickle.load(f)
            self.__dict__.clear()
            self.__dict__.update(tmp_dict)
