import logger
import pdb
pdb
L = logger.LogModule('Agent')
L.add_param('min', 3, 'minimum value')
L.add_param('max', 5, 'maximum value')
L.add_param('step_size', 30, 'step size')

L.add_data_item('Q', 'function Q value')
L.add_data_item('acc', 'accuracy')

L.append(0, Q=3, acc=12)
L.append(1, Q=4, acc=15)

pdb.set_trace()