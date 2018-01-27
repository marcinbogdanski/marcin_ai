import numpy as np
import pdb

class Memory:
    def __init__(self, state_shape, act_shape, dtypes, max_len):
        assert isinstance(state_shape, tuple)
        assert isinstance(act_shape, tuple)
        assert isinstance(dtypes, tuple)
        assert len(dtypes) == 5
        assert dtypes[-1] == bool
        assert isinstance(max_len, int)
        assert max_len > 0

        self._max_len = max_len
        self._curr_insert_ptr = 0
        self._curr_len = 0

        self._state_shape = state_shape
        self._act_shape = act_shape

        St_shape = [max_len] + list(state_shape)
        At_shape = [max_len] + list(act_shape)
        Rt_1_shape = [max_len] + [1]
        St_1_shape = [max_len] + list(state_shape)
        done_shape = [max_len] + [1]
        St_dtype, At_dtype, Rt_1_dtype, St_1_dtype, done_dtype = dtypes

        self._hist_St = np.zeros(St_shape, dtype=St_dtype)
        self._hist_At = np.zeros(At_shape, dtype=At_dtype)
        self._hist_Rt_1 = np.zeros(Rt_1_shape, dtype=Rt_1_dtype)
        self._hist_St_1 = np.zeros(St_1_shape, dtype=St_1_dtype)
        self._hist_done = np.zeros(done_shape, dtype=done_dtype)

    def append(self, St, At, Rt_1, St_1, done):
        assert isinstance(St, np.ndarray)
        assert St.shape == self._state_shape
        assert St.dtype == self._hist_St.dtype
        assert isinstance(At, int) or isinstance(At, np.int64)
        assert isinstance(Rt_1, int)
        assert isinstance(St_1, np.ndarray)
        assert St_1.shape == self._state_shape
        assert St_1.dtype == self._hist_St_1.dtype
        assert isinstance(done, bool)

        self._hist_St[self._curr_insert_ptr] = St
        self._hist_At[self._curr_insert_ptr] = At
        self._hist_Rt_1[self._curr_insert_ptr] = Rt_1
        self._hist_St_1[self._curr_insert_ptr] = St_1
        self._hist_done[self._curr_insert_ptr] = done

        if self._curr_len < self._max_len:
            self._curr_len += 1

        self._curr_insert_ptr += 1 
        if self._curr_insert_ptr >= self._max_len:
            self._curr_insert_ptr = 0

    def _print_all(self):
        print()
        print('_hist_St')
        print(self._hist_St)

        print()
        print('_hist_At')
        print(self._hist_At)

        print()
        print('_hist_Rt_1')
        print(self._hist_Rt_1)

        print()
        print('_hist_St_1')
        print(self._hist_St_1)

        print()
        print('_hist_done')
        print(self._hist_done)

    def length(self):
        return self._curr_len

    def get_batch(self, batch_len):
        assert self._curr_len > 0
        assert batch_len > 0

        #indices = np.random.choice(range(self._curr_len), batch_len)
        indices = np.random.randint(
            low=0, high=self._curr_len, size=batch_len, dtype=int)

        # states = self._hist_St[indices]
        # actions = self._hist_At[indices]
        # rewards_1 = self._hist_Rt_1[indices]
        # states_1 = self._hist_St_1[indices]
        # dones = self._hist_done[indices]

        # pdb.set_trace()

        states = np.take(self._hist_St, indices, axis=0)
        actions = np.take(self._hist_At, indices, axis=0)
        rewards_1 = np.take(self._hist_Rt_1, indices, axis=0)
        states_1 = np.take(self._hist_St_1, indices, axis=0)
        dones = np.take(self._hist_done, indices, axis=0)

        # batch = []
        # for i, idx in enumerate(indices):
        #     tup = (self._hist_St[idx],
        #            self._hist_At[idx],
        #            self._hist_Rt_1[idx],
        #            self._hist_St_1[idx],
        #            self._hist_done[idx])
        #     batch.append(tup)

        #     assert (tup[0] == states[i]).all()
        #     assert (tup[1] == actions[i]).all()
        #     assert (tup[2] == rewards_1[i]).all()
        #     assert (tup[3] == states_1[i]).all()
        #     assert (tup[4] == dones[i]).all()

        return states, actions, rewards_1, states_1, dones



if __name__ == '__main__':

    mem = Memory(state_shape=(2, ),
                 act_shape=(1, ),
                 dtypes=(float, int, float, float, bool),
                 max_len=10)

    i = 1
    mem.append(St=np.array([i, i], dtype=float),
        At=i, Rt_1=-i, St_1=np.array([i+1,i+1], dtype=float), done=False)
    res = mem.get_batch(3)
    print(res)

    i = 2
    mem.append(St=np.array([i, i], dtype=float),
        At=i, Rt_1=-i, St_1=np.array([i+1,i+1], dtype=float), done=False)
    res = mem.get_batch(3)
    print(res)

    i = 3
    mem.append(St=np.array([i, i], dtype=float),
        At=i, Rt_1=-i, St_1=np.array([i+1,i+1], dtype=float), done=False)
    res = mem.get_batch(3)
    print(res)


    for i in range(4, 12):
        mem.append(St=np.array([i, i], dtype=float),
            At=i, Rt_1=-i, St_1=np.array([i+1,i+1], dtype=float), done=False)

    i = 12
    mem.append(St=np.array([i, i], dtype=float),
        At=i, Rt_1=-i, St_1=np.array([i+1,i+1], dtype=float), done=True)
    res = mem.get_batch(3)
    print(res)
