from numpy import ndarray


def assert_same_shape(a: ndarray, b: ndarray):
    assert a.shape == b.shape,\
            '''
		Two ndarrays should have the same shape;
		instead, first ndarray's shape is {0}
		and second ndarray's shape is {1}
		'''.format(tuple(a.shape), tuple(b.shape))


def to_2d_np(a: ndarray, type: str = "col") -> ndarray:
    assert a.ndim == 1
    if type == "col":
        return a.reshape(-1, 1)
    elif type == "row":
        return a.reshape(1, -1)
