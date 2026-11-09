# The funcWrappers module contains wrapper functions for the C++ functions
# Other modules may import this module to call C++ function easily

from ctypes import byref, c_bool, c_double, c_int64, c_char_p, CDLL, POINTER
from pathlib import Path
import numpy as np

# g++ -std=c++20 -fPIC -shared -o func.so cpp/func.cpp

# to compile all available C++ functions for the project:
# g++ -std=c++20 -fPIC -shared -o extCheckRestrOnSubfmls.so cpp/extCheckRestrOnSubfmls.cpp && g++ -std=c++20 -fPIC -shared -o extGetHeadAndInputs.so cpp/extGetHeadAndInputs.cpp && g++ -std=c++20 -fPIC -shared -o extGetSubfmlByInd.so cpp/extGetSubfmlByInd.cpp && g++ -std=c++20 -fPIC -shared -o extParseEval.so cpp/extParseEval.cpp && g++ -std=c++20 -fPIC -shared -o extParseEvalTwoVar.so cpp/extParseEvalTwoVar.cpp && g++ -std=c++20 -fPIC -shared -o extEvalRPN.so cpp/extEvalRPN.cpp && g++ -std=c++20 -fPIC -shared -o extParseEvalBatch.so cpp/extParseEvalBatch.cpp && g++ -std=c++20 -fPIC -shared -o extParseEvalTwoVarBatch.so cpp/extParseEvalTwoVarBatch.cpp && g++ -std=c++20 -O2 -fPIC -shared -o extEvalPN.so cpp/extEvalPN.cpp

REPO_ROOT = Path(__file__).resolve().parent


def _shared_library(name):
    """Load compiled C++ libraries from the repository root."""
    return CDLL(str(REPO_ROOT / name))

def wrapExtEvalPN(funcStr, inputs):
    # handle the edge case
    if (funcStr == ''): return 0, 1 # 1 stack left indicating error

    # formula validity check
    assert "(" not in funcStr and ")" not in funcStr, "funcStr must be in polish notation"

    # convert a number to a list if for rebustness of code
    if (not isinstance(inputs, np.ndarray)):
        if (not isinstance(inputs, list)):
            inputs = [[inputs]]

        # [1, 2, 3,...] ==> [[1], [2], [3],...]
        for i in range(len(inputs)):
            if (not isinstance(inputs[i], list)):
                inputs[i] = [inputs[i]]

    # convert all sublists to the numpy type
    inputs = np.array(inputs).astype(np.float64)
        
    # build the callable from C++
    extEvalPN = _build_extEvalPN()

    # create a ctypes array of pointers
    doublePtr = POINTER(c_double)
    cinputs = (len(inputs) * POINTER(c_double))()
    for i in range(len(inputs)):
        cinputs[i] = inputs[i].ctypes.data_as(POINTER(c_double))

    # convert the inputs into corresponding C types
    cfuncStr = c_char_p(funcStr.encode("utf-8"))
    csize = c_int64(len(inputs))
    outputs = np.zeros((len(inputs),)).astype(np.float64)
    coutputs = outputs.ctypes.data_as(POINTER(c_double))
    stacklefts = np.zeros((len(inputs),)).astype(np.float64)
    cstacklefts = stacklefts.ctypes.data_as(POINTER(c_double))

    # call the extParseEval function
    extEvalPN(byref(cfuncStr), cinputs, byref(csize), byref(coutputs), byref(cstacklefts))

    # convert results to python types
    outputs = np.ctypeslib.as_array(coutputs, shape = (len(inputs),))
    outputs = np.nan_to_num(outputs, nan = 0.0) # nans are converted to 0.0
    stacklefts = np.ctypeslib.as_array(cstacklefts, shape = (len(inputs),))

    # return the value
    return outputs, stacklefts


def wrapExtEvalRPN(funcStr, inputs):
    # formula validity check
    assert "(" not in funcStr and ")" not in funcStr, "funcStr must be in reverse polish notation"

    # convert a number to a list if for rebustness of code
    if (not isinstance(inputs, np.ndarray)):
        if (not isinstance(inputs, list)):
            inputs = [[inputs]]
        for i in range(len(inputs)):
            if (not isinstance(inputs[i], list)):
                inputs[i] = [inputs[i]]
        inputs = np.array(inputs).astype(np.float64)

    if (len(inputs.shape)):
        inputs = inputs.reshape(-1, 1)

    # build the callable from C++
    extEvalRPN = _build_extEvalRPN()

    # create a ctypes array of pointers
    doublePtr = POINTER(c_double)
    cinputs = (len(inputs) * POINTER(c_double))()
    for i in range(len(inputs)):
        cinputs[i] = inputs[i].ctypes.data_as(POINTER(c_double))

    # convert the inputs into corresponding C types
    cfuncStr = c_char_p(funcStr.encode("utf-8"))
    csize = c_int64(len(inputs))
    outputs = np.zeros((len(inputs),)).astype(np.float64)
    coutputs = outputs.ctypes.data_as(POINTER(c_double))
    stacklefts = np.zeros((len(inputs),)).astype(np.float64)
    cstacklefts = stacklefts.ctypes.data_as(POINTER(c_double))

    # call the extParseEval function
    extEvalRPN(byref(cfuncStr), cinputs, byref(csize), byref(coutputs), byref(cstacklefts))

    # convert results to python types
    outputs = np.ctypeslib.as_array(coutputs, shape = (len(inputs),))
    outputs = np.nan_to_num(outputs, nan = 0.0) # nans are converted to 0.0
    stacklefts = np.ctypeslib.as_array(cstacklefts, shape = (len(inputs),))

    # return the value
    return outputs, stacklefts
# with average_frequency
# def wrapExtParseEvalTwoVarBatch(funcStr, f_arr, t_arr, mask, average_freq):
#     # convert to np.float64
#     f_arr = f_arr.astype(np.float64)
#     t_arr = t_arr.astype(np.float64)
#     mask = mask.astype(np.float64)
#     # average_freq = average_freq.astype(np.float64)
#     average_freq = np.float64(average_freq)

#     # get the sizes
#     f_arr_size = np.max(f_arr.shape)
#     t_arr_size = np.max(t_arr.shape)

#     # sanity check
#     assert(mask.shape[0] == f_arr_size and mask.shape[1] == t_arr_size)

#     # build the callable from C++
#     extParseEvalTwoVarBatch, free_array = _build_extParseEvalTwoVarBatch()

#     # convert the inputs into corresponding C types
#     cfuncStr = c_char_p(funcStr.encode("utf-8"))
#     cf_arr = f_arr.ctypes.data_as(POINTER(c_double))
#     ct_arr = t_arr.ctypes.data_as(POINTER(c_double))
#     cf_arr_size = c_int64(f_arr_size)
#     ct_arr_size = c_int64(t_arr_size)
#     cmask = mask.ravel().ctypes.data_as(POINTER(c_double))
#     caverage_freq = c_double(average_freq)

#     # call the extParseEval function
#     cres_arr_ptr = extParseEvalTwoVarBatch(byref(cfuncStr), cf_arr, byref(cf_arr_size), ct_arr, byref(ct_arr_size), cmask, caverage_freq)
#     res_arr = np.copy(np.ctypeslib.as_array(cres_arr_ptr, shape = (f_arr_size * t_arr_size,)))
#     free_array(cres_arr_ptr)

#     # handle -0.0
#     res_arr[res_arr == -0.0] = 0.0

#     # return the value
#     return res_arr.reshape(f_arr_size, t_arr_size)

# with average_frequency
# def wrapExtParseEvalBatch(funcStr, t_arr, average_freq):
#     # convert to np.float64
#     t_arr = t_arr.astype(np.float64)
#     average_freq = np.float64(average_freq)

#     # get the time array's length
#     t_arr_size = np.max(t_arr.shape)

#     # build the callable from C++
#     extParseEvalBatch, free_array = _build_extParseEvalBatch()

#     # convert the inputs into corresponding C types
#     cfuncStr = c_char_p(funcStr.encode("utf-8"))
#     ct_arr = t_arr.ctypes.data_as(POINTER(c_double))
#     ct_arr_size = c_int64(t_arr_size)
#     caverage_freq = c_double(average_freq)

#     # call the extParseEval function
#     cres_arr_ptr = extParseEvalBatch(byref(cfuncStr), ct_arr, byref(ct_arr_size), caverage_freq)
#     res_arr = np.copy(np.ctypeslib.as_array(cres_arr_ptr, shape = (t_arr_size,)))
#     free_array(cres_arr_ptr)

#     # handle -0.0
#     res_arr[res_arr == -0.0] = 0.0

#     # return the value
#     return res_arr.tolist()
def wrapExtParseEvalTwoVarBatch(funcStr, f_arr, t_arr, mask):
    # convert to np.float64
    f_arr = f_arr.astype(np.float64)
    t_arr = t_arr.astype(np.float64)
    mask = mask.astype(np.float64)

    # get the sizes
    f_arr_size = np.max(f_arr.shape)
    t_arr_size = np.max(t_arr.shape)

    # sanity check
    assert(mask.shape[0] == f_arr_size and mask.shape[1] == t_arr_size)

    # build the callable from C++
    extParseEvalTwoVarBatch, free_array = _build_extParseEvalTwoVarBatch()

    # convert the inputs into corresponding C types
    cfuncStr = c_char_p(funcStr.encode("utf-8"))
    cf_arr = f_arr.ctypes.data_as(POINTER(c_double))
    ct_arr = t_arr.ctypes.data_as(POINTER(c_double))
    cf_arr_size = c_int64(f_arr_size)
    ct_arr_size = c_int64(t_arr_size)
    cmask = mask.ravel().ctypes.data_as(POINTER(c_double))

    # call the extParseEval function
    cres_arr_ptr = extParseEvalTwoVarBatch(byref(cfuncStr), cf_arr, byref(cf_arr_size), ct_arr, byref(ct_arr_size), cmask)
    res_arr = np.copy(np.ctypeslib.as_array(cres_arr_ptr, shape = (f_arr_size * t_arr_size,)))
    free_array(cres_arr_ptr)

    # handle -0.0
    res_arr[res_arr == -0.0] = 0.0

    # return the value
    return res_arr.reshape(f_arr_size, t_arr_size)

def wrapExtParseEvalBatch(funcStr, t_arr):
    # convert to np.float64
    t_arr = t_arr.astype(np.float64)

    # get the time array's length
    t_arr_size = np.max(t_arr.shape)

    # build the callable from C++
    extParseEvalBatch, free_array = _build_extParseEvalBatch()

    # convert the inputs into corresponding C types
    cfuncStr = c_char_p(funcStr.encode("utf-8"))
    ct_arr = t_arr.ctypes.data_as(POINTER(c_double))
    ct_arr_size = c_int64(t_arr_size)

    # call the extParseEval function
    cres_arr_ptr = extParseEvalBatch(byref(cfuncStr), ct_arr, byref(ct_arr_size))
    res_arr = np.copy(np.ctypeslib.as_array(cres_arr_ptr, shape = (t_arr_size,)))
    free_array(cres_arr_ptr)

    # handle -0.0
    res_arr[res_arr == -0.0] = 0.0

    # return the value
    return res_arr.tolist()


def wrapExtParseEvalTwoVar(funcStr, f, t):
    # build the callable from C++
    extParseEvalTwoVar = _build_extParseEvalTwoVar()

    # convert the inputs into corresponding C types
    cfuncStr = c_char_p(funcStr.encode("utf-8"))
    cf = c_double(f)
    ct = c_double(t)

    # call the extParseEval function
    res = extParseEvalTwoVar(byref(cfuncStr), byref(cf), byref(ct))

    # handle -0.0
    if (res == -0.0 or res == 0.0): res = 0.0

    # return the value
    return res


def wrapExtParseEval(funcStr, t):
    # build the callable from C++
    extParseEval = _build_extParseEval()

    # convert the inputs into corresponding C types
    cfuncStr = c_char_p(funcStr.encode("utf-8"))
    ct = c_double(t)

    # call the extParseEval function
    res = extParseEval(byref(cfuncStr), byref(ct))

    # handle -0.0
    if (res == -0.0 or res == 0.0): res = 0.0

    # return the value
    return res


def wrapExtGetSubfmlByInd(fml, index):
    # build the callable from C++
    extGetSubfmlByInd = _build_extGetSubfmlByInd()

    # convert the inputs into correcponding C types
    cfml = c_char_p(fml.encode("utf-8"))
    index = c_int64(index)

    # call the extGetSubfmlByInd function
    subfml = extGetSubfmlByInd(byref(cfml), byref(index)).decode("utf-8")

    # return the value
    return subfml


def wrapExtGetHeadAndInputs(funcStr):
    # build the callable from C++
    [extGetHeadAndInputs, free_mem] = _build_extGetHeadAndInputs()

    # convert the inputs into correcponding C types
    cfuncStr = c_char_p(funcStr.encode("utf-8"))
    chead = c_char_p()
    cinputs = POINTER(c_char_p)()
    csize = c_int64()

    # call the extGetSubfmlByInd function
    extGetHeadAndInputs(byref(cfuncStr), byref(chead), byref(cinputs), byref(csize))
    head = chead.value.decode("utf-8")
    inputs = [cinputs[i].decode("utf-8") for i in range(csize.value)]

    # free memory
    free_mem(chead, cinputs, csize)

    # return the value
    return [head, inputs]


def wrapExtCheckRestrOnSubfmls(fml, restrLen):
    # build the callable from C++
    extCheckRestrOnSubfmls = _build_extCheckRestrOnSubfmls()

    # convert the inputs into correcponding C types
    cfml = c_char_p(fml.encode("utf-8"))
    crestrLen = c_int64(restrLen)

    # call the extGetSubfmlByInd function
    flag = extCheckRestrOnSubfmls(cfml, crestrLen)

    # return the value
    return flag


#################### Private Functions ###################
# The builder functions for building compiled C++ shared libraries

def _build_extEvalPN():
    # load the compiled library
    lib = _shared_library("extEvalPN.so")

    # define the input and output data types
    lib.extEvalPN.argtypes = (POINTER(c_char_p), POINTER(POINTER(c_double)), POINTER(c_int64), POINTER(POINTER(c_double)), POINTER(POINTER(c_double)))
    lib.extEvalPN.restype = None

    # return the prepared function
    return lib.extEvalPN

def _build_extEvalRPN():
    # load the compiled library
    lib = _shared_library("extEvalRPN.so")

    # define the input and output data types
    lib.extEvalRPN.argtypes = (POINTER(c_char_p), POINTER(POINTER(c_double)), POINTER(c_int64), POINTER(POINTER(c_double)), POINTER(POINTER(c_double)))
    lib.extEvalRPN.restype = None

    # return the prepared function
    return lib.extEvalRPN

def _build_extParseEvalTwoVarBatch():
    lib = _shared_library("extParseEvalTwoVarBatch.so")
    #lib.extParseEvalTwoVarBatch.argtypes = (POINTER(c_char_p), POINTER(c_double), POINTER(c_int64), POINTER(c_double), POINTER(c_int64), POINTER(c_double), POINTER(c_double))
    lib.extParseEvalTwoVarBatch.argtypes = (POINTER(c_char_p), POINTER(c_double), POINTER(c_int64), POINTER(c_double), POINTER(c_int64), POINTER(c_double))
    lib.extParseEvalTwoVarBatch.restype = POINTER(c_double)
    return lib.extParseEvalTwoVarBatch, lib.free_array

def _build_extParseEvalBatch():
    lib = _shared_library("extParseEvalBatch.so")
    #lib.extParseEvalBatch.argtypes = (POINTER(c_char_p), POINTER(c_double), POINTER(c_int64), POINTER(c_double))
    lib.extParseEvalBatch.argtypes = (POINTER(c_char_p), POINTER(c_double), POINTER(c_int64))
    lib.extParseEvalBatch.restype = POINTER(c_double)
    return lib.extParseEvalBatch, lib.free_array

def _build_extParseEvalTwoVar():
    lib = _shared_library("extParseEvalTwoVar.so")
    lib.extParseEvalTwoVar.argtypes = (POINTER(c_char_p), POINTER(c_double), POINTER(c_double))
    lib.extParseEvalTwoVar.restype = c_double
    return lib.extParseEvalTwoVar


def _build_extParseEval():
    lib = _shared_library("extParseEval.so")
    lib.extParseEval.argtypes = (POINTER(c_char_p), POINTER(c_double))
    lib.extParseEval.restype = c_double
    return lib.extParseEval


def _build_extGetSubfmlByInd():
    lib = _shared_library("extGetSubfmlByInd.so")
    lib.extGetSubfmlByInd.argtypes = (POINTER(c_char_p), POINTER(c_int64))
    lib.extGetSubfmlByInd.restype = c_char_p
    return lib.extGetSubfmlByInd


def _build_extGetHeadAndInputs():
    lib = _shared_library("extGetHeadAndInputs.so")
    lib.extGetHeadAndInputs.argtypes = (POINTER(c_char_p), POINTER(c_char_p), POINTER(POINTER(c_char_p)), POINTER(c_int64))
    lib.extGetHeadAndInputs.restype = None

    lib.free_mem.argtypes = (c_char_p, POINTER(c_char_p), c_int64)
    lib.free_mem.restype = None
    return [lib.extGetHeadAndInputs, lib.free_mem]


def _build_extCheckRestrOnSubfmls():
    lib = _shared_library("extCheckRestrOnSubfmls.so")
    lib.extCheckRestrOnSubfmls.argtypes = (POINTER(c_char_p), POINTER(c_int64))
    lib.extCheckRestrOnSubfmls.restype = c_bool
    return lib.extCheckRestrOnSubfmls


if (__name__ == "__main__"):
    # testing ParseEval
    '''
    funcStr = "(neg x)"
    x = 1
    y = 1
    t = 2
    res = wrapExtParseEval(funcStr, x, y, t)
    print(res)

    # testing getSubfmlByInd
    fml = "(sqr (max x t))"
    ind = 3
    subfml = wrapExtGetSubfmlByInd(fml, ind)
    print(subfml)

    # testing getHeadAndInputs
    fml = "(max x y)"
    [head, inputs] = wrapExtGetHeadAndInputs(fml)
    print(f"head: {head}")
    print(f"inputs: {inputs}")

    # testing CheckRestrOnSubfmls
    fml = "(sqr (max x t))"
    flag = wrapExtCheckRestrOnSubfmls(fml, 4)
    print(flag)
    '''

    fml = "(/ (neg (sqrt 0)) (- 3 t))"
    res = wrapExtParseEval(fml, 2)
    print(res)
