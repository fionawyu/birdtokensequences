# The grammar module contains the definition of the grammar and some relavent functions
from re import sub
from random import choice, randint, expovariate
from sys import maxsize
from funcWrappers import wrapExtGetSubfmlByInd, wrapExtGetHeadAndInputs

############################### Public Constants #############################

# define infinity for integers
intinf = maxsize

# Starting node
#S = ["B", "U", "T"]

# Terminal: x, y, t, C, BB

# C is an integer between 1 to 10
#T = ["x", "y", "t", "C"] # can be extended
# T = ["t", "V"] # can be extended F: average frequency
#T = ["t"] + ["C"]
#T = ["t", "V"] + [str(round(i * 0.1, 1)) for i in range(101)] + ["pi"]
#T = ["t"] + [str(round(i * 0.1, 1)) for i in range(11)]

# Numeric constants available as terminals.
# The C++ evaluator (cpp/extParseEval.cpp) reads any token starting with a digit
# or '.' via stod, so bare literals like "0.5" or "7" evaluate directly.
# Two things it does NOT accept:
#   - negative literals ("-1" is parsed as the binary minus and yields 0);
#     write negatives as (neg 1) instead
#   - symbolic constants ("pi" has no case in the evaluator and silently gives 0)
CONST_MIN = 0.0    # lowest constant
CONST_MAX = 10.0   # highest constant
CONST_STEP = 0.1   # spacing between constants
CONSTS = [f"{CONST_MIN + i * CONST_STEP:g}"
     for i in range(int(round((CONST_MAX - CONST_MIN) / CONST_STEP)) + 1)]

V= ["t"]
T = ["t"] + CONSTS + \
    [f"(sin (+ (* {i/10:.1f} t) {j}))" for i in range(10) for j in range(1, 11)] + \
    [f"(exp (* {i:.1f} t))" for i in range(10)] + \
    [f"(exp (neg (* {i:.1f} t)))" for i in range(10)] + \
    [f"(sin (+ (exp (* {i} t)) {j}))" for i in range(1, 11) for j in range(1, 11)] + \
    [f"(sin (+ (sin (* {i} t)) {j}))" for i in range(1, 11) for j in range(1, 11)] +\
    [f"(sin (cos (sin (sin (+ (sin (* {i} t)) {j} )))))" for i in range(1, 11) for j in range(1, 11)] + \
    [f"(sin (sin (sin (sin (+ (exp (* {i} t)) {j} )))))" for i in range(1, 11) for j in range(1, 11)]





# Unary: ^2, sqrt, e^n, -f, sin, cos, tan
#U = ["exp", "sin"] # log, 
U = ["sqr", "sqrt", "exp","neg", "sin", "tan", "cos", "log", "abs"] # log, 

# Binaries: +, -, *, /
B = ["+", "-", "*", "/"]

# conditionals

X = [">", ">=", "<", "<="]

# and or
XB = ["&&", "||"]

# if
I = ["if"]


# the flattened grammar
FlatGram = T + U + B

############################### Public Functions #############################
# bottom up search with just the unary and the terminals up to a length
def genUnaryFML(lenLim = 3):
    # Base cases
    if (lenLim == 0): return []
    elif (lenLim == 1): return _postproclst(T)

    # Inductive cases
    cands = _postproclst(T)
    for i in range(lenLim - 1): 
        curlst = []
        for j in range(len(U)):
            for cand in cands:
                curlst.append(_postproc(f"({U[j]} {cand})"))
        cands += curlst

    # return the list
    return cands


# the function mutates a formula (parent) into a child
# The child is obtained by applying k subtree-replacement operations to the
# parent in sequence (see _subtreeReplace). The number of operations k is
# drawn from an exponential distribution (see _randNumMutations), so most
# children are one operation away from the parent and a few are several.
# MUT_NUM_SCALE is the mean of the underlying exponential draw.
MUT_NUM_SCALE = 1.0

def fMutate(fml, fmlSizeLim = intinf, subfmlSizeLim = 7, mode = "freq",
            numScale = MUT_NUM_SCALE): # intinf: largest integer
    curFml = fml
    for _ in range(_randNumMutations(numScale)):
        curFml = _subtreeReplace(curFml, fmlSizeLim, subfmlSizeLim, mode)
    return curFml
    

# the function recieves a formula and outputs its size by walking through it recursively
def fSize(fml):
    # Base Case: a terminal
    if ("(" not in fml and len(fml) != 0): return 1
    
    # Recursive Case:
    fml = sub(r"\(", "", fml)
    fml = sub(r"\)", "", fml)
    
    return len(fml.split())


# function for generating a random formula limited by some size
# sizeLimit cannot be lower than 1
def  randFML(sizeLimit = intinf, mode = "freq"):
    # sanity check
    assert sizeLimit >= 1, "sizeLimit must not be lower than 1"
    
    # # switch synthesis mode
    # if (mode == "amp" and "f" not in T): T.append("f")
    local_T = T.copy()
    # if mode is "amp", still choose the same terminals
    if mode == "amp":
        local_T = local_T

    # if mode == "amp" and "f" not in local_T:
    #     local_T.append("f")
    
    # If sizeLimit is 1, only a terminal can be returned
    if sizeLimit == 1:
        return _postproc(choice(local_T))
    
    # For sizeLimit 2, only terminals or unary operations (which require 2 tokens) are allowed.
    if sizeLimit == 2:
        # Only choose between terminal and unary operators
        root = _randOp(local_T + U)
        if root in local_T:
            return _postproc(root)
        else:  # Unary operator
            return _postproc(f"({root} {randFML(1, mode)})")
    
    # For sizeLimit >= 3, you can use binary operators too.
    # Randomly select a root from all available options (T, U, and B)
    root = _randOp(local_T + U + B)
    
    # Terminal
    if root in local_T:
        return _postproc(root)
    
    # Unary operator: Ensure sizeLimit is enough (>=2)
    elif root in U:
        # If by any chance sizeLimit - 1 < 1, fallback to a terminal.
        if sizeLimit - 1 < 1:
            return _postproc(choice(local_T))
        return _postproc(f"({root} {randFML(sizeLimit - 1, mode)})")
    
    # Binary operator: Need at least 3 tokens
    elif root in B:
        # Ensure we have at least 3 tokens available
        if sizeLimit - 2 < 1:
            return _postproc(choice(local_T))
        childL = randFML(sizeLimit - 2, mode)
        # Compute remaining tokens for right child
        remaining = sizeLimit - 1 - fSize(childL)
        # If remaining tokens are less than 1, force a terminal for childR
        if remaining < 1:
            childR = choice(local_T)
        else:
            childR = randFML(remaining, mode)
        return _postproc(f"({root} {childL} {childR})")
    
    # Fallback in case nothing matches: return a terminal.
    return _postproc(choice(local_T))

# function for generating a random formula limited by some size
# sizeLimit cannot be lower than 1
'''
def randFML(sizeLimit = intinf): # set default value intinf
    # sanity check
    assert sizeLimit >= 1, "sizeLimit must not be lower than 1"

    # initilize result
    resfml = ""

    # randomly generate formulae
    if (sizeLimit == 1): resfml = choice(T)
    elif (sizeLimit == 2):
        # randomly get a root
        root = _randOp(T + U)

        # in the T list
        if (root in T): resfml = root

        # in the U list
        else: resfml = f"({root} {randFML(sizeLimit - 1)})"

    elif (sizeLimit < 6):
        # randomly get a root
        root = _randOp(T + U + B)

        # if a terminal, just return it
        if (root in T): resfml = root

        # recursives
        elif (root in U): resfml = f"({root} {randFML(sizeLimit - 1)})"
        elif (root in B):
            childL = randFML(sizeLimit - 2)
            childR = randFML(sizeLimit - 1 - fSize(childL))
            resfml = f"({root} {childL} {childR})"
    else:
        # randomly get a root
        root = _randOp()

        # if a terminal, just return it
        if (root in T): resfml = root

        # recursives
        elif (root in U): resfml = f"({root} {randFML(sizeLimit - 1)})"
        elif (root in B):
            childL = randFML(sizeLimit - 2)
            childR = randFML(sizeLimit - 1 - fSize(childL))
            resfml = f"({root} {childL} {childR})"
        elif (root in I):
            childC = _randCond(sizeLimit - 3)
            childL = randFML(sizeLimit - 2 - fSize(childC))
            childR = randFML(sizeLimit - 1 - fSize(childC) - fSize(childL))
            resfml = f"({root} {childC} {childL} {childR})"

    # post process and return
    return _postproc(resfml)
'''


##################### Private Functions #########################

# draw the number of mutation operations k >= 1 from an exponential
# distribution: k = floor(X) + 1 with X ~ Exp(mean = scale)
def _randNumMutations(scale = MUT_NUM_SCALE):
    # a non-positive scale degenerates to a single operation
    if (scale <= 0): return 1
    return int(expovariate(1.0 / scale)) + 1


# one mutation operation: pick a node of fml uniformly at random and replace
# the subtree rooted there with a fresh random subformula of at most
# subfmlSizeLim tokens. Attempts that exceed fmlSizeLim or leave fml
# unchanged are discarded and redrawn from fml, so exactly one replacement
# separates the input from the output.
def _subtreeReplace(fml, fmlSizeLim, subfmlSizeLim, mode):
    while (1):
        fInd = randint(0, fSize(fml) - 1)
        newSubfml = randFML(subfmlSizeLim, mode)
        newFml = _postproc(_replaceSubfmlByInd(fml, fInd, newSubfml))
        if (fSize(newFml) <= fmlSizeLim and newFml != fml): return newFml



# generate a random conditional
'''
def _randCond(sizeLimit = intinf):
    # sanity check
    assert sizeLimit >= 3

    # of size 1
    cond = _randOp(X + XB)
    childL = randFML(sizeLimit - 2)
    childR = randFML(sizeLimit - 1 - fSize(childL))
    resfml = f"({cond} {childL} {childR})"
    return resfml
'''

# Replace a subfml from fml specified by ind
def _replaceSubfmlByInd(fml, ind, newSubfml):
    # Handle base cases
    if (ind == 0): return newSubfml

    # the index of the element we want to replace cannot exceed the size of fml
    if (ind >= fSize(fml)): return fml

    # decrement index by 1
    ind -= 1

    # analyze the current element hct
    [head, inputs] = wrapExtGetHeadAndInputs(fml)

    # if there is no inputs, just return head
    if (len(inputs) == 0): return head

    # depth-first index search
    for inputI in range(len(inputs)):
        # depth-first search the subFml
        subfml = inputs[inputI]
        newSfml = _replaceSubfmlByInd(subfml, ind, newSubfml)

        # index is reduced by size of subfml
        ind -= fSize(subfml)
        
        # if element is found
        if (newSfml != subfml or ind < 0):
            inputs[inputI] = newSfml
            return f"({head} {' '.join(inputs)})"

# replaces all C's with random integers
def _postproc(formula):
    while ("C" in formula):
        formula = formula.replace("C", str(choice(list(range(0, 11)))), 1)

    return formula

# replaces all C's in a list with random integers
def _postproclst(fmllst):
    for i in range(len(fmllst)):
        fmllst[i] = _postproc(fmllst[i])
    return fmllst

# randomly sample any operator
def _randOp(lst = T + U + B):
    if not lst: return None
    choice_val = choice(lst)
    return choice_val

# backup function to get the head and inputs of a fml
#def getHeadInputs(fml):
    # base case
    #if (fml[0] == "("): fml = fml[1:-1]
###################### Some testing scripts #####################
if (__name__ == "__main__"):
    '''
    fml = "(+ f t)"
    print(fml2latex(fml))

    fml = "(sin f)"
    print(fml2latex(fml))
    '''
    # freq_fml = "(cond ((&& (>= t 0.0) (<= t 0.2799054373522459)) (tan (sin (cos (exp (sqrt (tan (neg (neg (cos t)))))))))) ((&& (>= t 0.4085106382978724) (<= t 0.6884160756501182)) (/ (tan 10) (sqrt (+ (neg 6) (sqr 9))))) ((&& (>= t 0.8094562647754138) (<= t 1.0893617021276596)) (neg (sqrt (exp (tan (+ (neg t) (cos (sin 10)))))))) ((&& (>= t 1.210401891252955) (<= t 1.497872340425532)) (/ (tan (- (sqr (cos (tan (neg 5)))) t)) V)) ((&& (>= t 1.6189125295508275) (<= t 1.8988179669030734)) (sqrt (/ (/ (sqrt V) V) t))) ((&& (>= t 2.019858156028369) (<= t 2.307328605200946)) (sqrt (cos (tan (exp (/ t V)))))) ((&& (>= t 2.428368794326241) (<= t 2.7082742316784874)) (neg (+ (tan (/ (tan 9) (+ t t))) V))) ((&& (>= t 2.8293144208037826) (<= t 3.1167848699763594)) (+ (cos (neg 4)) (- (exp (neg (sin t))) V))) (else 0))"
    # amp_fml = "(cond ((&& (>= t 0.0) (<= t 0.2799054373522459)) (tan V)) ((&& (>= t 0.4085106382978724) (<= t 0.6884160756501182)) (sqr (sin (/ 2 f)))) ((&& (>= t 0.8094562647754138) (<= t 1.0893617021276596)) (neg (cos (sqrt V)))) ((&& (>= t 1.210401891252955) (<= t 1.497872340425532)) (neg (+ (+ t (tan (neg V))) t))) ((&& (>= t 1.6189125295508275) (<= t 1.8988179669030734)) (- (sin (sqr (sqrt (* t (sin (exp V)))))) t)) ((&& (>= t 2.019858156028369) (<= t 2.307328605200946)) (sqrt (sqr (sin (cos (tan (- (tan f) t))))))) ((&& (>= t 2.428368794326241) (<= t 2.7082742316784874)) (sqrt (- (neg V) (sqr f)))) ((&& (>= t 2.8293144208037826) (<= t 3.1167848699763594)) f) (else 0))"
   
    # fml = "(cond ((&& (>= t 0.0) (<= t 0.2799054373522459)) (tan (sin (cos (exp (sqrt (tan (neg (neg (cos t)))))))))) ((&& (>= t 0.4085106382978724) (<= t 0.6884160756501182)) (/ (tan 10) (sqrt (+ (neg 6) (sqr 9))))) ((&& (>= t 0.8094562647754138) (<= t 1.0893617021276596)) (neg (sqrt (exp (tan (+ (neg t) (cos (sin 10)))))))) ((&& (>= t 1.210401891252955) (<= t 1.497872340425532)) (/ (tan (- (sqr (cos (tan (neg 5)))) t)) V)) ((&& (>= t 1.6189125295508275) (<= t 1.8988179669030734)) (sqrt (/ (/ (sqrt V) V) t))) ((&& (>= t 2.019858156028369) (<= t 2.307328605200946)) (sqrt (cos (tan (exp (/ t V)))))) ((&& (>= t 2.428368794326241) (<= t 2.7082742316784874)) (neg (+ (tan (/ (tan 9) (+ t t))) V))) ((&& (>= t 2.8293144208037826) (<= t 3.1167848699763594)) (+ (cos (neg 4)) (- (exp (neg (sin t))) V))) (else 0))"
    # fml = "(if (&& (>= t 0.09085753721778567) (<= t 0.15900069013112492)) (neg (/ (neg (* (cos t) f)) 2)) (if (&& (>= t 0.287715534522988) (<= t 0.3407157645666963)) (/ (cos (sin (exp (+ (sin t) (sin f))))) 8) (if (&& (>= t 0.43157330178448194) (<= t 0.49971645469782117)) (- (/ (tan (sqrt f)) 5) 0) (if (&& (>= t 0.575431069045976) (<= t 0.5905739919156069)) (neg (tan (/ (cos (neg (/ (exp t) t))) 7))) (if (&& (>= t 0.6435742219593151) (<= t 0.6511456833941306)) (- (tan (+ (/ (* t 5) 4) f)) 4) (if (&& (>= t 0.7420032206119163) (<= t 0.8707180650037794)) (neg (sqr (cos (* (sqr (neg (tan t))) (sqrt f))))) (if (&& (>= t 0.9085753721778567) (<= t 0.9994329093956423)) (sqr (sqrt (sin (/ (exp (sin (tan t))) 9)))) (if (&& (>= t 1.0827189851786125) (<= t 1.1811479838312138)) (sqr (- 1 (sqrt (tan (cos (- (cos t) f)))))) (if (&& (>= t 1.2038623681356602) (<= t 1.3250057510927078)) (/ (sqrt (tan (tan t))) (sqr 3)) (if (&& (>= t 1.3552915968319696) (<= t 1.4537205954845707)) (* (/ (+ t t) 3) t) (if (&& (>= t 1.4840064412238325) (<= t 1.5597210555719874)) (/ (sin (* (* (/ t t) f) 2)) 6) (if (&& (>= t 1.6051498241808801) (<= t 1.620292747050511)) (* (sin (tan (exp (tan (cos t))))) t) (if (&& (>= t 1.6430071313549575) (<= t 1.7111502842682969)) (sqr (sin (tan t))) (if (&& (>= t 1.7490075914423742) (<= t 1.908008281573499)) (sin 3) (if (&& (>= t 1.9458655887475764) (<= t 2.036723125965362)) (* (sin t) (/ (sqrt t) (+ (exp f) t))) (if (&& (>= t 2.0745804331394395) (<= t 2.195723816096487)) (sin (+ (neg (* 1 (sqr t))) t)) (if (&& (>= t 2.2335811232705645) (<= t 2.294152814749088)) (tan (exp (sqrt (exp (/ (cos (- t 5)) f))))) (if (&& (>= t 2.4152961977061356) (<= t 2.4758678891846597)) (sqrt (* t (tan (sqr (cos 8))))) 0))))))))))))))))))"
    # fml = '(cond ((&& (>= t 0.2350406062318431) (<= t 0.2653684263907906)) (+ (neg (- (tan t) (- f f))) t)) ((&& (>= t 0.3866797070265806) (<= t 0.424589482225265)) (/ (sin (sin (sqr (sqr (neg (- 5 t)))))) 5)) ((&& (>= t 0.6293022682981606) (<= t 0.6823759535763188)) (tan (tan (neg 1)))) ((&& (>= t 0.6975398636557925) (<= t 0.7202857287750031)) (- (sin (neg t)) (* (+ (exp f) t) t))) ((&& (>= t 0.8036872342121087) (<= t 0.8567609194902669)) (sin (* (* (cos (tan t)) f) t))) ((&& (>= t 0.8946706946889512) (<= t 0.9174165598081618)) (neg (- (* (tan f) t) f))) ((&& (>= t 1.0159819753247412) (<= t 1.0235639303644781)) (exp (/ (* 1 3) (neg t)))) ((&& (>= t 1.0690556606028994) (<= t 1.0766376156426363)) (* (- (sqrt t) (cos 6)) (sqrt t))) ((&& (>= t 1.152457166040005) (<= t 1.1600391210797418)) (tan (sqr (* (sqr (sin (sqrt f))) 1)))) ((&& (>= t 1.1752030311592156) (<= t 1.2586045365963212)) (exp (- (+ 4 (- (cos (sqrt t)) 2)) 8))) ((&& (>= t 1.5618827381857963) (<= t 1.57704664826527)) (* (exp (neg (exp (+ (sqr (exp f)) 6)))) t)) ((&& (>= t 1.5922105583447437) (<= t 1.6680301087421125)) (* (neg (sqr (sqr (exp t)))) (sqrt (tan (sqr 4))))) ((&& (>= t 1.6831940188215861) (<= t 1.7741774792984286)) (cos (cos (- (sqrt (sin 4)) (exp (cos (sin 8))))))) ((&& (>= t 1.8575789847355344) (<= t 1.9713083103315874)) (cos (neg 4))) ((&& (>= t 2.001636130490535) (<= t 2.2215128266429045)) (tan (/ (cos (sqrt (- (sin (sqr t)) f))) f))) ((&& (>= t 2.2821684669607993) (<= t 2.289750422000536)) (sqrt (/ (- t (tan (sqr (sqrt (sqr t))))) 7))) ((&& (>= t 2.312496287119747) (<= t 2.373151927437642)) (* (sqr (neg (* (neg (cos t)) t))) f)) (else 0))'

    # [head, inputs] = wrapExtGetHeadAndInputs(fml)
    # print(head, inputs)
    # print(fml2latex(fml))