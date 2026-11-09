# function to convert formula in lisp notation to latex
from funcWrappers import wrapExtGetHeadAndInputs, wrapExtParseEvalBatch, wrapExtParseEvalTwoVarBatch
from grammar import T, X, XB, I, U, B

def fml2latex(fml):
    # get the first and rest of the formula
    [head, inputs] = wrapExtGetHeadAndInputs(fml)
    # remove blank string from inputs
    inputs = [inp for inp in inputs if inp != ""]
    # inputs_lst = []
    # for inp in inputs:
    #     if inp:
    #         inputs_lst.append(fml2latex(inp))

    # base case: terminal
    if (len(inputs) == 0): return head

    # inductive case
    if (head in U):
        # sanity check
        assert(len(inputs) == 1)
        match head:
            case "sqr": return f"{fml2latex(inputs[0])}^2"
            case "sqrt": return f"\\sqrt{{{fml2latex(inputs[0])}}}"
            case "exp": return f"\\exp{{{fml2latex(inputs[0])}}}"
            case "neg": return f"\\left(-{fml2latex(inputs[0])}\\right)"
            case "sin": return f"\\sin\\left({fml2latex(inputs[0])}\\right)"
            case "tan": return f"\\tan\\left({fml2latex(inputs[0])}\\right)"
            case "cos": return f"\\cos\\left({fml2latex(inputs[0])}\\right)"
            case "log": return f"\\log\\left({fml2latex(inputs[0])}\\right)"
            case "abs": return f"\\left|{fml2latex(inputs[0])}\\right|"
            case _: print(f"Unary operator not supported: {head}")
    elif (head in (B + X + XB)):
        assert(len(inputs) == 2)
        match head:
            case "+": return f"{fml2latex(inputs[0])} + {fml2latex(inputs[1])}"
            case "-": return f"{fml2latex(inputs[0])} - {fml2latex(inputs[1])}"
            case "*": return f"{fml2latex(inputs[0])} \\times {fml2latex(inputs[1])}"
            case "/": return f"\\frac{{{fml2latex(inputs[0])}}}{{{fml2latex(inputs[1])}}}"
            case "==": return f"{fml2latex(inputs[0])} = {fml2latex(inputs[1])}"
            case ">": return f"{fml2latex(inputs[0])} > {fml2latex(inputs[1])}"
            case "<": return f"{fml2latex(inputs[0])} < {fml2latex(inputs[1])}"
            case ">=": return f"{fml2latex(inputs[0])} \\ge {fml2latex(inputs[1])}"
            case "<=": return f"{fml2latex(inputs[0])} \\le {fml2latex(inputs[1])}"
            case "&&": return f"{fml2latex(inputs[0])} \\land {fml2latex(inputs[1])}"
            case "||": return f"{fml2latex(inputs[0])} \\lor {fml2latex(inputs[1])}"
            case _: print(f"Binary operator not supported: {head}")
    elif (head in I):
        assert(len(inputs) == 3)
        return f"\\begin{{cases}} {fml2latex(inputs[1])} & \\text{{if }} {fml2latex(inputs[0])} \\\\ {fml2latex(inputs[2])} & \\text{{otherwise}}\\end{{cases}}"
    elif (head == "cond"):
        latex = f"\\begin{{cases}}"
        for inI in range(len(inputs) - 1):
            [subhead, subinputs] = wrapExtGetHeadAndInputs(inputs[inI])
            latex = f"{latex} {fml2latex(subinputs[0])} & \\text{{if }} {fml2latex(subhead)} \\\\"
            #print("inputs:", inputs)
            #print("inI is:", 'inI' in locals())

        [_, subinputs] = wrapExtGetHeadAndInputs(inputs[inI + 1])
        latex = f"{latex} {fml2latex(subinputs[0])} & \\text{{otherwise}} \\\\"
        latex = f"{latex} \\end{{cases}}"
        return latex
    
if __name__ == "__main__":
    # test cases
    # test_fmls = ["(cond ((&& (>= t 3.490) (<= t 3.678)) (sin (sin (+ (sin (+ (* 0.5 t) 3)) 9)))) ((&& (>= t 5.229) (<= t 12.573)) (neg (sin (+ (sin (exp (* 3 t))) 5)))) (else 0))"]
    # test_fmls = ["(cond ((&& (>= t 4.787) (<= t 4.987)) (sin (+ (sin (+ t 3)) 9))) ((&& (>= t 5.237) (<= t 5.375)) (sin (+ (abs (* 2.0 t)) 9))) ((&& (>= t 5.900) (<= t 6.087)) (sin (+ (+ (abs (* 2.0 t)) 6) 9))) ((&& (>= t 6.487) (<= t 7.149)) (sin (+ (* 0.3 t) 9))) ((&& (>= t 7.674) (<= t 7.899)) (sin (+ (+ (abs (* 2.0 t)) 10) 5))) ((&& (>= t 8.237) (<= t 8.937)) (sin (+ (* 0.3 t) 9))) ((&& (>= t 9.374) (<= t 9.549)) (neg (sin (+ (sin (+ (* 0.8 t) 7)) 5)))) ((&& (>= t 9.824) (<= t 10.574)) (sin (cos (tan (sin (+ (* 0.5 t) 4)))))) ((&& (>= t 10.912) (<= t 12.374)) (sin (sin (+ (sin (+ (* 0.3 t) 6)) 9)))) ((&& (>= t 12.649) (<= t 14.049)) (sin (+ (sin (+ (* 0.2 t) 7)) 2))) ((&& (>= t 14.374) (<= t 15.349)) (sin (+ (sin (sin (+ (* 0.4 t) 7))) 2))) ((&& (>= t 16.111) (<= t 16.286)) (sin (exp (sin (+ (abs (* 5.0 t)) 5))))) ((&& (>= t 16.811) (<= t 16.974)) (sin (sqrt (+ (abs (* 8.0 t)) 7)))) (else 0))"]
    # test_fmls = ['(exp (sin (+ (* (exp (* 2.0 t)) t) 5)))', '(sin (+ (sin (+ (* 0.9 t) 7)) 6))', '(sin (sin (sqr (cos (sin (+ (* 0.8 t) 7))))))', '(sin (+ (sin (sin (+ (* 0.8 t) 3))) 9))', '(sin (+ (* 0.1 (exp (* 7.0 t))) 9))', '(sin (+ (sin (+ t 6)) 9))', '(sin (sqrt (sin (sin (sin (+ (* 0.5 t) 9))))))', '(exp (sin (+ (* (+ (* 0) 5) t) 10)))', '(sin (+ (sqr (sin (+ (* t t) 3))) 9))', '(sin (sin (exp (sin (exp (* 1.0 t))))))', '(sin (+ (sin (sin (+ (* 0.6 t) 3))) 9))', '(sin (+ t 9))', '(sin (+ (sin (+ (* 0.6 t) 3)) 7))', '(sin (+ (sin (+ (* 0.0 t) 9)) t))', '(sin (+ (sin (+ (* 0.8 t) 7)) 2))', '(sin (exp (sin (+ t 8))))', '(sin (+ (* 0.4 t) 9))']
    # chickadee
    # test_fmls = ['(cond ((&& (>= t 1.091) (<= t 1.680)) (tan (sqr (sin (sin (+ (* 2 t) 1)))))) ((&& (>= t 2.169) (<= t 2.670)) (sin (sqr (sin (sin (+ (* 3 (sin t)) 4)))))) ((&& (>= t 3.673) (<= t 4.726)) (sqr (sin (sin (exp (* 2 (sin 6))))))) ((&& (>= t 5.767) (<= t 6.996)) (sin (sin (tan 5)))) (else 0))']
    # feebee
    # test_fmls = ['(cond ((&& (>= t 0.901) (<= t 4.393)) (sin 2.7)) ((&& (>= t 6.020) (<= t 9.875)) (cos 1.2)) (else 0))']
# tseet
    # test_fmls = ['(cond ((&& (>= t 0.728) (<= t 1.150)) (sin (+ (* (sin (exp (* 4.0 t))) t) 7))) (else 0))']
# wtsp
    test_fmls = ['(cond ((&& (>= t 1.611) (<= t 7.167)) (cos 5.3)) ((&& (>= t 9.340) (<= t 13.473)) (/ 2.3 5.2)) ((&& (>= t 15.334) (<= t 19.055)) (sin 6.6)) ((&& (>= t 20.616) (<= t 23.300)) (tan 6.6)) ((&& (>= t 24.199) (<= t 24.736)) (cos 5.0)) ((&& (>= t 25.848) (<= t 28.120)) (/ 1.7 5.5)) ((&& (>= t 29.032) (<= t 29.494)) (sin 6.6)) ((&& (>= t 30.568) (<= t 32.953)) (sin 6.6)) (else 0))']

    for fml in test_fmls:
        print(f"Formula: {fml}")
        print(f"LaTeX: {fml2latex(fml)}")
        print()
