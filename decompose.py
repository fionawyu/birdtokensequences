## This script is used to decompose the formula into time segments and subformulae.
from funcWrappers import wrapExtGetSubfmlByInd, wrapExtGetHeadAndInputs, wrapExtParseEvalTwoVarBatch, wrapExtParseEvalBatch
from funcWrappers import wrapExtParseEvalTwoVar, wrapExtParseEval
import numpy as np
from time import time
import scipy.io as sio
import os
import re
import yaml
from pathlib import Path
import sys

BASE_DIR = Path(os.getenv("FML_RUN_DIR", "./1017_plots47_feebee_1017")).resolve()

def extract_clauses(fml_str):
    if not fml_str.startswith('(cond'):
        raise ValueError("Formula string must start with '(cond'")

    clauses = []
    i = fml_str.find('(cond') + len('(cond')
    depth = 0
    current_clause = ''
    
    while i < len(fml_str):
        char = fml_str[i]
        
        if char == '(':
            if depth == 0:
                current_clause = ''
            depth += 1
            current_clause += char
        elif char == ')':
            current_clause += char
            depth -= 1
            if depth == 0:
                # Avoid catching (else 0)
                if not current_clause.startswith('(else'):
                    clauses.append(current_clause)
        else:
            if depth > 0:
                current_clause += char
        i += 1

    return clauses


def decomp_fml_string(fml_str):
    """
    Parse formula string of form '(cond (...)(...)(...)(else 0))'
    Extracts time segments and subformulae from each clause using head/rest decomposition.
    """
    clauses = extract_clauses(fml_str)
    segments = []
    subfmls = []

    for clause in clauses:
        # Decompose into head (&& block) and rest (subformula)
        head, rest = wrapExtGetHeadAndInputs(clause)

        # Extract time boundaries from head block (assuming it's of form (&& (>= t ...) (<= t ...)))
        t1, t2 = None, None
        _, time_conditions = wrapExtGetHeadAndInputs(head)  # Get conditions inside && block

        for cond in time_conditions:
            cond_head, cond_args = wrapExtGetHeadAndInputs(cond)
            if cond_head == '>=' and cond_args[0] == 't':
                t1 = float(cond_args[1])
            elif cond_head == '<=' and cond_args[0] == 't':
                t2 = float(cond_args[1])

        segments.append((t1, t2))
        # Instead of appending parsed tree/list, convert back to string
        subfmls.append(rest[0])
        # Instead of appending parsed tree/list, convert back to string
        # raw_subfml = rest_to_string(rest, remove_outer=True)
        # subfmls.append(raw_subfml)

    return segments, subfmls

def decomp_fml_string_with_silence(fml_str, t0=0.0):
    segments, subfmls = decomp_fml_string(fml_str)
    # print(f"Notes segments: {segments}")
    # print(f"Notes subfmls: {subfmls}")

    clean = []
    for (t1, t2), f in zip(segments, subfmls):
        if t1 is None or t2 is None or t2 <= t1:
            continue
        clean.append((float(t1), float(t2), str(f)))

    if not clean:
        return [], []

    clean.sort(key=lambda x: x[0])

    t_end = max(e for _, e, _ in clean)

    clipped = []
    for s, e, f in clean:
        ss, ee = max(s, t0), min(e, t_end)
        if ee > ss:
            clipped.append((ss, ee, f))
    if not clipped:
        return [], []

    timeline = []
    eps = 1e-12

    cur = t0
    prev_end = cur

    # NOTE: removed the pre-loop "leading silence" block

    for s, e, f in clipped:
        if s > prev_end + eps:
            timeline.append((prev_end, s, "0"))
        start = max(s, prev_end)
        if e > start + eps:
            timeline.append((start, e, f))
            prev_end = e
        else:
            prev_end = max(prev_end, e)

    if prev_end < t_end - eps:
        timeline.append((prev_end, t_end, "0"))

    timeline.sort(key=lambda x: x[0])

    merged_segments, merged_subfmls = [], []
    for s, e, f in timeline:
        if not merged_segments:
            merged_segments.append((s, e))
            merged_subfmls.append(f)
        else:
            ps, pe = merged_segments[-1]
            pf = merged_subfmls[-1]
            if f == pf and abs(s - pe) <= eps:
                merged_segments[-1] = (ps, max(pe, e))
            else:
                s = max(s, pe) if s < pe and f != pf else s
                if e > s + eps:
                    merged_segments.append((s, e))
                    merged_subfmls.append(f)

    return merged_segments, merged_subfmls

def load_formulae(region, mat_fml_path=None):
    """
    Load formulae from the MATLAB .mat file.
    """
    # mat_fml_path = f'plots/{region}/{region}_formulae.mat'
    # mat_fml_path = f'./0801_plots47_feebee_1001/{region}/formulae/{region}_formulae.mat'
    # mat_fml_path = f'./0801_plots47_feebee/west/formulae/W0a 5947 S21 M 2023.10.11 Feebee 5_formulae.mat'

    if not os.path.exists(mat_fml_path):
        raise FileNotFoundError(f"File not found: {mat_fml_path}")
    print(f"Loading MAT file from: {mat_fml_path}")
    mat_fml = sio.loadmat(mat_fml_path)

    # Extract MATLAB variables correctly
    spec_formulae = mat_fml.get('specfml_lst', None)
    # amp_formulae = "1"
    amp_formulae = mat_fml.get('ampfml_lst', None)

    if spec_formulae is None or amp_formulae is None:
        raise ValueError("MAT file does not contain expected variables 'specfml_lst' or 'ampfml_lst'.")

    # Debug prints AFTER extracting them
    print(f"Type of spec_formulae: {type(spec_formulae)}")
    print(f"Shape of spec_formulae: {spec_formulae.shape if hasattr(spec_formulae, 'shape') else 'no shape'}")
    print(f"Raw spec_formulae content: {spec_formulae}")

    # Ensure data is extracted properly as strings
    spec_formulae = [str(item[0]) for item in spec_formulae] if spec_formulae.ndim > 1 else [str(spec_formulae[0])]
    amp_formulae = [str(item[0]) for item in amp_formulae] if amp_formulae.ndim > 1 else [str(amp_formulae[0])]

    return spec_formulae, amp_formulae

def main():
    regions = ['west']
    results = {}

    for region in regions:
        try:
            mat_fml_path = BASE_DIR/region/f"{region}_formulae.mat"

            spec_formulae, amp_formulae = load_formulae(region, mat_fml_path)
        except (FileNotFoundError, ValueError) as e:
            print(f"Error loading data for region '{region}': {e}")
            continue

        spec_results = []
        for spec_formula in spec_formulae:
            spec_formula_str = spec_formula.strip()
            print(f"Processing formula: {spec_formula_str}")
            try:
                result = decomp_fml_string_with_silence(spec_formula_str)
                spec_results.append(result)
            except Exception as e:
                print(f"Error processing spec formula: {e}")

        amp_results = []
        for amp_formula in amp_formulae:
            amp_formula_str = amp_formula.strip()
            try:
                result = decomp_fml_string_with_silence(amp_formula_str)
                amp_results.append(result)
            except Exception as e:
                print(f"Error processing amp formula: {e}")

        results[region] = {
            'spec': spec_results,
            'amp': amp_results
        }


    def to_builtin(obj):
        """Recursively convert NumPy types to plain Python for YAML."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer, np.bool_)):
            return obj.item()
        if isinstance(obj, dict):
            return {k: to_builtin(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [to_builtin(v) for v in obj]
        return obj

    # Save results as .yml
    with open(BASE_DIR/region/"results.yml", 'w') as f:
        yaml.safe_dump(
            {"results": to_builtin(results)},
            f,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False
        )
    print("Results saved to 'results.yml'")


def test_with_sample_fml():
    
    # Test with the sample formula
    # sample_fml = '(cond ((&& (>= t 1.103) (<= t 4.485)) (tan 3.5)) ((&& (>= t 5.663) (<= t 9.309)) (log 1.4)) (else 0))'
    # sample_fml = '(cond ((&& (>= t 1.251) (<= t 1.451)) (sin (+ (sin (+ (* 0.8 t) 1)) 6))) 
    # ((&& (>= t 3.090) (<= t 3.566)) (sin (+ (* 0.8 t) 2))) ((&& (>= t 3.666) (<= t 3.991)) 
    # (sin (exp (sin (tan (sin (+ (* 0.5 t) 1))))))) ((&& (>= t 5.292) (<= t 6.193)) (sin
    # (+ (* 0.6 t) 2))) ((&& (>= t 6.894) (<= t 7.945)) (abs (sin (+ (sin (+ (* 0.6 t)
    # 3)) 6)))) ((&& (>= t 8.633) (<= t 9.734)) (sin (+ (sin (+ (* 0.9 t) 8)) 2))) ((&&
    # (>= t 10.447) (<= t 10.622)) (sin (+ (sin (+ (* 0.2 t) 3)) t))) ((&& (>= t 10.747)
    # (<= t 11.510)) (sin (+ (sin (+ (* 0.9 (sin t)) 2)) 2))) ((&& (>= t 12.323) (<= t
    #  12.961)) (sin (exp (sin (+ (* 0.9 t) 4))))) ((&& (>= t 13.049) (<= t 13.199)) (sin
    # (+ (sin (+ (* 0.9 t) 4)) 1))) (else 0))'
    [
  '(sin (+ (sin (+ (* 0.8 t) 1)) 6))',
  '(sin (+ (* 0.8 t) 2))',
  '(sin (exp (sin (tan (sin (+ (* 0.5 t) 1))))))',
  '(sin (+ (* 0.6 t) 2))',
  '(abs (sin (+ (sin (+ (* 0.6 t) 3)) 6)))',
  '(sin (+ (sin (+ (* 0.9 t) 8)) 2))',
  '(sin (+ (sin (+ (* 0.2 t) 3)) t))',
  '(sin (+ (sin (+ (* 0.9 (sin t)) 2)) 2))',
  '(sin (exp (sin (+ (* 0.9 t) 4))))',
  '(sin (+ (sin (+ (* 0.9 t) 4)) 1))',
]

    print("\n=== Testing with sample formula ===")
    
    # Call the function
    segments, subfmls = decomp_fml_string(sample_fml)
    # segments, subfmls = decomp_fml_string_with_silence(sample_fml, t0=0.0)
    
    # Print extracted time segments
    print("\nExtracted Time Segments:")
    for segment in segments:
        print(segment)
    
    # Print corresponding subformulae
    print("\nCorresponding Subformulae:")
    for subfml in subfmls:
        print(subfml)
    
    return segments, subfmls

if __name__ == '__main__':
    # Uncomment to run tests
    test_with_sample_fml()
    # main()
