import numpy as np
from fractions import Fraction as frac

from Ham import *

def pi_axis_formatter(val, pos, denomlim=10, pi=r'\pi'):
    """Formats axis ticks with fractions of pi."""
    minus = "-" if val < 0 else ""
    val = abs(val)
    ratio = frac(val / np.pi).limit_denominator(denomlim)
    n, d = ratio.numerator, ratio.denominator

    if n == 0:
        return "$0$"
    elif d == 1:
        return f"${minus}{n}{pi}$"
    else:
        return f"${minus}\\frac{{{n}{pi}}}{{{d}}}$"

def step(x):
    """Step function: returns 1 if x >= 0, otherwise 0."""
    return 1.0 if x >= 0 else 0.0

def kronecker_delta(p, q):
    """Kronecker delta function."""
    return 1.0 if p == q else 0.0

def print_formatted_matrix(matrix):
    """
    Prints a 2D matrix in the specified format.
    Args:
        matrix (list or numpy.ndarray): A 2D matrix.
    """
    # Ensure matrix is a numpy array for consistency in handling
    if not isinstance(matrix, np.ndarray):
        matrix = np.array(matrix)

    rows, cols = matrix.shape

    # Print the header row
    print("    i |" + "".join(f"   {col:^4}   |" for col in range(cols)))

    # Print the separator line
    print("  j   |" + "-" * (9 * cols))

    # Print each row with indices
    for i in range(rows):
        formatted_row = " | ".join(f"{matrix[i, j]:.3f}" for j in range(cols))
        print(f"  {i:<3} | {formatted_row} |")

def find_pairs(array):
    """
    Finds pairs of indices (i, j) in the array where array[j] is the negative of array[i].
    Identifies degenerate pairs and returns unique pairs including one from each degenerate group.
    """
    pairs = []
    used_indices = set()  # To track already paired indices
    abs_value_map = {}    # To group pairs by absolute values

    for i, value in enumerate(array):
        if i in used_indices:
            continue

        # Find the negative counterpart
        for j in range(i + 1, len(array)):
            if j in used_indices:
                continue

            if np.allclose(array[j], -value):
                pair = [i, j]
                pairs.append(pair)
                used_indices.add(i)
                used_indices.add(j)

                # Group by absolute value
                abs_val = round(abs(value), 8)  # Round for numerical stability
                if abs_val not in abs_value_map:
                    abs_value_map[abs_val] = []
                abs_value_map[abs_val].append(pair)
                break

    # Separate degenerate and unique pairs
    degenerate_pairs = []
    unique_pairs = []
    for abs_val, grouped_pairs in abs_value_map.items():
        if len(grouped_pairs) > 1:  # More than one pair for the same absolute value
            degenerate_pairs.append(grouped_pairs)
            unique_pairs.append(grouped_pairs[0])  # Include one representative
        else:
            unique_pairs.extend(grouped_pairs)

    return np.array(degenerate_pairs), np.array(unique_pairs)

def simpson_integrate(func, args, T_spc, rule="3/8"):
    """General implementation of Simpson's 1/3, 3/8, and Euler methods for numerical integration."""
    # Assume uniform T_spc spacing
    dt = np.abs(T_spc[1] - T_spc[0])  # T_spc step

    # Number of T_spc steps
    n_steps = len(T_spc)

    # Set up rule-specific coefficients
    if rule == "1/3":
        step = 2  # Simpson's 1/3 rule requires an even number of intervals
        coeffs = [1, 4, 1]
        factor = dt / 3.0
    elif rule == "3/8":
        step = 3  # Simpson's 3/8 rule requires multiples of 3 intervals
        coeffs = [1, 3, 3, 1]
        factor = (3.0 / 8.0) * dt
    elif rule == "euler":
        coeffs = [1]  # Euler's method has no weighted sum, just one term
        factor = dt  # As per Euler method definition
    else:
        raise ValueError("Unsupported rule. Use '1/3', '3/8' or 'euler'.")

    # Initialize integral result and temporary storage for function values
    I = np.zeros_like(func(*args, T_spc[0]))  # Assuming V_time returns a numpy array
    f = np.zeros((len(coeffs), *I.shape), dtype=np.cdouble)  # Shape matches `V_time` outputs

    if rule == "euler":
        # Euler's method
        for t in T_spc:
            I += func(*args, t) * dt
    else:
        # Perform integration using Simpson's rule (1/3 or 3/8)
        for j in range((n_steps - 1) // step):
            # Accumulate the weighted sum of the function evaluations
            for k, coeff in enumerate(coeffs):
                idx = step * j + k
                if idx < n_steps:
                    t_current = T_spc[idx]
                else:
                    # Extrapolate T_spc by adding dt
                    t_current = T_spc[-1] + dt
                f[k] = func(*args, t_current)
            I += sum(coeff * f_val for coeff, f_val in zip(coeffs, f)) * factor

    return I