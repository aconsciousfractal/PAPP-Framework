import argparse

# Data: Free Amino Acid Masses (Daltons)
# Source: Standard Monoisotopic Masses for Free AA (approx)
AA_MASSES = {
    'G': ('Gly', 75.07),
    'A': ('Ala', 89.09),
    'S': ('Ser', 105.09),
    'P': ('Pro', 115.13),
    'V': ('Val', 117.15),
    'T': ('Thr', 119.12),
    'C': ('Cys', 121.16),
    'I': ('Ile', 131.17),
    'L': ('Leu', 131.17),
    'N': ('Asn', 132.12),
    'D': ('Asp', 133.10),
    'Q': ('Gln', 146.15),
    'K': ('Lys', 146.19),
    'E': ('Glu', 147.13),
    'M': ('Met', 149.21),
    'H': ('His', 155.15), # 155 approx
    'F': ('Phe', 165.19),
    'R': ('Arg', 174.20),
    'Y': ('Tyr', 181.19),
    'W': ('Trp', 204.23)
}

def generate_fib_lucas(limit=300):
    """Generates Fibonacci and Lucas series up to a mass limit."""
    fib = [0, 1]
    while fib[-1] < limit:
        fib.append(fib[-1] + fib[-2])
    
    lucas = [2, 1]
    while lucas[-1] < limit:
        lucas.append(lucas[-1] + lucas[-2])
        
    # Remove early duplicates/zeros for matching purposes
    fib_set = sorted(list(set([x for x in fib if x > 50])))
    lucas_set = sorted(list(set([x for x in lucas if x > 50])))
    
    return fib_set, lucas_set

def find_resonance(mass, series, name="Series"):
    """Finds nearest number in series and deviation."""
    best_target = -1
    min_dev = float('inf')
    
    for val in series:
        dev = abs(mass - val)
        if dev < min_dev:
            min_dev = dev
            best_target = val
            
    return best_target, min_dev

def map_amino_acids():
    fib, lucas = generate_fib_lucas()
    print(f"Fibonacci Targets (>50): {fib}")
    print(f"Lucas Targets (>50): {lucas}")
    print("-" * 60)
    print(f"{'AA':<5} {'Mass':<8} | {'Fib Match':<10} {'Dev':<6} | {'Luc Match':<10} {'Dev':<6} | {'Best':<6}")
    print("-" * 60)
    
    golden_aas = []
    
    for code, (name, mass) in AA_MASSES.items():
        f_target, f_dev = find_resonance(mass, fib)
        l_target, l_dev = find_resonance(mass, lucas)
        
        # Scoring: Lower deviation is better
        best_dev = min(f_dev, l_dev)
        is_fib = f_dev < l_dev
        
        best_type = "Fib" if is_fib else "Luc"
        
        print(f"{name} ({code}) {mass:<8.2f} | {f_target:<10} {f_dev:<6.2f} | {l_target:<10} {l_dev:<6.2f} | {best_type}")
        
        if best_dev < 2.0: # Rigid tolerance
            golden_aas.append(code)
            
    print("-" * 60)
    print(f"Golden Amino Acids (Delta < 2.0 Da): {golden_aas}")
    return golden_aas

def generate_sequence(length, pool):
    """Generates a sequence from the Golden Pool."""
    # Simple strategy: Repeat the pool to fill length?
    # Or random walk?
    import random
    seq = []
    for _ in range(length):
        seq.append(random.choice(pool))
    return "".join(seq)

def generate_patterned_sequence(length, pool):
    """
    Generates a sequence with a Fibonacci-like pattern.
    e.g. Concatenation of Golden AAs strings inspired by Fib words?
    Let's just alternate for now to maximize variety within the golden set.
    """
    seq = []
    while len(seq) < length:
        for residue in pool:
            seq.append(residue)
            if len(seq) >= length: break
    return "".join(seq)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--len", type=int, default=60, help="Sequence Length")
    parser.add_argument("--out", type=str, default="golden_sequence.fasta")
    args = parser.parse_args()
    
    golden_pool = map_amino_acids()
    
    # Priority Overrides based on user theory
    # Gly (75) -> L9 (76)
    # Ala (89) -> F11 (89)
    # Trp (204) -> L11? (User mentioned Trp)
    
    # Ensure Gly and Ala are in pool (they should be naturally)
    
    sequence = generate_patterned_sequence(args.len, golden_pool)
    print(f"\nGenerated Golden Sequence ({len(sequence)} aa):")
    print(sequence)
    
    with open(args.out, "w") as f:
        f.write(f">Golden_Sequence_L{args.len}_PAPP_Resonance\n")
        f.write(sequence + "\n")
    print(f"\nSaved to {args.out}")
