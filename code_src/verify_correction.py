import numpy as np

PHI = (1 + np.sqrt(5)) / 2

def lucas(n):
    return int(round(PHI**n + (-PHI)**(-n)))

def verify_rectification():
    print("--- UNIVERSAL RECTIFICATION VERIFICATION ---")
    print(f"{'n':<4} {'L_n':<12} {'Correction Factor':<20} {'Error (%)':<15} {'Status'}")
    print("-" * 65)
    
    X_exact = (PHI**0.5 * 12**0.25) / np.pi**0.5
    
    for n in range(1, 41):
        Ln = lucas(n)
        
        # 1. Physical Radius (Based on Lucas)
        # Assuming R scales as L_n^(1/4) * constant
        # Note: In 123-Lock, R^4 approx 123 * factor
        # So we verify if L_n^(1/4) matches the corrected formula
        
        R_physical = X_exact * PHI**2 * (Ln / 123)**0.25 
        # Wait, let's use the USER's formula directly:
        # R_phys(n) = (root4(12) * phi^(n/4)) / root(pi) * CORRECTION
        
        R_divine = (12**0.25 * PHI**(n/4)) / np.pi**0.5
        
        term = (-1)**n / (PHI**(2*n))
        correction = (1 + term)**0.25
        
        R_corrected = R_divine * correction
        
        # Now compare R_corrected with "Actual" R derived from L_n
        # Actual R derived from L_n would be replacing phi^n with L_n in the base scaling?
        # R_actual = (12**0.25 * Ln**(1/4)) / np.pi**0.5
        
        # Let's verify the identity: L_n^(1/4) == phi^(n/4) * correction
        # L_n = phi^n + (-1/phi)^n = phi^n (1 + (-1)^n phi^(-2n))
        # So L_n^(1/4) = phi^(n/4) * (1 + ...)^(1/4)
        # It is an algebraic identity.
        
        val_Ln = Ln**(0.25)
        val_Calc = PHI**(n/4) * correction
        
        delta = abs(val_Ln - val_Calc)
        
        # Calculate Error between Physical (L_n) and Archetype (Phi^n) without correction
        # This shows why low n are unstable.
        err_raw = abs(Ln**(0.25) - PHI**(n/4)) / Ln**(0.25) * 100
        
        status = "LOCK" if err_raw < 0.0001 else "UNSTABLE"
        if n == 10: status = "**METATRON**"
        if n == 18: status = "**TIER 0**"
        
        print(f"{n:<4} {Ln:<12} {correction:.8f}           {err_raw:.6f}%        {status}")

if __name__ == "__main__":
    verify_rectification()
