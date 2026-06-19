import os
import sys

def check_imports():
    print("Checking dependencies...")
    dependencies = {
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
        "pydantic": "pydantic",
        "pandas": "pandas",
        "sklearn": "scikit-learn",
        "joblib": "joblib",
        "numpy": "numpy",
        "requests": "requests"
    }
    
    missing = []
    for module_name, pip_name in dependencies.items():
        try:
            __import__(module_name)
            print(f"  [OK] {module_name} is installed.")
        except ImportError:
            print(f"  [ERROR] {module_name} is MISSING.")
            missing.append(pip_name)
            
    return missing

def check_models():
    print("\nChecking pre-trained models and encoders...")
    models_dir = "models"
    required_files = [
        "health_model.pkl",
        "failure_model.pkl",
        "vehicle_encoder.pkl",
        "health_encoder.pkl",
        "failure_encoder.pkl"
    ]
    
    if not os.path.exists(models_dir):
        print(f"  [ERROR] Models directory '{models_dir}' is MISSING.")
        return False
        
    missing_files = []
    for f in required_files:
        path = os.path.join(models_dir, f)
        if os.path.exists(path):
            print(f"  [OK] Found {f}")
        else:
            print(f"  [ERROR] Missing {f}")
            missing_files.append(f)
            
    if missing_files:
        return False
        
    # Attempt to load them to verify no joblib/scikit-learn version issues
    print("\nAttempting to load models...")
    try:
        import joblib
        for f in required_files:
            path = os.path.join(models_dir, f)
            joblib.load(path)
            print(f"  [OK] Successfully loaded {f}")
        return True
    except Exception as e:
        print(f"  [ERROR] Failed to load models: {e}")
        print("      This usually indicates a scikit-learn or joblib version mismatch.")
        return False

def main():
    print("=" * 50)
    print("Hydraulic EV Predictive Maintenance API - Verification")
    print("=" * 50)
    
    missing_deps = check_imports()
    models_ok = check_models()
    
    print("\n" + "=" * 50)
    if not missing_deps and models_ok:
        print("VERIFICATION SUCCESSFUL: Everything is ready to run!")
        print("You can start the server using:")
        print("  uvicorn app:app --reload")
        print("And run the test script using:")
        print("  python test_api.py")
    else:
        print("VERIFICATION FAILED:")
        if missing_deps:
            print(f"  - Install missing dependencies: pip install {' '.join(missing_deps)}")
        if not models_ok:
            print("  - Please ensure all .pkl files are placed inside the 'models/' directory.")
    print("=" * 50)

if __name__ == "__main__":
    main()
