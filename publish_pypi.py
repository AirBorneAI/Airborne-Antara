
import os
import sys
import shutil
import subprocess
import getpass
import time
from pathlib import Path

# Colors for output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(msg):
    print(f"\n{Colors.HEADER}{Colors.BOLD}=== {msg} ==={Colors.ENDC}")

def print_step(msg):
    print(f"{Colors.CYAN}➜ {msg}{Colors.ENDC}")

def print_success(msg):
    print(f"{Colors.GREEN}✓ {msg}{Colors.ENDC}")

def print_error(msg):
    print(f"{Colors.FAIL}✗ Error: {msg}{Colors.ENDC}")

def check_dependencies():
    """Ensure build and twine are installed"""
    print_header("Checking Dependencies")
    
    required = ["build", "twine"]
    missing = []
    
    for tool in required:
        try:
            importlib = __import__("importlib.util")
            if importlib.util.find_spec(tool) is None:
                # Some packages have different import names, but these are tools usually run as modules
                # Let's try running them to check
                subprocess.check_call([sys.executable, "-m", tool, "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            missing.append(tool)
    
    if missing:
        print_step(f"Installing missing tools: {', '.join(missing)}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
            print_success("Dependencies installed")
        except subprocess.CalledProcessError:
            print_error("Failed to install dependencies. Please run: pip install build twine")
            sys.exit(1)
    else:
        print_success("All dependencies found")

def clean_build_artifacts():
    """Clean dist/ and build/ directories"""
    print_header("Cleaning Build Artifacts")
    
    dirs_to_clean = ["dist", "build", "antara.egg-info", "airborne_antara.egg-info"]
    
    for d in dirs_to_clean:
        path = Path(d)
        if path.exists():
            print_step(f"Removing {d}...")
            shutil.rmtree(path)
            print_success(f"Removed {d}")
        else:
            print_step(f"{d} not found (clean)")

def build_package():
    """Build sdist and wheel"""
    print_header("Building Package")
    
    try:
        cmd = [sys.executable, "-m", "build"]
        print_step("Running build backend...")
        subprocess.check_call(cmd)
        print_success("Package built successfully")
        
        # Verify output
        dist = Path("dist")
        files = list(dist.glob("*"))
        print_step(f"Generated artifacts ({len(files)}):")
        for f in files:
            print(f"  - {f.name} ({f.stat().st_size / 1024:.1f} KB)")
            
    except subprocess.CalledProcessError:
        print_error("Build failed")
        sys.exit(1)

def validate_package():
    """Run twine check"""
    print_header("Validating Package")
    
    try:
        cmd = [sys.executable, "-m", "twine", "check", "dist/*"]
        print_step("Running twine check...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(result.stdout)
            print_success("Package validation passed")
        else:
            print(result.stdout)
            print(result.stderr)
            print_error("Package validation failed")
            sys.exit(1)
            
    except Exception as e:
        print_error(f"Validation error: {e}")
        sys.exit(1)

def publish_to_pypi():
    """Upload to PyPI"""
    print_header("Publishing to PyPI")
    
    print(f"{Colors.WARNING}⚠️  You are about to upload to PyPI.{Colors.ENDC}")
    print("Please ensure you have increased the version number in pyproject.toml if this is a new release.")
    
    confirm = input(f"\n{Colors.BOLD}Continue? (y/N): {Colors.ENDC}")
    if confirm.lower() != 'y':
        print_error("Aborted by user")
        sys.exit(0)
    
    print("\nPlease enter your PyPI API token.")
    print("It should start with 'pypi-'")
    
    try:
        token = getpass.getpass(prompt=f"{Colors.BLUE}API Token: {Colors.ENDC}")
    except Exception:
        # Fallback for some terminals
        token = input(f"{Colors.BLUE}API Token (hidden input failed): {Colors.ENDC}")
        
    if not token.strip():
        print_error("Empty token provided")
        sys.exit(1)
        
    try:
        # Construct command
        # twine upload dist/* -u __token__ -p <token>
        cmd = [
            sys.executable, "-m", "twine", "upload", 
            "dist/*",
            "-u", "__token__",
            "-p", token,
            "--verbose"
        ]
        
        print_step("Uploading to PyPI...")
        subprocess.check_call(cmd)
        
        print_header("🚀 SUCCESS")
        print_success("Package successfully published to PyPI!")
        print(f"\nView it at: https://pypi.org/project/antara/")
        
    except subprocess.CalledProcessError:
        print_error("Upload failed")
        sys.exit(1)

def main():
    try:
        # Ensure we are in root
        if not Path("pyproject.toml").exists():
            print_error("pyproject.toml not found. Are you in the root directory?")
            sys.exit(1)
            
        print_header("ANTARA Publishing Utility")
        
        check_dependencies()
        clean_build_artifacts()
        build_package()
        validate_package()
        publish_to_pypi()
        
    except KeyboardInterrupt:
        print("\n\nAborted by user")
        sys.exit(1)

if __name__ == "__main__":
    main()
