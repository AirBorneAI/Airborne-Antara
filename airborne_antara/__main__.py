"""
ANTARA - Adaptive Neural Telemetry & Research Assistant
========================================================
Advanced Sci-Fi CLI with Detailed Framework Information

Usage: python -m airborne_antara
"""
import sys
import time
import platform
import importlib
import os
from typing import Dict, Tuple
from pathlib import Path

# Ensure package is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Version info
VERSION = "0.0.9"
FRAMEWORK_NAME = "ANTARA"
TAGLINE = "Adaptive Neural Telemetry & Research Assistant"

# Detailed ASCII Art
LOGO = r"""
   _____    _______________________ __________    _____   
  /  _  \   \      \__    ___/  _  \\______   \  /  _  \  
 /  /_\  \  /   |   \|    | /  /_\  \|       _/ /  /_\  \ 
/    |    \/    |    \    |/    |    \    |   \/    |    \
\____|__  /\____|__  /____|\____|__  /____|_  /\____|__  /
        \/         \/              \/       \/         \/ 
"""

def ensure_rich() -> bool:
    """Self-healing dependency installer for Rich library"""
    try:
        import rich
        return True
    except ImportError:
        print("⚡ Rich library not found. Installing for enhanced display...")
        print("   This only happens once and takes ~5 seconds.\n")
        
        try:
            import subprocess
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "rich", "-q"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            # Invalidate import cache and try again
            importlib.invalidate_caches()
            import rich
            
            print("✅ Rich installed successfully! Restarting with enhanced display...\n")
            time.sleep(1)
            return True
            
        except Exception as e:
            print(f"⚠️  Could not install Rich: {e}")
            print("   Continuing with basic display...\n")
            return False

# Self-healing: Install rich if needed
HAS_RICH = ensure_rich()

# Try to import rich components
if HAS_RICH:
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich import box
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
        from rich.layout import Layout
        from rich.live import Live
        from rich.text import Text
        from rich.markdown import Markdown
        from rich.syntax import Syntax
        from rich.columns import Columns
        console = Console()
    except ImportError as e:
        # If rich imports fail after installation, fall back gracefully
        HAS_RICH = False
        print(f"⚠️  Rich import error: {e}")
        print("   Falling back to basic display...\n")

if not HAS_RICH:
    class SimpleConsole:
        def print(self, *args, **kwargs):
            if args:
                text = str(args[0])
                # Strip rich markup
                for tag in ['[bold]', '[/bold]', '[cyan]', '[/cyan]', '[green]', '[/green]', 
                           '[yellow]', '[/yellow]', '[dim]', '[/dim]', '[red]', '[/red]',
                           '[magenta]', '[/magenta]', '[blue]', '[/blue]']:
                    text = text.replace(tag, '')
                print(text)
        def clear(self):
            os.system('cls' if platform.system() == 'Windows' else 'clear')
    console = SimpleConsole()

def get_system_info() -> Dict[str, str]:
    """Collect comprehensive system telemetry"""
    info = {
        "Operating System": f"{platform.system()} {platform.release()}",
        "Architecture": platform.machine(),
        "Processor": platform.processor() or "Unknown",
        "Python Version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    }
    
    # Check PyTorch
    try:
        import torch
        info["PyTorch Version"] = torch.__version__
        
        if torch.cuda.is_available():
            info["Compute Backend"] = f"NVIDIA CUDA {torch.version.cuda}"
            info["GPU Device"] = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            info["VRAM Available"] = f"{vram:.1f} GB"
            info["CUDA Devices"] = str(torch.cuda.device_count())
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            info["Compute Backend"] = "Apple Metal Performance Shaders"
            info["GPU Device"] = "Apple Neural Engine"
            info["VRAM Available"] = "Unified Memory Architecture"
        else:
            info["Compute Backend"] = "CPU Only (No GPU)"
            info["GPU Device"] = "None Detected"
            info["VRAM Available"] = "N/A"
    except ImportError:
        info["PyTorch Version"] = "❌ Not Installed"
        info["Compute Backend"] = "Unknown"
        info["Status"] = "⚠️  Install PyTorch: pip install torch"
    
    return info

def check_modules() -> Dict[str, Tuple[bool, str]]:
    """Verify core modules with detailed error reporting"""
    modules = {
        "Core Framework": ("airborne_antara.core", "AdaptiveFramework"),
        "Memory System": ("airborne_antara.memory", "UnifiedMemoryHandler"),
        "Consciousness": ("airborne_antara.consciousness_v2", "EnhancedConsciousnessCore"),
        "Meta Controller": ("airborne_antara.meta_controller", "MetaController"),
        "World Model": ("airborne_antara.world_model", "WorldModel"),
        "Health Monitor": ("airborne_antara.health_monitor", "NeuralHealthMonitor"),
        "Adapters": ("airborne_antara.adapters", "AdapterBank"),
        "MoE Router": ("airborne_antara.moe", "HierarchicalMoE"),
    }
    
    status = {}
    for name, (path, class_name) in modules.items():
        try:
            mod = importlib.import_module(path)
            if hasattr(mod, class_name):
                status[name] = (True, "✓ Online")
            else:
                status[name] = (False, f"⚠️  Missing {class_name}")
        except ImportError as e:
            status[name] = (False, f"✗ Import Error")
        except Exception as e:
            status[name] = (False, f"✗ {str(e)[:30]}")
    
    return status

def animate_startup():
    """Enhanced sci-fi startup sequence"""
    if not HAS_RICH:
        print("Initializing ANTARA Framework...")
        time.sleep(1)
        return
    
    startup_steps = [
        ("Initializing neural substrate", 0.4),
        ("Loading cognitive modules", 0.4),
        ("Establishing memory pathways", 0.4),
        ("Activating consciousness layer", 0.4),
        ("Calibrating meta-learning core", 0.4),
        ("Synchronizing world model", 0.3),
        ("Engaging adaptive systems", 0.3),
        ("System online", 0.2),
    ]
    
    with Progress(
        SpinnerColumn(spinner_name="dots", style="bold red"),
        TextColumn("[bold red]{task.description}"),
        BarColumn(complete_style="red", finished_style="bold red"),
        TextColumn("[bold red]{task.percentage:>3.0f}%"),
        transient=True,
    ) as progress:
        task = progress.add_task("[red]Starting up...", total=len(startup_steps))
        
        for desc, delay in startup_steps:
            progress.update(task, description=f"[red]{desc}...")
            time.sleep(delay)
            progress.advance(task)
        
        progress.update(task, description="[bold red]✓ ANTARA Framework Ready")
        time.sleep(0.3)

def animate_logo_reveal():
    """Animated logo reveal effect"""
    if not HAS_RICH:
        return
    
    console.clear()
    
    # Split logo into lines
    logo_lines = LOGO.split('\n')
    colors = ["#400000", "#600000", "#800000", "#A00000", "#C00000", "#E00000", "#FF0000"]
    
    # Reveal each line with animation
    for i, line in enumerate(logo_lines):
        if line.strip():
            color_idx = int((i / len(logo_lines)) * (len(colors) - 1))
            color = colors[min(color_idx, len(colors) - 1)]
            
            # Print line with color
            console.print(line, style=f"bold {color}", justify="center")
            time.sleep(0.08)  # Delay between lines
    
    console.print()  # Spacing after logo
    
    # Typing effect for tagline - Fixed version
    from rich.live import Live
    from rich.align import Align
    
    tagline_display = Text("", justify="center", style="italic red")
    
    with Live(Align.center(tagline_display), refresh_per_second=20, transient=False) as live:
        for char in TAGLINE:
            tagline_display.append(char)
            live.update(Align.center(tagline_display))
            time.sleep(0.03)
    
    console.print()  # New line after typing
    
    # Version text with fade-in effect
    time.sleep(0.2)
    console.print(f"Version {VERSION} | Production Release", 
                 style="dim red", justify="center")
    time.sleep(0.5)


def create_header():
    """Create header with flowing gradient logo (static version for display_info)"""
    if not HAS_RICH:
        return None
    
    # Split logo into lines and apply gradient
    logo_lines = LOGO.split('\n')
    gradient_logo = Text(justify="center")
    
    # Create color gradient: cyan -> blue -> magenta -> purple
    # Create color gradient: Black/Maroon -> Bright Red
    colors = ["#400000", "#600000", "#800000", "#A00000", "#C00000", "#E00000", "#FF0000"]
    
    for i, line in enumerate(logo_lines):
        if line.strip():  # Only apply to non-empty lines
            # Calculate color index based on line position
            color_idx = int((i / len(logo_lines)) * (len(colors) - 1))
            color = colors[min(color_idx, len(colors) - 1)]
            gradient_logo.append(line + "\n", style=f"bold {color}")
        else:
            gradient_logo.append(line + "\n")
    
    # Tagline with gradient
    tagline_text = Text(TAGLINE, justify="center")
    tagline_text.stylize("italic red")
    
    # Version with subtle color
    version_text = Text(f"Version {VERSION} | Production Release", style="dim red", justify="center")
    
    content = Text()
    content.append_text(gradient_logo)
    content.append_text(tagline_text)
    content.append("\n")
    content.append_text(version_text)
    
    return Panel(
        content,
        box=box.DOUBLE_EDGE,
        style="red",
        border_style="bold #800000",
        padding=(1, 2)
    )




def create_about_panel():
    """Detailed information about ANTARA"""
    if not HAS_RICH:
        return None
    
    about_text = """[bold red]What is ANTARA?[/bold red]

ANTARA is a next-generation adaptive meta-learning framework that transforms 
standard neural networks into [bold]self-learning, self-aware systems[/bold] with 
continuous memory and emotional intelligence.

[bold #FF4444]Key Innovation:[/bold #FF4444]
• [red]Prevents Catastrophic Forgetting[/red] - Learns new tasks without losing old knowledge
• [red]Adaptive Learning Rates[/red] - Adjusts learning intensity based on confidence
• [red]Recursive State Monitoring[/red] - Internal states guide learning decisions
• [red]Predictive World Model[/red] - Anticipates future states (I-JEPA architecture)

[bold #FF4444]Research Foundation:[/bold #FF4444]
Built on cutting-edge research combining Elastic Weight Consolidation (EWC),
Synaptic Intelligence (SI), Reptile meta-learning, and hierarchical sparse
mixture-of-experts routing for efficient continual learning.

[dim]Published: 2026 | MIT License | Production-Ready[/dim]
"""
    
    return Panel(
        about_text,
        title="[bold]📖 About ANTARA Framework",
        border_style="#800000",
        box=box.ROUNDED,
        padding=(1, 2)
    )

def create_system_table():
    """Comprehensive system information"""
    if not HAS_RICH:
        return None
    
    sys_info = get_system_info()
    
    table = Table(
        title="🖥️  System Telemetry",
        box=box.HEAVY_EDGE,
        show_header=True,
        header_style="bold white",
        border_style="#800000",
        title_style="bold red"
    )
    
    table.add_column("Component", style="bold red", width=20)
    table.add_column("Specification", style="white", no_wrap=False)
    
    for key, value in sys_info.items():
        if "❌" in value or "⚠️" in value:
            table.add_row(key, f"[bold #FF4444]{value}[/bold #FF4444]")
        elif "CUDA" in value or "GPU" in key:
            table.add_row(key, f"[bold red]{value}[/bold red]")
        else:
            table.add_row(key, value)
    
    return table

def create_module_table():
    """Detailed module status with descriptions"""
    if not HAS_RICH:
        return None
    
    modules = check_modules()
    
    table = Table(
        title="📦 Core Module Status",
        box=box.HEAVY_EDGE,
        show_header=True,
        header_style="bold white",
        border_style="#800000",
        title_style="bold red"
    )
    
    table.add_column("Module Name", style="bold red", width=20)
    table.add_column("Status", style="red", width=15)
    table.add_column("Description", style="dim white", no_wrap=False)
    
    descriptions = {
        "Core Framework": "Main adaptive learning orchestrator",
        "Memory System": "EWC + SI hybrid for continual learning",
        "Consciousness": "Emotional states & self-awareness",
        "Meta Controller": "Reptile meta-learning engine",
        "World Model": "I-JEPA predictive foresight",
        "Health Monitor": "Autonomic self-repair daemon",
        "Adapters": "Task-specific FiLM modulation",
        "MoE Router": "Hierarchical sparse experts",
    }
    
    online_count = sum(1 for status, _ in modules.values() if status)
    total_count = len(modules)
    
    for name, (status, msg) in modules.items():
        if status:
            table.add_row(name, f"[bold green]{msg}[/bold green]", descriptions.get(name, ""))
        else:
            table.add_row(name, f"[bold red]{msg}[/bold red]", descriptions.get(name, ""))
    
    # Add summary row
    table.add_row(
        "[bold]SUMMARY",
        f"[bold yellow]{online_count}/{total_count} Operational",
        "" if online_count == total_count else "[yellow]Some modules unavailable - check dependencies[/yellow]"
    )
    
    return table

def create_features_panel():
    """Enhanced features showcase"""
    if not HAS_RICH:
        return None
    
    features = """[bold red]Framework Capabilities[/bold red]

[bold #FF4444]🧠 Cognitive Systems:[/bold #FF4444]
  [red]✓[/red] Meta-Learning via Reptile Algorithm
  [red]✓[/red] Recursive State Monitoring
  [red]✓[/red] World Model for Predictive Foresight (I-JEPA)
  [red]✓[/red] Hierarchical Mixture-of-Experts (Sparse Routing)

[bold #FF4444]💾 Memory Architecture:[/bold #FF4444]
  [red]✓[/red] Elastic Weight Consolidation (EWC)
  [red]✓[/red] Synaptic Intelligence (SI)
  [red]✓[/red] Relational Graph Memory (Semantic Clustering)
  [red]✓[/red] Prioritized Experience Replay (Generative Replay)

[bold #FF4444]⚡ Adaptive Systems:[/bold #FF4444]
  [red]✓[/red] Dynamic Learning Rate Modulation
  [red]✓[/red] Autonomic Health Monitoring & Repair
  [red]✓[/red] Surprise-Driven Consolidation
  [red]✓[/red] Multi-Task Interference Prevention

[bold #FF4444]🎯 Production Ready:[/bold #FF4444]
  [red]✓[/red] 10+ Pre-configured Presets
  [red]✓[/red] Full PyTorch Integration
  [red]✓[/red] GPU/CPU Auto-Detection
  [red]✓[/red] Comprehensive Documentation
"""
    
    return Panel(
        features,
        title="[bold]✨ Core Features",
        border_style="#800000",
        box=box.ROUNDED,
        padding=(1, 2)
    )

def create_quickstart():
    """Quick start code example"""
    if not HAS_RICH:
        return None
    
    code = '''from airborne_antara import AdaptiveFramework, AdaptiveFrameworkConfig
import torch.nn as nn

# 1. Define your base model
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

# 2. Initialize ANTARA with production config
config = AdaptiveFrameworkConfig.production()
agent = AdaptiveFramework(model, config)

# 3. Train with adaptive learning
for x, y in dataloader:
    metrics = agent.train_step(x, target_data=y)
    
    # Monitor cognitive state
    print(f"Loss: {metrics['loss']:.4f}")
    print(f"Mode: {metrics['mode']}")  # NORMAL/NOVELTY/PANIC
    print(f"Emotion: {metrics.get('emotion', 'N/A')}")
'''
    
    syntax = Syntax(code, "python", theme="monokai", line_numbers=True)
    
    return Panel(
        syntax,
        title="[bold]🚀 Quick Start Example",
        border_style="#800000",
        box=box.ROUNDED,
        padding=(1, 1)
    )

def display_info():
    """Main information display with animations"""
    if not HAS_RICH:
        # Plain text fallback
        console.clear()
        print("=" * 70)
        print(LOGO)
        print(TAGLINE.center(70))
        print(f"Version {VERSION}".center(70))
        print("=" * 70)
        print("\nSYSTEM INFORMATION:")
        for key, value in get_system_info().items():
            print(f"  {key:20s}: {value}")
        print("\nCORE MODULES:")
        for name, (status, msg) in check_modules().items():
            print(f"  {name:20s}: {msg}")
        print("\n" + "=" * 70)
        print("Install 'rich' for enhanced display: pip install rich")
        print("=" * 70)
        return
    
    # Animated logo reveal
    animate_logo_reveal()
    
    console.print()  # Spacing
    
    # Sequential panel reveals with delays
    panels = [
        (create_about_panel(), 0.15),
        (create_system_table(), 0.15),
        (create_module_table(), 0.15),
        (create_features_panel(), 0.15),
        (create_quickstart(), 0.15),
    ]
    
    for panel, delay in panels:
        console.print(panel)
        console.print()
        time.sleep(delay)
    
    # Footer with pulsing effect (simulated with colors)
    footer = Panel(
        f"[bold red]ANTARA Framework v{VERSION}[/bold red] | "
        f"[dim]© 2026 Suryaansh Prithvijit Singh[/dim]\n"
        f"[#FF4444]Documentation:[/#FF4444] See [red]docs/[/red] directory | "
        f"[#FF4444]License:[/#FF4444] MIT",
        border_style="bold #800000",
        box=box.DOUBLE_EDGE
    )
    console.print(footer)

def main():
    """Main entry point"""
    try:
        animate_startup()
        display_info()
    except KeyboardInterrupt:
        if HAS_RICH:
            console.print("\n[bold yellow]⚠️  Interrupted by user[/bold yellow]\n")
        else:
            print("\n\nInterrupted by user\n")
    except Exception as e:
        if HAS_RICH:
            console.print(f"\n[bold red]❌ Error:[/bold red] {e}\n")
        else:
            print(f"\nError: {e}\n")

if __name__ == "__main__":
    main()