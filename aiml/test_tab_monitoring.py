"""
Quick test script to check if tab monitoring works.
"""
import platform
import sys
import os

# Fix Windows encoding issues
if sys.platform == 'win32':
    os.system('chcp 65001 >nul 2>&1')

print("Testing tab monitoring...")
print("=" * 50)

system = platform.system()
print(f"OS: {system}")

if system == "Windows":
    print("\nChecking Windows dependencies...")
    try:
        import win32gui
        print("  [OK] pywin32 installed")
        
        # Test getting window title
        hwnd = win32gui.GetForegroundWindow()
        title = win32gui.GetWindowText(hwnd)
        print(f"  [OK] Can get window title: '{title[:50]}...'")
        print("\n[SUCCESS] Tab monitoring should work!")
        
    except ImportError:
        print("  [ERROR] pywin32 NOT installed")
        print("\n  Install with:")
        print("    pip install pywin32")
        sys.exit(1)
    except Exception as e:
        print(f"  [ERROR] Error: {e}")
        sys.exit(1)

elif system == "Linux":
    print("\nChecking Linux dependencies...")
    import subprocess
    try:
        result = subprocess.run(
            ['xdotool', '--version'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("  [OK] xdotool installed")
            print(f"     Version: {result.stdout.strip()}")
            
            # Test getting window title
            result = subprocess.run(
                ['xdotool', 'getactivewindow', 'getwindowname'],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                title = result.stdout.strip()
                print(f"  [OK] Can get window title: '{title[:50]}...'")
                print("\n[SUCCESS] Tab monitoring should work!")
            else:
                print("  [WARNING] xdotool installed but cannot get window title")
        else:
            print("  [ERROR] xdotool not working properly")
            sys.exit(1)
    except FileNotFoundError:
        print("  [ERROR] xdotool NOT installed")
        print("\n  Install with:")
        print("    sudo apt-get install xdotool")
        sys.exit(1)
    except Exception as e:
        print(f"  [ERROR] Error: {e}")
        sys.exit(1)

elif system == "Darwin":  # Mac
    print("\nChecking Mac dependencies...")
    import subprocess
    try:
        result = subprocess.run(
            ['osascript', '-e', 'tell application "System Events" to get name of first process whose frontmost is true'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            app_name = result.stdout.strip()
            print(f"  [OK] Can get active app: '{app_name}'")
            print("\n[SUCCESS] Tab monitoring should work!")
        else:
            print("  [WARNING] osascript may need permissions")
            print("  Check: System Preferences -> Security & Privacy -> Privacy -> Accessibility")
    except Exception as e:
        print(f"  [ERROR] Error: {e}")

else:
    print(f"\n[WARNING] Unknown OS: {system}")
    print("Tab monitoring may not work")

print("\n" + "=" * 50)

