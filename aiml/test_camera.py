"""
Quick test script to check if camera is accessible.
"""
import cv2
import sys
import os

# Fix Windows encoding issues
if sys.platform == 'win32':
    os.system('chcp 65001 >nul 2>&1')

print("Testing camera access...")
print("=" * 50)

# Try cameras 0, 1, 2
for i in range(3):
    print(f"\nTrying camera {i}...")
    cap = cv2.VideoCapture(i)
    
    if not cap.isOpened():
        print(f"  [ERROR] Camera {i}: Cannot open")
        continue
    
    # Try to read a frame
    ret, frame = cap.read()
    
    if ret and frame is not None:
        height, width = frame.shape[:2]
        print(f"  [OK] Camera {i}: WORKING!")
        print(f"     Resolution: {width}x{height}")
        print(f"     Frame shape: {frame.shape}")
        
        # Try to set properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"     Set to: {actual_width}x{actual_height}")
        
        cap.release()
        print(f"\n[SUCCESS] Use camera index {i} in your code")
        sys.exit(0)
    else:
        print(f"  [ERROR] Camera {i}: Opened but cannot read frames")
        cap.release()

print("\n" + "=" * 50)
print("[ERROR] NO WORKING CAMERA FOUND")
print("\nTroubleshooting:")
print("1. Check if camera is being used by another app")
print("2. Check Windows Privacy Settings → Camera")
print("3. Restart your computer")
print("4. Try unplugging and reconnecting USB camera")
print("5. Check Device Manager for camera issues")

