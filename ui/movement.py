from pynput import keyboard, mouse

mouse_ctrl = mouse.Controller()
STEP = 40

def on_press(key):
    # We use 'global' to allow the escape key to stop the listener
    try:
        if key == keyboard.Key.up:
            mouse_ctrl.move(0, -STEP)
        elif key == keyboard.Key.down:
            mouse_ctrl.move(0, STEP)
        elif key == keyboard.Key.left:
            mouse_ctrl.move(-STEP, 0)
        elif key == keyboard.Key.right:
            mouse_ctrl.move(STEP, 0)
        
        # Shift key now acts as a 'Cancel/Clear' or simply does nothing
        # while being suppressed, preventing focus shifts.
        elif key == keyboard.Key.shift or key == keyboard.Key.shift_r:
            print("Shift pressed: Ignoring input to maintain focus.")
            
        elif key == keyboard.Key.enter:
            mouse_ctrl.click(mouse.Button.left, 1)
            print("Mouse Clicked (Native Enter suppressed)")

        # Emergency Exit
        if key == keyboard.Key.esc:
            print("Restoring keyboard control and exiting...")
            return False
            
    except Exception as e:
        print(f"Error: {e}")

print("--- STEALTH MODE ACTIVE ---")
print("Arrow Keys: Move | Enter: Click Only | Shift: Prevent Focus Change | Esc: STOP")

# suppress=True prevents the keys from being sent to the active application
with keyboard.Listener(on_press=on_press, suppress=True) as listener:
    listener.join()