import cv2
import numpy as np

# Canvas and app states
canvas = np.ones((500, 600, 3), np.uint8) * 255
drawing = False
tool = "Brush"
start_x, start_y = -1, -1
brush_size = 5
r_val, g_val, b_val = 0, 0, 0
in_edit_mode = False
img_loaded = False
edit_image = None
preview_canvas = canvas.copy()

# Dummy callback
def nothing(x): pass

# Mouse callback
def draw(event, x, y, flags, param):
    global drawing, start_x, start_y, canvas, preview_canvas

    if x > 500 or in_edit_mode:
        return
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = x, y
        if tool in ["Brush", "Eraser"]:
            cv2.circle(canvas, (x, y), brush_size, get_color(), -1)
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        if tool in ["Brush", "Eraser"]:
            cv2.line(canvas, (start_x, start_y), (x, y), get_color(), brush_size)
            start_x, start_y = x, y
        else:
            preview_canvas = canvas.copy()
            draw_shape(preview_canvas, start_x, start_y, x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if tool in ["Line", "Rectangle", "Circle"]:
            draw_shape(canvas, start_x, start_y, x, y)

# Get current color
def get_color():
    return (b_val, g_val, r_val) if tool != "Eraser" else (255, 255, 255)

# Draw shapes
def draw_shape(img, x1, y1, x2, y2):
    if tool == "Line":
        cv2.line(img, (x1, y1), (x2, y2), get_color(), brush_size)
    elif tool == "Rectangle":
        cv2.rectangle(img, (x1, y1), (x2, y2), get_color(), brush_size)
    elif tool == "Circle":
        radius = int(((x2 - x1)**2 + (y2 - y1)**2)**0.5)
        cv2.circle(img, (x1, y1), radius, get_color(), brush_size)

# Right panel info
def draw_info_panel():
    canvas[:, 500:] = (240, 240, 240)
    if not in_edit_mode:
        cv2.putText(canvas, "Press 'x' to edit image", (505, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        tools = ["b: Brush", "e: Eraser", "l: Line", "r: Rect", "c: Circle/Clear", "s: Save", "+/-: Size", "q: Quit"]
        for i, t in enumerate(tools):
            cv2.putText(canvas, t, (505, 90 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(canvas, f"Tool: {tool}", (505, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 200), 1)
        cv2.putText(canvas, f"Size: {brush_size}", (505, 345), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(canvas, "Color:", (505, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.rectangle(canvas, (505, 390), (595, 410), get_color(), -1)
    else:
        cv2.putText(canvas, "Edit Mode: HSV Extraction", (505, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(canvas, "Use sliders to mask", (505, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)
        cv2.putText(canvas, "Press Enter to apply", (505, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 0), 1)

# Create UI
cv2.namedWindow("Paint App")
cv2.setMouseCallback("Paint App", draw)

# Color sliders
cv2.createTrackbar("R", "Paint App", 0, 255, nothing)
cv2.createTrackbar("G", "Paint App", 0, 255, nothing)
cv2.createTrackbar("B", "Paint App", 0, 255, nothing)

# HSV sliders for image editing
cv2.createTrackbar("LH", "Paint App", 0, 179, nothing)
cv2.createTrackbar("LS", "Paint App", 0, 255, nothing)
cv2.createTrackbar("LV", "Paint App", 0, 255, nothing)
cv2.createTrackbar("UH", "Paint App", 179, 179, nothing)
cv2.createTrackbar("US", "Paint App", 255, 255, nothing)
cv2.createTrackbar("UV", "Paint App", 255, 255, nothing)

print("🎨 Ready: Paint and Extract Image by pressing 'x'.")

# Main loop
while True:
    preview_canvas = canvas.copy()

    # RGB Sliders
    r_val = cv2.getTrackbarPos("R", "Paint App")
    g_val = cv2.getTrackbarPos("G", "Paint App")
    b_val = cv2.getTrackbarPos("B", "Paint App")

    # HSV Sliders
    lh = cv2.getTrackbarPos("LH", "Paint App")
    ls = cv2.getTrackbarPos("LS", "Paint App")
    lv = cv2.getTrackbarPos("LV", "Paint App")
    uh = cv2.getTrackbarPos("UH", "Paint App")
    us = cv2.getTrackbarPos("US", "Paint App")
    uv = cv2.getTrackbarPos("UV", "Paint App")

    # Edit Mode Preview
    if in_edit_mode and img_loaded:
        hsv = cv2.cvtColor(edit_image, cv2.COLOR_BGR2HSV)
        lower = np.array([lh, ls, lv])
        upper = np.array([uh, us, uv])
        mask = cv2.inRange(hsv, lower, upper)
        result = cv2.bitwise_and(edit_image, edit_image, mask=mask)
        display = np.ones((500, 600, 3), np.uint8) * 255
        display[:, :500] = result
        draw_info_panel()
        cv2.imshow("Paint App", display)
    else:
        draw_info_panel()
        display = preview_canvas if drawing and tool in ["Line", "Rectangle", "Circle"] else canvas
        cv2.imshow("Paint App", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('b') and not in_edit_mode:
        tool = "Brush"
    elif key == ord('e') and not in_edit_mode:
        tool = "Eraser"
    elif key == ord('l') and not in_edit_mode:
        tool = "Line"
    elif key == ord('r') and not in_edit_mode:
        tool = "Rectangle"
    elif key == ord('c') and not in_edit_mode:
        tool = "Circle"
        canvas[:, :500] = 255
    elif key == ord('s') and not in_edit_mode:
        cv2.imwrite("paint_output.png", canvas[:, :500])
        print("✅ Saved as 'paint_output.png'")
    elif key == ord('+') or key == ord('='):
        brush_size = min(50, brush_size + 1)
    elif key == ord('-'):
        brush_size = max(1, brush_size - 1)
    elif key == ord('x') and not in_edit_mode:
        edit_image = cv2.imread("hi.png")
        if edit_image is not None:
            edit_image = cv2.resize(edit_image, (500, 500))
            in_edit_mode = True
            img_loaded = True
            print("🧠 Edit mode activated. Adjust HSV then press Enter.")
        else:
            print("❌ Could not load 'hi.png'")
    elif key == 13:  # Enter
        if in_edit_mode and img_loaded:
            canvas[:, :500] = display[:, :500]
            in_edit_mode = False
            tool = "Brush"
            print("✅ Object extracted and loaded to canvas.")

cv2.destroyAllWindows()
