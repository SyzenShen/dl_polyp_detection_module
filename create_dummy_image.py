from PIL import Image, ImageDraw

def create_dummy_image(filename="test_polyp.jpg"):
    # Create a 640x640 white image
    img = Image.new('RGB', (640, 640), color = 'white')
    draw = ImageDraw.Draw(img)
    
    # Draw a red circle to simulate a "polyp" (though YOLO won't recognize it without proper features)
    # Just to have a valid image file
    draw.ellipse((200, 200, 400, 400), fill = 'red', outline ='red')
    
    img.save(filename)
    print(f"Created {filename}")

if __name__ == "__main__":
    create_dummy_image()
