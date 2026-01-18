from PIL import Image, ImageChops

def trim_whitespace(image_path, output_path, padding=20):
    im = Image.open(image_path)
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    
    if bbox:
        # Add padding
        left, upper, right, lower = bbox
        width, height = im.size
        left = max(0, left - padding)
        upper = max(0, upper - padding)
        right = min(width, right + padding)
        lower = min(height, lower + padding)
        
        im = im.crop((left, upper, right, lower))
        im.save(output_path)
        print(f"Cropped {image_path} to {output_path}")
    else:
        print(f"No content found in {image_path}")

# Paths (using the paths known from previous steps)
src = r"C:\Users\carlos\.gemini\antigravity\brain\e47dd69b-7f79-41b9-bd3f-37ecfedfff18\bayes_spam_filter_visual_v2_1768683500358.png"
dst1 = r"d:\OneDrive - Vaal University of Technology\WORK\2026\AI_v2\images\week_2_bayes_v2.png"
dst2 = r"d:\OneDrive - Vaal University of Technology\WORK\2026\AI_v2_html\images\week_2_bayes_v2.png"

# Crop original and save to destinations
trim_whitespace(src, dst1)
trim_whitespace(src, dst2)
