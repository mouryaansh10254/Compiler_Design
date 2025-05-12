import os
from pathlib import Path

def rename_animal_images(root_folder="animals"):
    """
    Renames all images in animal subfolders to standardized names (cat1.jpg, dog2.png, etc.)
    
    Args:
        root_folder (str): Name of your main animals folder (default: 'animals')
    """
    try:
        # Convert to Path object and get absolute path
        animals_path = Path(root_folder).absolute()
        
        # Verify the animals folder exists
        if not animals_path.exists():
            raise FileNotFoundError(f"Folder '{animals_path}' not found. Please make sure it exists.")
        
        if not animals_path.is_dir():
            raise NotADirectoryError(f"'{animals_path}' is not a directory.")
        
        print(f"Found animals folder at: {animals_path}")

        # Get all animal subfolders (cat, dog, etc.)
        animal_folders = [f for f in animals_path.iterdir() if f.is_dir()]
        
        if not animal_folders:
            print(f"No animal subfolders found in {animals_path}")
            return

        for folder in animal_folders:
            animal_name = folder.name.lower()  # e.g., 'cat'
            print(f"\nProcessing {animal_name} folder...")
            
            # Get all image files (jpg, jpeg, png)
            image_files = [
                f for f in folder.iterdir() 
                if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']
            ]
            
            if not image_files:
                print(f"No images found in {animal_name} folder")
                continue

            # Rename each image with numbering
            for index, img_file in enumerate(image_files, start=1):
                # Get file extension (.jpg, .png etc.)
                ext = img_file.suffix.lower()
                
                # Create new name (cat1.jpg, cat2.jpg, etc.)
                new_name = f"{animal_name}{index}{ext}"
                new_path = folder / new_name
                
                # Handle name collisions if file exists
                counter = index
                while new_path.exists():
                    counter += 1
                    new_name = f"{animal_name}{counter}{ext}"
                    new_path = folder / new_name
                
                # Rename the file
                try:
                    img_file.rename(new_path)
                    print(f"Renamed: {img_file.name} → {new_name}")
                except Exception as e:
                    print(f"Failed to rename {img_file.name}: {e}")

        print("\nAll images renamed successfully!")

    except Exception as e:
        print(f"\nError: {e}")
        print("Please check:")
        print(f"1. The '{root_folder}' folder exists in the same directory as this script")
        print(f"2. It contains subfolders (cat, dog, etc.) with images")

if __name__ == "__main__":
    # Run the renamer (change 'animals' if your folder has different name)
    rename_animal_images("animals")