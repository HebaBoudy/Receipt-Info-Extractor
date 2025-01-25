import os

def rename_images_in_folder(base_folder):
    # Iterate through each subfolder in the base folder
    for subfolder in os.listdir(base_folder):
        subfolder_path = os.path.join(base_folder, subfolder)
        
        # Check if the subfolder is a directory
        if os.path.isdir(subfolder_path):
            # Iterate through each file in the subfolder
            for index, filename in enumerate(os.listdir(subfolder_path)):
                file_path = os.path.join(subfolder_path, filename)
                
                # Check if the file is an image (you can add more extensions if needed)
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                    # Get the file extension
                    file_extension = os.path.splitext(filename)[1]
                    
                    # Create the new filename
                    new_filename = f"{subfolder}_{index}{file_extension}"
                    new_file_path = os.path.join(subfolder_path, new_filename)
                    
                    # Rename the file
                    os.rename(file_path, new_file_path)
                    print(f"Renamed {file_path} to {new_file_path}")

# Define the base folder containing the subfolders
base_folder = 'templates'

# Call the function to rename images
rename_images_in_folder(base_folder)