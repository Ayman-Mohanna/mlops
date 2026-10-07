from pathlib import Path
from PIL import Image
import shutil


# Paths
RAW_DIR = Path("data/food11_raw")
PROCESSED_DIR = Path("data/food11_processed")
MINI_DIR = Path("data/food11_processed_mini")

# Food-11 categories
CATEGORIES = {
    "0": "Bread",
    "1": "Dairy product",
    "2": "Dessert",
    "3": "Egg",
    "4": "Fried food",
    "5": "Meat",
    "6": "Noodles-Pasta",
    "7": "Rice",
    "8": "Seafood",
    "9": "Soup",
    "10": "Vegetable-Fruit",
}

SPLITS = ["training", "evaluation", "validation"]
IMAGE_SIZE = (128, 128)
MINI_LIMIT = 100


def prepare_data():
    # Remove old processed folders if they already exist
    if PROCESSED_DIR.exists():
        shutil.rmtree(PROCESSED_DIR)

    if MINI_DIR.exists():
        shutil.rmtree(MINI_DIR)

    for split in SPLITS:
        source_folder = RAW_DIR / split

        # Keep track of mini images for each category
        mini_counts = {category: 0 for category in CATEGORIES.values()}

        for image_path in source_folder.iterdir():
            if not image_path.is_file():
                continue

            # Category number is the first part of the filename
            category_number = image_path.stem.split("_")[0]

            if category_number not in CATEGORIES:
                continue

            category_name = CATEGORIES[category_number]

            processed_category = PROCESSED_DIR / split / category_name
            mini_category = MINI_DIR / split / category_name

            processed_category.mkdir(parents=True, exist_ok=True)
            mini_category.mkdir(parents=True, exist_ok=True)

            try:
                # Open and resize image
                with Image.open(image_path) as image:
                    image = image.convert("RGB")
                    image = image.resize(IMAGE_SIZE)

                    output_path = processed_category / image_path.name
                    image.save(output_path)

                    # Add maximum 100 images per category to mini dataset
                    if mini_counts[category_name] < MINI_LIMIT:
                        mini_output = mini_category / image_path.name
                        image.save(mini_output)
                        mini_counts[category_name] += 1

            except Exception as error:
                print(f"Could not process {image_path}: {error}")

    print("Data preparation finished.")


if __name__ == "__main__":
    prepare_data()