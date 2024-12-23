from utils import * 

def get_expected(i):
    if (i >= 10 and i <= 18) or i == 24:
        expected_digits = "4811 3672 1092 6684"
        expected_price = "10.00EGP"
    elif i >= 19 and i <= 23:
        expected_digits = "1130 7763 3550 8717"
        expected_price = "10.00EGP"
    elif i >= 25 and i <= 26:
        expected_digits = "0086 7539 5367 5630"
        expected_price = "16.50EGP"
    elif i >= 27 and i <= 30:
        expected_digits = "7667 6413 2016 9051"
        expected_price = "16.50EGP"
    elif i >= 31 and i <= 32:
        expected_digits = "3784 3713 7877 2058"
        expected_price = "16.50EGP"
    elif i >= 33 and i <= 34:
        expected_digits = "3784 3713 7877 2058"
        expected_price = "16.50EGP"
    elif i == 35:
        expected_digits = "9080 4158 1356 6057"
        expected_price = "25.00EGP"
    return expected_digits, expected_price

base_path = "imgs/"
min_image_index = 10
max_image_index = 35
score = 0
for i in range(min_image_index, max_image_index + 1):
    if i == 15 or i == 16 or i == 17 or i == 18 or i == 24:
        print(f"Skipping Image {i}. Receipt is rotated with an angle of >= 90 degrees.")
        continue
    try:

        print(f"Running Image {i}")
        image_path = base_path + str(i) + ".jpg"
        if not os.path.exists(image_path):
            image_path = base_path + str(i) + ".jpeg"
        image = load_image(image_path)

        os.makedirs("results", exist_ok=True)
        reciept, reciept_gray= find_reciept_kmeans(image)
        digits = find_digits_basel(reciept_gray)
        price = get_price(reciept_gray)

        expected_digits, expected_price = get_expected(i)
        print(f"Expected Digits: {expected_digits}, Digits: {digits}")
        print(f"Expected Price: {expected_price}, Price: {price}")
        if digits == expected_digits: # and price == expected_price:
            score += 1
    except Exception as e:
        print(f"Error in Image {i}: {e}")

    print(f"Image {i} done.")
print("All images done.")
print(f"Accuracy: {score}/{max_image_index - min_image_index + 1}")
