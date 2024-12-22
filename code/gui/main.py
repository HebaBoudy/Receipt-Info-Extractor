
from utils import * 
uploaded_file = "imgs/28.jpeg"
image = load_image(uploaded_file) 

reciept ,reciept_gray= find_reciept_kmeans(image) 
print("after find reciept kmenas")
digits = find_digits_basel(reciept_gray) 
print("after find digits basel",digits)
price = get_price(reciept_gray)
print("after get price",price)
# 350 , 650
