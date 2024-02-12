from math import log
onegin = open("onegin.txt", "r", encoding="utf8")
text = onegin.read()
alphabet_up = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧЪШЬЩЭЮЯ"
alphabet_down = "абвгдеёжзийклмнопрстуфхцчъшьэюя"
freq = {}
amount = 0


for ch in text:
    if ch in alphabet_down:
        ch = alphabet_up[alphabet_down.index(ch)]

    if ch in alphabet_up:
        if freq.get(ch) is None:
            freq[ch] = 1
            amount += 1
        else:
            freq[ch] += 1
            amount += 1
entropy = 0
for letter, frequency in freq.items():
    p = frequency/amount
    print(f"{letter}: {p}")
    entropy += p*log(1/p, 2)
print(f"entropy = {entropy}")
print(f"amount of information = {entropy*amount}")

from PIL import Image
image = Image.open("onegin.png")
width, heigth = image.size
pixels = list(image.getdata())
amount_of_pixels = 0
pixels_freq = {}
for pixel in pixels:
    if pixels_freq.get(pixel[0]) is None:
        pixels_freq[pixel[0]] = 1
    else:
        pixels_freq[pixel[0]] += 1
    amount_of_pixels += 1

print(pixels_freq[109])
image_entropy = 0
for brightness, frequency in pixels_freq.items():
    p = frequency/amount_of_pixels
    image_entropy += p*log(1/p)
print(f"amount of information of image = {image_entropy*amount_of_pixels}")
print(f"image entropy = {image_entropy}")
