import main as m

p = m.read("/Users/koviressler/Daily-Tefillin_2025/tefillin_detection/finallyTef/train/images/17dae0c8-139a-4c0d-aa5e-c65f815e374c 2.JPG")
z = m.read("/Users/koviressler/Desktop/DailyTefillin/people/zacky.JPG")

#kovi testing
blank = m.read("/Users/koviressler/Daily-Tefillin_2025/people/kovi_blank.JPG")
blank2 = m.read("/Users/koviressler/Daily-Tefillin_2025/people/Screenshot 2025-10-01 at 3.50.12 PM.png")


left = m.read("/Users/koviressler/Daily-Tefillin_2025/people/kovi_left.JPG")
right = m.read("/Users/koviressler/Daily-Tefillin_2025/people/kovi_right.JPG")
low = m.read("/Users/koviressler/Daily-Tefillin_2025/people/kovi_low.JPG")
good = m.read("/Users/koviressler/Daily-Tefillin_2025/people/kovi_good.JPG")


m.initialize(z)
print(m.tef_good(z, debug=True))

m.initialize(blank)

print(m.tef_good(left, debug=True))
print(m.tef_good(right, debug=True))
print(m.tef_good(low, debug=True))


m.initialize(blank2, manual_hairline=True, manual_value=1550)

print(m.tef_good(good, debug=True))