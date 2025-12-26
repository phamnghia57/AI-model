import numpy as np

print("Hello, World!")

a = np.array([1, 2, 3, 4, 5])
b = np.array([10, 20, 30, 40, 50])

tong = a + b
hieu = b - a
tich = a * b
thuong = b / a

print("Tổng:", tong)
print("Hiệu:", hieu)
print("Tích:", tich)
print("Thương:", thuong)

print("Giá trị trung bình của a:", np.mean(a))
print("Giá trị lớn nhất của b:", np.max(b))


print("Goodbye!")