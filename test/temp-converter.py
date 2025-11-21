def convert_temp():
    user_input = input("Enter Temp (e.g, 25C or 77F): ").strip()
    temp_unit = user_input[-1]
    temp = float(user_input[:-1])

    if temp_unit == 'C':
        f = (temp * 9/5) + 32
        print(f"{temp:.2f}°C is {f:.2f}°F")
    elif temp_unit == 'F':
        c = (temp - 32) * 5/9
        print(f"{temp:.2f}°F is {c:.2f}°C")

convert_temp()