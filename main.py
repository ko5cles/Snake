# This is a sample Python script.
from tkinter import *
from tkinter import filedialog
import imageio.v3 as iio
from PIL import ImageTk, Image
import numpy as np
import time
import random

prev_loc = None
canvas = None
points = []
draw_id = None
file_name = None
image_gradient_mag = None
img = None
initial_length = []


def AskForFileName():
    file_name = filedialog.askopenfilename(title="Select an image file:")
    return file_name


def Gaussian_filter1D(size, sigma):
    normalized_range = range(-size // 2 + 1, size // 2 + 1)
    horizontal_filter = np.array(
        [[1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-x ** 2 / (2 * sigma ** 2)) for x in normalized_range]])
    return horizontal_filter


def convolution(f, I):
    # Handle boundary of I, e.g. pad I according to size of f
    im_conv = np.zeros(I.shape)
    width = I.shape[1]
    height = I.shape[0]

    # Compute im_conv = f*I
    if f.shape[0] == 1:  # horizontal 1D
        f_len = f.shape[1]
        f_half_len = f_len // 2
        normalized_range = range(-f_half_len, f_half_len + 1)
        for i in range(height):
            for j in range(width):
                convolution_sum = 0
                for k in normalized_range:
                    if j + k < 0 or j + k > width - 1:
                        continue
                    else:
                        convolution_sum += I[i][j + k] * f[0][k + f_half_len]
                im_conv[i][j] = convolution_sum

    elif f.shape[1] == 1:  # vertical 1D
        f_len = f.shape[0]
        f_half_len = f_len // 2
        normalized_range = range(-f_half_len, f_half_len + 1)
        for j in range(width):
            for i in range(height):
                convolution_sum = 0
                for k in normalized_range:
                    if i + k < 0 or i + k > height - 1:
                        continue
                    else:
                        convolution_sum += I[i + k][j] * f[k + f_half_len][0]
                im_conv[i][j] = convolution_sum

    return im_conv


def CalculateGradientMagnitude(img):
    # RGB image
    if len(img.shape) == 3:
        # Fake gray scale
        if (img[:, :, 0] == img[:, :, 1]).all():
            img = img[:, :, 0]
        # Convert RGB to gray scale
        else:
            img = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    global image_gradient_mag
    # Gaussian filter
    gaussian_filter = Gaussian_filter1D(13, 1)
    image_filtered = convolution(gaussian_filter, img)
    image_filtered = convolution(gaussian_filter.reshape((-1, 1)), image_filtered)
    # derivative filter
    derivative_filter = np.array([[-1, 0, 1]])
    image_horizontal_filtered = convolution(derivative_filter, image_filtered)
    image_vertical_filtered = convolution(derivative_filter.reshape((-1, 1)), image_filtered)
    # calculate magnitude
    image_gradient_mag = 1 / np.sqrt(2) * np.sqrt(image_horizontal_filtered ** 2 + image_vertical_filtered ** 2)
    # remove border value, this is due to padding with 0
    image_gradient_mag[0, :] = 0
    image_gradient_mag[-1, :] = 0
    image_gradient_mag[:, 0] = 0
    image_gradient_mag[:, -1] = 0
    # save result
    iio.imwrite("image_gradient_mag.png", image_gradient_mag.astype(int).astype(np.uint8))
    image_gradient_mag = (image_gradient_mag - np.min(image_gradient_mag)) / (
            np.max(image_gradient_mag) - np.min(image_gradient_mag))
    return


def Preprocess():
    global canvas
    global prompt2
    global button2
    global draw_id
    global file_name

    canvas.create_line((points[0], points[-1]), fill="red")

    # calculate gradient magnitude
    img_2_process = iio.imread(file_name)
    CalculateGradientMagnitude(img_2_process)

    canvas.unbind("<Button 1>", draw_id)
    prompt2.config(text="Let's show the process:")
    button2.config(text="Ok", command=Snake)


def CalculateL2(t1, t2):
    return np.sqrt((t1[0] - t2[0]) ** 2 + (t1[1] - t2[1]) ** 2)


def CalculateCurv(t1, t2, t3):
    l1 = (t2[0] - t1[0], t2[1] - t1[1])
    l2 = (t3[0] - t2[0], t3[1] - t2[1])
    cos = (l1[0] * l2[0] + l1[1] * l2[1]) / np.sqrt(l1[0] ** 2 + l1[1] ** 2) / np.sqrt(l2[0] ** 2 + l2[1] ** 2)
    return 1 - cos


def RecordLength():
    global initial_length
    for i in range(len(points)):
        if i == len(points) - 1:
            distance = CalculateL2(points[i], points[0])
            initial_length.append(distance)
        else:
            distance = CalculateL2(points[i], points[i + 1])
            initial_length.append(distance)
    return


def CalculateEnergy(prev, cur, next, k, average_proportion):
    global image_gradient_mag
    global initial_length
    global points
    cont = abs(1 - CalculateL2(cur, next) / initial_length[k] / average_proportion)
    curv = CalculateCurv(prev, cur, next)
    image = image_gradient_mag[cur[1]][cur[0]]
    return (cont, curv, image)


def CalculateAvergeProportion():
    global points
    global initial_length
    sum = 0
    for i in range(len(points)):
        if i == len(points) - 1:
            proportion = CalculateL2(points[i], points[0]) / initial_length[i]
            sum += proportion
        else:
            proportion = CalculateL2(points[i], points[i + 1]) / initial_length[i]
            sum += proportion
    return sum / len(points)


def Redraw():
    global canvas
    global points
    global file_name
    global img

    canvas.delete("all")
    canvas.create_image((0, 0), image=img, anchor=NW)

    for i in range(len(points)):
        x = points[i][0]
        y = points[i][1]
        canvas.create_oval((x - 5, y - 5, x + 5, y + 5), fill="red", outline="red")
        if i == len(points) - 1:
            x1 = points[0][0]
            y1 = points[0][1]
        else:
            x1 = points[i + 1][0]
            y1 = points[i + 1][1]
        canvas.create_line((x, y, x1, y1), fill="red")

    canvas.update()
    return


def Snake():
    global prompt2
    global button2
    global image_gradient_mag

    global_counter = 1

    button2.pack_forget()

    alpha = 1
    beta = 1
    gamma = 3
    window_size_base = 3
    window_size = [window_size_base] * len(points)
    shape = image_gradient_mag.shape

    move_count = 0
    percent = 1

    RecordLength()
    average_proportion = 1
    max_cont = 0
    target_location = None
    temp_result = {}
    range_list = list(range(len(points)))

    while (percent > 0.15):
        prompt2.config(text="Step " + str(global_counter) + ":")
        random.shuffle(range_list)
        for k in range_list:
            for i in range(max(0, points[k][0] - window_size[k]), min(shape[1] - 1, points[k][0] + window_size[k] + 1)):
                for j in range(max(0, points[k][1] - window_size[k]),
                               min(shape[0] - 1, points[k][1] + window_size[k] + 1)):
                    cont, curv, image = CalculateEnergy(points[(k - 1) % len(points)], (i, j),
                                                        points[(k + 1) % len(points)], k, average_proportion)
                    if cont > max_cont:
                        max_cont = cont
                    temp_result[(i, j)] = (cont, curv, image)
            local_energy = temp_result[points[k]]
            min_energy = alpha * local_energy[0] / max_cont + beta * local_energy[1] - gamma * local_energy[2]
            for key in temp_result.keys():
                value = temp_result[key]
                energy = alpha * value[0] / max_cont + beta * value[1] - gamma * value[2]
                if energy < min_energy:
                    min_energy = energy
                    target_location = key
            if target_location and points[k] != target_location:
                move_count += 1
                points[k] = target_location

            # Clear variables
            max_cont = 0
            target_location = None
            temp_result.clear()

        Redraw()
        percent = move_count / len(points)
        global_counter += 1

        for i in range(len(points)):
            temp_gradient_mag = image_gradient_mag[points[i][1]][points[i][0]]
            window_size[i] = window_size_base * int(-1.5 * temp_gradient_mag ** 2 + 2)

        # update average proportion
        average_proportion = CalculateAvergeProportion()

        # Clear variables
        move_count = 0
        time.sleep(0.5)


def DrawPoint(event):
    global canvas
    global prev_loc
    global points
    points.append((event.x, event.y))
    canvas.create_oval((event.x - 5, event.y - 5, event.x + 5, event.y + 5), fill="red", outline="red")
    if prev_loc != None:
        canvas.create_line((prev_loc, (event.x, event.y)), fill="red")
    prev_loc = (event.x, event.y)


def OpenImage():
    global prompt2
    global button2
    global draw_id
    global file_name
    global img

    file_name = AskForFileName()

    prompt2 = Label(root, text="Draw an approximate outline:")
    prompt2.pack()

    img = ImageTk.PhotoImage(Image.open(file_name))
    img_2_process = iio.imread(file_name)

    global canvas
    canvas = Canvas(width=img_2_process.shape[1], height=img_2_process.shape[0])
    canvas.create_image((0, 0), image=img, anchor=NW)
    draw_id = canvas.bind("<Button-1>", DrawPoint)
    canvas.pack()

    prompt1.pack_forget()
    button1.pack_forget()

    button2 = Button(text="Finish", command=Preprocess)
    button2.pack()


root = Tk()
root.title("Snake")

# ask for an image file
prompt1 = Label(root, text="Select an image file:")
prompt1.pack()
button1 = Button(text="Open", command=OpenImage)
button1.pack()
prompt2 = None
button2 = None

root.mainloop()
