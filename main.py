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
img=None

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
    #iio.imwrite("image_gradient_mag.png", image_gradient_mag.astype(int).astype(np.uint8))
    image_gradient_mag=(image_gradient_mag-np.min(image_gradient_mag))/(np.max(image_gradient_mag)-np.min(image_gradient_mag))
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

def CalculateL2(t1,t2):
    return np.sqrt((t1[0]-t2[0])**2+(t1[1]-t2[1])**2)

def CalculateCurv(t1,t2,t3):
    return np.sqrt((t1[0]+t3[0]-2*t2[0])**2+(t1[1]+t3[1]-2*t2[1])**2)

def CalculateDAverage():
    sum = 0
    for i in range(len(points)):
        if i == len(points) - 1:
            sum += CalculateL2(points[i],points[0])
        else:
            sum += CalculateL2(points[i],points[i+1])
    return sum/len(points)

def ConvertAxis(cur):
    global image_gradient_mag
    return (cur[0],image_gradient_mag.shape[1]-1-cur[1])

def CalculateEnergy(prev,cur,next,averge_d):
    global image_gradient_mag
    cont=(averge_d-CalculateL2(prev,cur))**2
    curv=CalculateCurv(prev,cur,next)
    axis=ConvertAxis(cur)
    image=image_gradient_mag[axis[0]][axis[1]]
    return (cont,curv,image)

def Redraw():
    global canvas
    global points
    global file_name
    global img

    canvas.delete("all")
    canvas.create_image((0, 0), image=img, anchor=NW)

    for i in range(len(points)):
        x=points[i][0]
        y=points[i][1]
        canvas.create_oval((x - 5, y - 5, x + 5, y + 5), fill="red", outline="red")
        if i==len(points)-1:
            x1=points[0][0]
            y1=points[0][1]
        else:
            x1=points[i+1][0]
            y1=points[i+1][1]
        canvas.create_line((x,y,x1,y1),fill="red")

    canvas.update()
    return

def Snake():
    global prompt2
    global button2
    global image_gradient_mag

    global_counter=1

    button2.pack_forget()

    alpha = 1
    beta = 1
    gamma = 2
    window_size_base=2
    window_size=[window_size_base]*len(points)
    curv_list=[0]*len(points)
    shape=image_gradient_mag.shape

    move_count=0
    percent = 1

    d_bar = CalculateDAverage()
    max_cont=0
    max_curv=0
    target_location=None
    temp_result={}
    range_list=list(range(len(points)))

    while (percent > 0.1):
        prompt2.config(text="Step "+str(global_counter)+":")
        random.shuffle(range_list)
        for k in range_list:
            for i in range(max(0,points[k][0]-window_size[k]),min(shape[0]-1,points[k][0]+window_size[k]+1)):
                for j in range(max(0,points[k][1]-window_size[k]),min(shape[1]-1,points[k][1]+window_size[k]+1)):
                    cont,curv,image=CalculateEnergy(points[(k-1)%len(points)],(i,j),points[(k+1)%len(points)],d_bar)
                    if cont>max_cont:
                        max_cont=cont
                    if curv>max_curv:
                        max_curv=curv
                    temp_result[(i,j)]=(cont,curv,image)
            local_energy=temp_result[points[k]]
            min_energy=alpha*local_energy[0]/max_cont+beta*local_energy[1]/max_curv-gamma*local_energy[2]
            for key in temp_result.keys():
                value=temp_result[key]
                energy=alpha*value[0]/max_cont+beta*value[1]/max_curv-gamma*value[2]
                if energy<min_energy:
                    min_energy=energy
                    target_location=key
            if target_location and points[k]!=target_location:
                move_count+=1
                points[k]=target_location

            # Clear variables
            max_cont = 0
            max_curv = 0
            target_location = None
            temp_result.clear()

        Redraw()
        percent=move_count/len(points)
        global_counter+=1
        # Calculate curv and update window size
        for i in range(len(points)):
            curv_list[i] = CalculateCurv(points[(i - 1) % len(points)], points[i], points[(i + 1) % len(points)])
        average_curv = sum(curv_list) / len(curv_list)
        for i in range(len(curv_list)):
            point=ConvertAxis(points[i])
            temp_gradient_mag=image_gradient_mag[point[0]][point[1]]
            window_size[i] = window_size_base*int(max(curv_list[i]/average_curv,1)*(-3*temp_gradient_mag**2+4))
        # update d average
        d_bar = CalculateDAverage()
        # Clear variables
        move_count=0
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
    canvas = Canvas(width=img_2_process.shape[0], height=img_2_process.shape[1])
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