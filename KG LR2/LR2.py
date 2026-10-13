import numpy as np
from PIL import Image
from math import floor #округление вниз

def baricentric_coord (x, y, x0, y0, x1, y1, x2, y2):
    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda2 = 1.0 - lambda0 - lambda1
    return lambda0, lambda1, lambda2


""" np.cross(a,b) -векторное
np.dot(a,b) -  скалярное
np.linalg.norm() - норма
"""
def normal(x0, y0,z0,x1, y1,z1, x2, y2, z2):
    a = [x1-x2, y1-y2, z1-z2]
    b = [x1-x0, y1-y0, z1-z0]
    return -np.cross(a, b)


def clipping_faces(n):
    a = [x1-x2, y1-y2, z1-z2]
    b = [x1-x0, y1-y0, z1-z0]
    l = [0,0,1]
    return np.dot(n,l)/ (np.linalg.norm(n) * np.linalg.norm(l))


def draw_triangle(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    #color = np.random.randint(0,255)
    color = -255 * clipping_faces(normal(x0,y0,z0, x1, y1, z1, x2, y2, z2))
    xmin = int(min(x0, x1, x2))
    if (xmin < 0): xmin = 0
    ymin = int(min(y0, y1, y2))
    if(ymin < 0): ymin = 0
    xmax = int(max(x0, x1, x2)) + 1
    if (xmax > 1000): xmax = 1000
    ymax = int(max(y0, y1, y2)) + 1
    if(ymax > 1000): ymax = 1000

    for x in range(xmin, xmax):
        for y in range(ymin, ymax):
          bc1, bc2, bc3 = baricentric_coord( x + 0.5, y + 0.5, x0, y0, x1, y1, x2, y2)
          if bc1 >= 0 and bc2 >= 0 and bc3 >= 0:
            z = bc1*z0 + bc2*z1 + bc3*z2
            if z < z_buff[y, x]:
                  z_buff[y,x] = z
                  img_mat[y, x] = [color,0,color]
          

scale = 5000
img_mat = np.zeros((1000,1000,3), dtype = np.uint8) #беззнаковое целочисленное 8-битное, полутоновый -> цветной белый квадрат


def make_line (img_mat, x0, y0, x1, y1):
    d_max = max(abs(floor(x0) - floor(x1)), abs(floor(y0) - floor(y1)) )
    L = d_max + 1
    if L == 1:
        img_mat[floor(y0), floor(x0)] = 255 #ставим точку
        return
    d_x = (x1 - x0)/ (L - 1)
    d_y = (y1 - y0)/ (L - 1)

    x = x0
    y = y0
    for _ in range(L): 
        img_mat[floor(y), floor(x)] = 255
        x += d_x
        y += d_y

#x0,y0,x1,y1 = 30.5, 40.1, 500.0, 950.9
z_buff = np.full((1000, 1000), np.inf, dtype = np.float32)


v = []
f = []
file = open('model.obj')
for s in file:
    spl = s.split()
    if(spl[0] == 'v'):
        v.append([float(spl[1]), float(spl[2]), float(spl[3])])
    elif (spl[0] == 'f' ):
        f.append( [spl[1].split('/')[0], spl[2].split('/')[0], spl[3].split('/')[0]] )#берем первую часть
        
color = np.random.randint(0,255)

for i in range(len(v)):
    x =  v[i][0] * scale + 500 #добавляем центр изображения, переносим его
    y = -v[i][1] * scale + 500
    img_mat[int(y), int(x)] = [color, 0, color]   


for i in range (len(f)):   # f[i] - номера вершин i-ого полигона, f[i][0] - номера нулевой вершины i-ого полигона
    x0 = v[int(f[i][0]) - 1][0] * scale + 500  # x координата нулевой точки i-ого полигона
    y0 = -v[int(f[i][0]) - 1][1] * scale + 500
    z0 = v[int(f[i][0]) - 1][2] * scale + 500
    x1 = v[int(f[i][1]) - 1][0] * scale + 500
    y1 = -v[int(f[i][1]) - 1][1] * scale + 500
    z1 = v[int(f[i][1]) - 1][2] * scale + 500
    x2 = v[int(f[i][2]) - 1][0] * scale + 500
    y2 = -v[int(f[i][2]) - 1][1] * scale + 500
    z2 = v[int(f[i][2]) - 1][2] * scale + 500

    #make_line(img_mat, x0, y0, x1, y1)
    #make_line(img_mat, x0, y0, x2, y2)
    #make_line(img_mat, x1, y1, x2, y2)
    if (clipping_faces(normal(x0, y0, z0, x1, y1, z1, x2, y2, z2)) < 0):
        draw_triangle(x0, y0, z0, x1, y1, z1, x2, y2, z2)


img = Image.fromarray(img_mat)
img.save('img.png')