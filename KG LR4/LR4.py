import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from numba import njit
from math import sin, cos

WIDTH = 1000
HEIGHT = 1000
SCALE = 5000


@njit(cache=True)
def baricentric_coord (x, y, x0, y0, x1, y1, x2, y2):
    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda2 = 1.0 - lambda0 - lambda1
    return lambda0, lambda1, lambda2

@njit(cache=True)
def clipping_faces(n):
    p = np.array([0,0,1])
    return (n[0]*p[0] + n[1]*p[1] + n[2]*p[2]) / (np.sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]) * np.sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2]))

@njit(cache=True)
def normal(x0,y0,z0,x1,y1,z1,x2,y2,z2):
    a = np.array([x1-x2, y1-y2, z1-z2])
    b = np.array([x1-x0, y1-y0, z1-z0])
    return np.cross(a,b)


@njit(cache=True)
def draw_triangle(px0, py0, z0, px1, py1, z1, px2, py2, z2, color, z_buf, img_mat):
    h, w = img_mat.shape[0:2]
    
    xmin = int(min (min(px0, px1), px2))
    ymin = int(min (min(py0, py1), py2))
    xmax = int(max (max(px0, px1), px2)) + 1
    ymax = int(max (max(py0, py1), py2)) + 1
        
    if xmin < 0: xmin = 0
    if ymin < 0: ymin = 0
    if xmax > w: xmax = w
    if ymax > h: ymax = h       

    for x in range(xmin, xmax):
        for y in range(ymin, ymax):
            bc1, bc2, bc3 = baricentric_coord(x + 0.5, y + 0.5, px0, py0, px1, py1, px2, py2)
            if bc1 >= 0 and bc2 >= 0 and bc3 >= 0:
                z = bc1 * z0 + bc2 * z1 + bc3 * z2
                if z < z_buf[y,x]:
                    z_buf[y,x] = z
                    img_mat[y, x, 0] = color[0]
                    img_mat[y, x, 1] = color[1]
                    img_mat[y, x, 2] = color[2]

class Scene:
    def __init__(self, filename = 'model.obj'):
      self.v = []
      self.f = []
      file = open(filename)
      for s in file:
        spl = s.split()
        if(len(spl)>0 and spl[0]=='v'):
            self.v.append([float(spl[1]),float(spl[2]), float(spl[3])])
        elif(len(spl)>0 and spl[0]=='f'):
            self.f.append([int(spl[1].split('/')[0])-1, int(spl[2].split('/')[0])-1, int(spl[3].split('/')[0])-1])

      self.v = np.array(self.v, dtype=np.float32)
      self.f = np.array(self.f, dtype=np.int32)

class Renderer:
    def __init__(self, W, H, scale):
       self.H, self.W = H, W
       self.scale = scale
       self.img_mat  = np.zeros((self.H, self.W, 3), dtype = np.uint8)
       self.z_buf = np.full((self.H, self.W,), np.inf, dtype = np.float32)
       self.shift_x = W / 2.0
       self.shift_y = H / 2.0

       self.alpha, self.beta, self.gamma = 0.0, 0.6, 0.0
       self.tx, self.ty, self.tz = 0.0, 0.0, 1.0

       self.is_perspective = True

    def get_rotation_matrix(self, alpha, beta, gamma):
        Rx = np.array([
         [1.0, 0.0, 0.0],
         [0.0, cos(alpha), sin(alpha)],
         [0.0, -sin(alpha), cos(alpha)]
        ], dtype=np.float32)

        Ry = np.array([
         [cos(beta), 0.0, sin(beta)],
         [0.0, 1.0, 0.0],
         [-sin(beta), 0.0, cos(beta)]
        ], dtype=np.float32)

        Rz = np.array([
         [cos(gamma), sin(gamma), 0.0],
         [-sin(gamma), cos(gamma), 0.0],
         [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        return Rx @ Ry @ Rz

    def transform(self, v, alpha, beta, gamma, tx, ty, tz):
       R = self.get_rotation_matrix(alpha, beta, gamma)
       return v @ R.T + np.array([tx, ty, tz])

    def render(self, scene):
       arr = self.transform(scene.v, self.alpha, self.beta, self.gamma, self.tx, self.ty, self.tz)
       render_model(self.img_mat, self.z_buf, arr, scene.f, self.shift_x, self.shift_y, self.scale, self.is_perspective)

            
@njit(cache=True)
def render_model(img_mat, z_buf, v, f, shift_x, shift_y, scale, is_perspective):
    img_mat[...] = 0
    z_buf[...] = np.inf

    for f0, f1, f2 in f:
        x0, y0, z0 = v[f0]
        x1, y1, z1 = v[f1]
        x2, y2, z2 = v[f2]

        c = clipping_faces(normal(x0, y0, z0, x1, y1, z1, x2, y2, z2))
        if c < 0:
            if is_perspective:
                if z0 <= 1e-4 or z1 <= 1e-4 or z2 <= 1e-4: continue
                px0, py0 = (x0 * scale) / z0 + shift_x, -(y0 * scale) / z0 + shift_y
                px1, py1 = (x1 * scale) / z1 + shift_x, -(y1 * scale) / z1 + shift_y
                px2, py2 = (x2 * scale) / z2 + shift_x, -(y2 * scale) / z2 + shift_y
            else:
                px0, py0 = x0 * scale + shift_x, -y0 * scale + shift_y
                px1, py1 = x1 * scale + shift_x, -y1 * scale + shift_y
                px2, py2 = x2 * scale + shift_x, -y2 * scale + shift_y
         
            color = (100, int(-255 * c), 255)
            draw_triangle(px0, py0, z0, px1, py1, z1, px2, py2, z2, color, z_buf, img_mat)

class App:
    def __init__(self, scene, renderer, win_w = 900, win_h = 900):
        self.scene = scene
        self.renderer = renderer
        # Оставляем как есть, Окно верхнего уровня
        self.root = tk.Tk()
        # Размеры окна
        self.root.geometry(f"{win_w}x{win_h}")
        # Область для рисования внутри окна
        self.canvas = tk.Canvas(self.root, bg='black', highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)
        self.canvas.focus_set()
        # Обновление изображения при изменении размеров окна
        self.canvas.bind('<Configure>', lambda e: self._refit())
        # Привязка обработчика клавиш
        self.root.bind('<Key>', self.on_key)

        # Изображение внутри окна
        self.photo = None
        self.full_image = None

        self.render()

    def on_key(self, event):
        k = event.keysym
        
        if k in ('left', 'a'):
            self.renderer.tx -= 0.02
        elif k in ('right', 'd'):
            self.renderer.tx += 0.02
        elif k in ('up', 'w'):
            self.renderer.ty += 0.02
        elif k in ('down', 's'):
            self.renderer.ty -= 0.02

        
        elif k == 'z':
            self.renderer.tz += 0.05
        elif k == 'x':
            self.renderer.tz -= 0.05


        elif k == 'q':
            self.renderer.beta -= 0.1
        elif k == 'e':
            self.renderer.beta += 0.1
        elif k == 'r':
            self.renderer.alpha -= 0.1
        elif k == 'f':
            self.renderer.alpha += 0.1

    
        elif k == 'p':
            self.renderer.is_perspective = not self.renderer.is_perspective


        elif k in ('plus', 'equal'):
            self.renderer.scale *= 1.1
        elif k in ('minus'):
            self.renderer.scale /= 1.1
        elif k == 'space':
            Image.fromarray(self.renderer.img_mat).save('IMG.png')
            print("Изображение сохранено в файл")
        else:
            return

        self.render()
        # Не обрабатываем дальше – return 'brake'
        return 'break'

    def render(self):
        self.renderer.render(self.scene)
        self.full_image = Image.fromarray(self.renderer.img_mat)
        self._refit()
 
    def _refit(self):
        # Если ещё не отрендерилось, пропускаем
        if self.full_image is None: return
        # Ширина и высота области для рисования
        ww = self.canvas.winfo_width()
        wh = self.canvas.winfo_height()
        # Если вдруг сделали область слишком маленькой
        if ww < 10 or wh < 10: return
        # Вписать с сохранением пропорций
        s = min(ww / self.renderer.W, wh / self.renderer.H)
        dw = max(1, int(self.renderer.W * s))
        dh = max(1, int(self.renderer.H * s))
        disp = self.full_image.resize((dw, dh), Image.BILINEAR)
        # Очень важно сохранить ImageTk.PhotoImage(disp) в поле класса
        # Если оно будет локальным, то удалится после выхода сборщиком мусора
        self.photo = ImageTk.PhotoImage(disp)
        self.canvas.delete('all')
        self.canvas.create_image(ww // 2, wh // 2, image=self.photo)   

    # скопировать как есть
    def run(self):
        self.root.mainloop()

if __name__ == '__main__':
    scene = Scene()
    renderer = Renderer(W = WIDTH, H = HEIGHT, scale = SCALE)

    renderer.render(scene)
    Image.fromarray(renderer.img_mat).save('img.png')
    App(scene, renderer, win_w = 600, win_h = 600).run()