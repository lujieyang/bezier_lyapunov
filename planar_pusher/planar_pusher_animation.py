import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

fig, ax = plt.subplots(figsize=(5, 5))
plt.xlim([-0.35, 0.2])
plt.ylim([-0.2, 0.35])

d = 0.05
traj = np.load("planar_pusher/data/traj.npy")[::4]
u_traj = np.load("planar_pusher/data/u_traj.npy")[::4]

box = Rectangle((0, 0), 2*d, 2*d, 0, edgecolor='b', facecolor='none', antialiased="True")
r = 0.01
circle = plt.Circle(([], []), r, edgecolor='k')
# circle.facecolor = "r"

def init():
    ax.add_patch(box)
    ax.add_artist(circle)
    return box, circle

def animate(n):
    x = traj[n]
    u = u_traj[n]
    theta = x[2]
    x_object = x[0] - d*(np.cos(theta) - np.sin(theta))
    y_object = x[1] - d* (np.cos(theta) + np.sin(theta))

    py = x[-1]
    x_pusher = x[0] - d*np.cos(theta) - py*np.sin(theta) - r * np.cos(theta)
    y_pusher = x[1] - d*np.sin(theta) + py*np.cos(theta) - r * np.sin(theta)

    box.set_x(x_object)
    box.set_y(y_object)
    box.set_angle(theta/np.pi*180)
    box.set_antialiased = True

    if u[-1] == 0:
        facecolor = "r"
    elif u[-1] > 0:
        facecolor = "orange"
    else:
        facecolor = "blueviolet"
    circle.center = (x_pusher, y_pusher)
    circle.facecolor = facecolor
    circle.set_facecolor(facecolor)
    return box, circle

ani = animation.FuncAnimation(
    fig, animate, u_traj.shape[0], interval=10, init_func=init, blit=True)
# plt.show()
ani.save('animation.mp4', fps=15, 
          extra_args=['-vcodec', 'h264', 
                      '-pix_fmt', 'yuv420p'])