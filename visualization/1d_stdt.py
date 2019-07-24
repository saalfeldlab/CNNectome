import matplotlib.pyplot as plt
import matplotlib
import scipy.ndimage
import numpy as np

data_red = np.array(
    [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
)
data = np.repeat(data_red, 20)
sdt = scipy.ndimage.distance_transform_edt(
    data, sampling=(0.05)
) - scipy.ndimage.distance_transform_edt(np.logical_not(data), sampling=(0.05))
stdt = np.tanh(sdt / 2.0)
tanh = np.tanh(np.linspace(-3.0, 3.0))

fig, ax = plt.subplots()
fig.patch.set_visible(False)
ax.axis("off")
plt.axhline(color="gray", linewidth=1)
plt.ylim([-13, 6])
plt.yticks([0])
plt.xticks([])
plt.plot(data, color=[0, 0, 1, 1])
plt.show()

fig, ax = plt.subplots()
fig.patch.set_visible(False)
ax.axis("off")
plt.axhline(color="gray", linewidth=1)
plt.ylim([-13, 6])
plt.yticks([0])
plt.xticks([])
plt.plot(data, color=[0, 0, 1, 0.5], ls="-")
plt.plot(sdt, color=[0, 1, 0, 1])
plt.show()

fig, ax = plt.subplots()
fig.patch.set_visible(False)
ax.axis("off")
plt.axhline(color="gray", linewidth=1)
plt.ylim([-13, 6])
plt.yticks([0])
plt.xticks([])
plt.plot(data, color=[0, 0, 1], ls="-")
plt.plot(sdt, color=[0, 1, 0], ls="-", label="SEDT")
plt.plot(stdt, color=[1, 0, 0], label="STDT")
leg = plt.legend()
leg.get_frame().set_linewidth(0)
plt.show()

fig, ax = plt.subplots()
fig.patch.set_visible(False)
ax.axis("off")
plt.axhline(color="gray", linewidth=1)
plt.axvline(color="gray", linewidth=1)
ax.annotate(
    "1",
    xy=(0, 1),
    xycoords="data",
    xytext=(-5, 0),
    textcoords="offset points",
    horizontalalignment="right",
    verticalalignment="center",
    fontproperties=matplotlib.font_manager.FontProperties(size=15),
)
ax.annotate(
    "-1",
    xy=(0, -1),
    xycoords="data",
    xytext=(-5, 0),
    textcoords="offset points",
    horizontalalignment="right",
    verticalalignment="center",
    fontproperties=matplotlib.font_manager.FontProperties(size=15),
)

plt.ylim([-1, 1])
plt.yticks([-1, 0, 1])
plt.xticks([])
plt.plot(np.linspace(-3.0, 3.0), tanh, color="k")
plt.show()
