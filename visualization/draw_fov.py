import matplotlib.image
import matplotlib.pyplot


def draw_fov(raw_file, fov_list, color, target_file):
    print(target_file)
    arr = matplotlib.image.imread(raw_file)
    print(arr.shape)
    midpoint = (int(arr.shape[0] / 10.0 / 2), int(arr.shape[1] / 2))
    print(midpoint)
    for fov in fov_list:
        fov_offset = (int(fov[0] / 2), int(fov[1] / 2.0))

        # arr[10*(midpoint[0]-fov_offset[0]):10*(midpoint[0]+fov_offset[0]+1), midpoint[1]-fov_offset[1]:
        # midpoint[
        # 1]+fov_offset[1]+1,:] = color

        # print("lower",10*(midpoint[0]-fov_offset[0]), midpoint[1]-fov_offset[1],midpoint[1]+fov_offset[1]+1)
        arr[
            10 * (midpoint[0] - fov_offset[0]),
            midpoint[1] - fov_offset[1] : midpoint[1] + fov_offset[1] + 1,
            :,
        ] = color
        # print("upper",10 * (midpoint[0] + fov_offset[0]+1)-1, midpoint[1] - fov_offset[1],midpoint[1] + fov_offset[1] + 1)
        arr[
            10 * (midpoint[0] + fov_offset[0] + 1) - 1,
            midpoint[1] - fov_offset[1] : midpoint[1] + fov_offset[1] + 1,
            :,
        ] = color
        # print("left",10*(midpoint[0]-fov_offset[0]),10*(midpoint[0]+fov_offset[0]+1), midpoint[1]-fov_offset[1])
        arr[
            10 * (midpoint[0] - fov_offset[0]) : 10 * (midpoint[0] + fov_offset[0] + 1),
            midpoint[1] - fov_offset[1],
            :,
        ] = color
        # print("right",10*(midpoint[0]-fov_offset[0]),10*(midpoint[0]+fov_offset[0]+1), midpoint[1]+fov_offset[1])
        arr[
            10 * (midpoint[0] - fov_offset[0]) : 10 * (midpoint[0] + fov_offset[0] + 1),
            midpoint[1] + fov_offset[1],
            :,
        ] = color
    matplotlib.pyplot.imshow(arr)
    matplotlib.pyplot.show()
    # print(upper_line.shape)
    matplotlib.image.imsave(target_file, arr)
    # upper_line =


if __name__ == "__main__":
    raw_file = "/groups/saalfeld/home/heinrichl/figures/raw_xz-290_213.png"
    # list_of_fovs = [(1, 3), (1, 5), (1, 11), (1, 17), (3, 35), (5, 53), (11, 107), (17, 161), (19,179), (21, 197),
    #                (21, 203), (21, 209), (21, 211), (21, 213)]
    list_of_fovs = [
        (3, 3),
        (5, 5),
        (7, 11),
        (9, 17),
        (11, 35),
        (13, 53),
        (15, 107),
        (17, 161),
        (19, 179),
        (21, 197),
        (23, 203),
        (25, 209),
        (27, 211),
        (29, 213),
    ]
    # for no, fov in enumerate(list_of_fovs):
    # fov = (21, 213)
    target_file = raw_file[:-4] + "_all.png"  # .format(no)
    draw_fov(raw_file, list_of_fovs, [1, 0, 0.5], target_file)
