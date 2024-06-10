import numpy as np
from tqdm import tqdm


def check_nonzero(a, x, y):
    try:
        if 0 <= x < a.shape[0] and 0 <= y < a.shape[1]:
            return a[x, y] == 1
        return False
    except IndexError:
        return False


def nearest_nonzero_idx(a, x, y):
    try:
        if 0 <= x < a.shape[0] and 0 <= y < a.shape[1]:
            if a[x, y] != 0:
                return x, y
    except IndexError:
        pass

    r,c = np.nonzero(a)
    min_idx = ((r - x)**2 + (c - y)**2).argmin()
    return r[min_idx], c[min_idx]


def main(id):
    # IMAGE_SCALE_DOWN = 8
    img_file_list = ['seq_eth', 'seq_hotel', 'students003', 'crowds_zara01', 'crowds_zara02'][id]
    print(img_file_list)

    import PIL.Image as Image
    img = Image.open(f'./datasets/image/{img_file_list}_map.png')
    img = img.convert('RGB')
    img = np.array(img)
    # img = img[::IMAGE_SCALE_DOWN, ::IMAGE_SCALE_DOWN, :]
    img = img[:, :, 0]
    img = img > 0.5
    img = img.astype(np.int32)

    img_padded = np.pad(img, ((img.shape[0] // 2,) * 2, (img.shape[1] // 2,) * 2), 'constant', constant_values=0)
    print(img.shape, img_padded.shape)
    img = img_padded

    vector_field = np.zeros(img.shape + (2,))
    pbar = tqdm(total=img.shape[0] * img.shape[1])
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            vector_field[x, y] = np.array(nearest_nonzero_idx(img, x, y))
            pbar.update(1)
    pbar.close()

    # Faster version with ProcessPoolExecutor()
    # import concurrent.futures
    # def nearest_nonzero_idx_wrapper(args):
    #     return nearest_nonzero_idx(img, args[0], args[1])

    # vector_field_fast = np.zeros(img.shape + (2,))
    # pbar = tqdm(total=img.shape[0] * img.shape[1])
    # with concurrent.futures.ProcessPoolExecutor(max_workers=64) as executor:
    #     coords = []
    #     for x in range(img.shape[0]):
    #         for y in range(img.shape[1]):
    #             print(x, y)
    #             coords.append((x, y))
    #             # future = executor.submit(nearest_nonzero_idx_wrapper, (img, x, y))
    #             # vector_field_fast[x, y] = future.result()

    #     for coord, vector in zip(coords, executor.map(nearest_nonzero_idx_wrapper, coords)):
    #         vector_field_fast[coord[0], coord[1]] = vector
    #         pbar.update(1)

    # print("allcolse:", np.allclose(vector_field, vector_field_fast))
    # pbar.close()

    np.save(f"./datasets/vectorfield/{img_file_list}_vector_field.npy", vector_field)
    # np.savetxt(f"./datasets/vectorfield/{img_file_list}_vector_field_x.txt", vector_field[:, :, 0], fmt='%d')
    # np.savetxt(f"./datasets/vectorfield/{img_file_list}_vector_field_y.txt", vector_field[:, :, 1], fmt='%d')


if "__main__" == __name__:
    for i in range(5):
        main(id=i)
