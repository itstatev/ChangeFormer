import os
import cv2
from PIL import Image
from similarity_score import cosine_similarity


img_dir = 'images/'
names = [file for file in sorted(os.listdir(os.path.join(img_dir + 'A/')))]

def sliding_window(window_size, search_window_size, stride):
    for name in names:
        img1 = cv2.imread(img_dir + 'A/' + name)
        img2 = cv2.imread(img_dir + 'B/' + name)
        print('shape', img2.shape)
        height = img1.shape[0]
        width = img1.shape[1]
        print('height, width', height, width)
        range_start = [int(window_size[0] / 2), int(window_size[1] / 2)]
        range_end = [height - int(window_size[1] / 2), width - int(window_size[0] / 2)]
        print('range_start, range_end', range_start, range_end)
        for y in range(range_start[0], range_end[0] + 1, stride):
            for x in range(range_start[1], range_end[1] + 1, stride):
                # print('y, x', y, x)

                window = img1[x - int(window_size[0] / 2) : x + int(window_size[0] / 2), y - int(window_size[1] / 2) : y + int(window_size[1] / 2), :]
                # cv2.imwrite(f'window{x}.jpg', window)
                # input()
                if (x - int(search_window_size[0] / 2)) < 0:
                    neg_part = int(search_window_size[0] / 2) - x
                    x_coord = [0, x + int(search_window_size[0] / 2) + neg_part] 
                elif (width - (x + int(search_window_size[0] / 2)) < 0):
                    neg_part = int(search_window_size[0] / 2) - (width - x) 
                    x_coord = [x - int(search_window_size[0] / 2) - neg_part, width]
                else:
                    x_coord = [x - int(search_window_size[0] / 2), x + int(search_window_size[0] / 2)]

                if y - int(search_window_size[1] / 2) < 0:
                    neg_part = int(search_window_size[1] / 2) - y
                    y_coord = [0, y + int(search_window_size[1] / 2) + neg_part]
                elif (height - (y + int(search_window_size[1] / 2)) < 0):
                    neg_part = int(search_window_size[1] / 2) - (height - y)
                    # print('neg from bottom', y,  neg_part)
                    y_coord = [y - int(search_window_size[1] / 2) - neg_part, height]
                else:
                    y_coord = [y - int(search_window_size[1] / 2), y + int(search_window_size[1] / 2)] 
                
                # if x == 240:
                #     print('x', x, y)
                #     print('x_coord, y_coord', x_coord, y_coord)
                # print('x_coord, y_coord', x_coord, y_coord)
                # if x_coord[0] == 95 and x_coord[1] == 145 and y_coord[0] == 165:
                #     search_window = img2[x_coord[0]:x_coord[1], y_coord[0]: y_coord[1], :]
                #     cv2.imwrite('windowCHECK.jpg', window)
                #     cv2.imwrite('search_windowCHECK.jpg', search_window)
                #     print('y, x', y, x, window.shape, search_window.shape, img2.shape)
                #     input()
                
                # print('x_coord', 'y_coord', x_coord, y_coord)
                search_window = img2[x_coord[0]:x_coord[1], y_coord[0]: y_coord[1], :]
    
                imgA = Image.fromarray(window)
                imgB = Image.fromarray(search_window)
                score = cosine_similarity(imgA, imgB)
                print(score)

                # cv2.imwrite(f'window{x}.jpg', window)
                # cv2.imwrite(f'search_window{x}.jpg', search_window)
    # return window, search_window
        return cosine_similarity 
                
print(sliding_window([10, 10], [50, 50], 5))