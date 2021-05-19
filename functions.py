import cv2
import numpy as np
from tensorflow import keras

##all functions here are listed in the order by which they are called in the main.py##


# cell border size, used to detect digits
border = 70
# a shift, used to make sure that boarder pizels do not contribute in digit detection
delta = (100-border)/2

model = keras.models.load_model('mnist_model')


def all_cells():
    # returns a tuple of (top-left corner) coordinates of every cell in the sodoku board,
    # projected to resolution of 900x900
    board_coordinates = []
    for i in range(9):
        for j in range(9):
            x0 = int(delta+(border+2*delta)*i)
            y0 = int(delta+(border+2*delta)*j)
            cell = (x0, y0)
            board_coordinates.append(cell)
    board_coordinates = tuple(board_coordinates)
    return board_coordinates


def gray_gaus_thresh(frame):
    # takes a frame, applies grayscale, gaussian blur, thresholding and returns a result
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gaus = cv2.GaussianBlur(gray, (5, 5), 0)
    thresholded_gaus_adaptive = cv2.adaptiveThreshold(
        gaus, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.CHAIN_APPROX_NONE, 21, 10)
    return thresholded_gaus_adaptive


def contour(input_frame):
    # takes a processed frame, finds contours, returns the biggest (by area) one, its corners cordinates and area
    contours, hierarchy = cv2.findContours(
        input_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    cnt = contours[0]
    perim = cv2.arcLength(cnt, True)
    epsilon = 0.03*perim
    approxCorners = cv2.approxPolyDP(cnt, epsilon, True)
    area = cv2.contourArea(cnt)

    return cnt, approxCorners, area


def order_corners(corners):
    # takes corners corninates and sorts it in order
    # this order is needed to make sure that corners match for cv2.wrapPerspective in fit_sudoku function
    top_left = None
    top_right = None
    bottom_left = None
    bottom_right = None
    mid_x = (corners[0][0][0] + corners[1][0][0] +
             corners[2][0][0] + corners[3][0][0])/4
    mid_y = (corners[0][0][1] + corners[1][0][1] +
             corners[2][0][1] + corners[3][0][1])/4

    for i in range(4):
        if corners[i][0][0] > mid_x:
            if corners[i][0][1] > mid_y:
                bottom_right = corners[i][0]
            else:
                top_right = corners[i][0]
        else:
            if corners[i][0][1] > mid_y:
                bottom_left = corners[i][0]
            else:
                top_left = corners[i][0]

    return [top_left, top_right, bottom_right, bottom_left]


def fit_sudoku(input_img, approxCorners):
    # takes the processed frame and sorted coordinates of the corners
    # returns a bird-view 900x900 projected image of the sudoku board
    pts_src = np.array(approxCorners)
    pts_dst = np.array([[0, 0], [900, 0], [900, 900], [0, 900]])
    h, status = cv2.findHomography(pts_src, pts_dst)
    im_out = cv2.warpPerspective(input_img, h, (900, 900))

    return im_out


def detect_cells(all_cells, matrix_img):
    # takes the tuple of all cells coordinates and the 900x900 image of the sudoku board
    # returns an integer of the total detected numbers and the dictionary, where keys are coordinates of the detected numbers
    # and values are None
    digit_dict = dict()
    total = 0
    shift = 20

    for x, y in all_cells:

        x0 = x + shift
        y0 = y + shift
        x1 = x + border - shift
        y1 = y + border - shift

        current_cell = matrix_img[y0:y1, x0:x1]
        non_zero_pixels = np.count_nonzero(current_cell)

        if non_zero_pixels > 100:
            # storing coordinates of non-empty cells
            digit_dict[(x, y)] = None
            total += 1

    return total, digit_dict


def center_cell(my_array):
    # takes a numpy array and centers the non-zero values
    # this is used in recognize_digits in order to improve CNN model prediction accuracy
    for k in range(2):
        nonempty = np.nonzero(np.any(my_array, axis=1-k))[0]
        first, last = nonempty.min(), nonempty.max()
        shift = (my_array.shape[k] - first - last)//2
        my_array = np.roll(my_array, shift, axis=k)
    return my_array


def recognize_digits(digit_dict, matrix_img, read_digits):
    # takes the dictionary, where keys are (x,y) of the detected cells and values are None
    # at the first call read_digits=0
    # checks the cells which contains digits and tries to detect it
    # if the digit was detected with confidence greater than 0.9, the value is saved in the dictionary
    # if not, then the value stays None
    # this function is looped frame-by-frame until the read_digits == total_digits
    shift = 20

    for key in digit_dict:

        if digit_dict[key] == None:

            left = key[0]
            top = key[1]
            right = key[0]+border
            bottom = key[1]+border
            current_cell = matrix_img[top:bottom, left:right]

            img = cv2.resize(current_cell, (28, 28),
                             interpolation=cv2.INTER_AREA)
            ret, img = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)

            img = center_cell(img)
            img = img / 255

            img = img.reshape(1, 28, 28, 1)
            result = model.predict(img)
            number = np.argmax(result)

            if result[0][number] > 0.9:
                digit_dict[key] = number
                read_digits += 1

    return digit_dict, read_digits


def dict_to_array(digit_dict):
    # takes the dictionary with (x,y) coordinates as keys and detected digits as values
    # returns a 9x9 integer array, where empty cells have value of 0
    sudoku_array = [[0]*9 for _ in range(9)]

    for key in digit_dict:
        sudoku_array[key[1]//100][key[0]//100] = digit_dict[key]

    return sudoku_array


def rules_check(grid, row, col, num):
    # this function is used in the solve_sudoku function, whenever a new digit is being tried

    # checks if the new digit violates the row rule
    for x in range(9):
        if grid[row][x] == num:
            return False

    # checks if the new digit violates the column rule
    for x in range(9):
        if grid[x][col] == num:
            return False

    # checks if the new digit violates the 3x3 sub-grid rule
    startRow = row - row % 3
    startCol = col - col % 3
    for i in range(3):
        for j in range(3):
            if grid[i + startRow][j + startCol] == num:
                return False
    return True


def solve_sudoku(grid, row, col):
    # takes the 9x9 array of integers and solves the sudoku recursively
    # it scans the whole board top to down, left to right, trying all possible combinations
    # and backtracking, whenever rules are violated

    # last cell check
    if (row == 8 and col == 9):
        return True

    # end line check
    if col == 9:
        row += 1
        col = 0

    # occupation check
    if grid[row][col] > 0:
        return solve_sudoku(grid, row, col + 1)
    for num in range(1, 10, 1):

        # rules check
        if rules_check(grid, row, col, num):

            # filling the number
            grid[row][col] = num

            # calling the func again on the following cell
            if solve_sudoku(grid, row, col + 1):
                return True

        # if the func did not return, it means the filled digit was wrong, so its removed
        grid[row][col] = 0
    return False


def display_result_on_contour(img, corners, array, digit_dict, cells_coord):

    # takes the sudoku board, defined by the contour and projects it
    # into 900x900 image, just like in fit_sudoku function
    pts_src = np.array(corners)
    pts_dst = np.array([[0, 0], [900, 0], [900, 900], [0, 900]])
    h, status = cv2.findHomography(pts_src, pts_dst)
    im_out = cv2.warpPerspective(img, h, (900, 900))

    font = cv2.FONT_HERSHEY_SIMPLEX

    # writes the solution digits in the empty cells
    for cell in cells_coord:
        if cell not in digit_dict:
            i = cell[0] // 100
            j = cell[1] // 100
            x = 50 + 100*i - 20
            y = 50 + 100*j + 20
            cv2.putText(im_out, str(array[j][i]), (x, y),
                        font, 2, (0, 0, 255), 4, cv2.LINE_AA)

    # Creating a copy of solved board image
    virtualBillboardClone = img.copy()

    # in order to paste the board correct, corner coordinates has to be lowered by 1
    pts_banner = np.array([[0, 0], [im_out.shape[1] - 1, 0],
                          [im_out.shape[1] - 1, im_out.shape[0] - 1], [0, im_out.shape[0] - 1]])

    # Calculate homography
    homographyMat, status = cv2.findHomography(pts_banner, pts_src)

    # shape the sudoku board into the contour perspective
    result1 = cv2.warpPerspective(
        im_out, homographyMat, (img.shape[1], img.shape[0]))

    # Black out the polygonal contour are in the source frame
    cv2.fillConvexPoly(virtualBillboardClone, pts_src, 0, 16)

    # Add warped sudoku board back into the frame
    result2 = virtualBillboardClone + result1

    # this function makes sure that the text written on the frame rotates together with the contour and keeps the same persepctive

    return result2
