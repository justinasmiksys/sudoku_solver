from functions import *

detection_on = False

# count for digits detected after image processing
total_digits = 0

# count for digits read by  CNN
read_digits = 0

# this turns into True, once the sudoku is solved
solved_array = False

# tuple, storing all the coordinates of the cells in the board
cells_coord = all_cells()

# setting the screen size to max resolution supported by webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# an empty image, where the sudoku board will be projected later
sudoku_board = np.zeros((900, 900, 3), dtype=np.uint8)

while True:
    ret, frame = cap.read()
    if ret == True:

        # applying grayscale, gaussian blur and threshiolding the frame
        processed = gray_gaus_thresh(frame)

        # detecting the biggest contour, its corner coordinates and area
        cnt, corners, area = contour(processed)

        if area > 50000 and len(corners) == 4:
            cv2.drawContours(frame, [cnt], 0, (0, 255, 0), 3)

            corners = order_corners(corners)

            sudoku_board = fit_sudoku(processed, corners)

            if solved_array:

                frame = display_result_on_contour(
                    frame, corners, sudoku_array, digit_dict, cells_coord)

        cv2.imshow('contour', frame)

        k = cv2.waitKey(1) & 0xff

        if k == 27:
            break

        elif k == ord('q'):
            # enables the detection
            detection_on = True

            if total_digits == 0:

                total_digits, digit_dict = detect_cells(
                    cells_coord, sudoku_board)

        if detection_on:

            if total_digits != read_digits:
                # this part is repeated until all the digits are recognized
                digit_dict, read_digits = recognize_digits(
                    digit_dict, sudoku_board, read_digits)

            else:
                # once all digits are recognized, sudoku is solved (solved_array=True)
                sudoku_array = dict_to_array(digit_dict)
                solved_array = solve_sudoku(sudoku_array, 0, 0)
                #detection is disabled
                detection_on = False

        elif k == ord('r'):
            # resets everything, so the new board could be solved
            solved_array = False
            total_digits = 0
            read_digits = 0

    else:
        break


cv2.destroyAllWindows()
cap.release()
