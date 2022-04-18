this is a program, which solves the sudoku, placed in front of the camera.
DEMO: https://www.youtube.com/watch?v=WkNf8CDHrJ8


CONTENTS:

- CNN_model.py: Convolutional neural network architecture which is trained on mnist dataset available in keras library. 
this script is used to createmnist_model file.\

- mnist_model: the result model file which is created after running CNN_model.py.
it is imported and used by the functions.py file

- functions.py: functions which carry out actions used in the main.py, such as
finding the contour, applying image processing, dividing the sudoku board into 9x9 cells, detecting numbers, solving the sudoku and displaying the result

- main.py: the file that runs the whole program. press q to start the detection. press r to reset the data



REQUIRED LIBRARIES:
- cv2
- tensorflow
- matplotlib
- numpy
