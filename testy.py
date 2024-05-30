import cv2
import numpy as np
import dlib
from math import hypot
import pyglet
import time

# Load sounds
sound = pyglet.media.load("sound.m4a", streaming=False)
left_sound = pyglet.media.load("left.m4a", streaming=False)
right_sound = pyglet.media.load("right.m4a", streaming=False)

# Create a primary window
cv2.namedWindow("Primary Window", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Primary Window", 1280, 720)

# Initialize the camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Face detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize the keyboard display
keyboard = np.zeros((600, 1000, 3), np.uint8)

# Initialize the board
board = np.zeros((100, 1400, 3), dtype=np.uint8)
board.fill(255)  # White background for the board

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#Keyboard settings
keyboard = np.zeros((600, 1000, 3), np.uint8)
keys_set = {0: "Food", 1: "Water", 2: "Washroom", 3: "Getup", 4: "Help",
            5: "Medicine", 6: "Clothes", 7: "Bed", 8: "Fan", 9: "Light",
            10: "TV", 11: "Call", 12: "Door", 13: "Window", 14: "<"}

def letter(letter_index,text,letter_light):
    #Keys
    x = (letter_index % 5) * 200  # 5 keys per row
    y = (letter_index // 5) * 200 # 3 rows
    width = 200
    height = 200
    th = 3 # Thickness

    if letter_light is True:
        cv2.rectangle(keyboard,(x+th,y+th),(x+width-th,y+height-th),(255,255,255),-1) #-1 here means fill the rectangle
    else:
        cv2.rectangle(keyboard,(x+th,y+th),(x+width-th,y+height-th),(255,0,0),th)
    

    # Text settings
    font_letter = cv2.FONT_HERSHEY_PLAIN
    font_scale = 10
    font_th = 4 # Font thickness
    text_size = cv2.getTextSize(text, font_letter, font_scale, font_th)[0]
    text_x = int((width - text_size[0]) / 2) + x
    text_y = int((height + text_size[1]) / 2) + y

    cv2.putText(keyboard, text, (text_x, text_y), font_letter, font_scale, (255, 255, 255), font_th)

    # print(text_size)

def draw_menu():
    rows, cols, _ = keyboard.shape
    th_lines = 4 # thickness lines
    cv2.line(keyboard, (int(cols/2) - int(th_lines/2), 0),(int(cols/2) - int(th_lines/2), rows),
             (51, 51, 51), th_lines)
    cv2.putText(keyboard, "LEFT", (50, 300), font, 5, (255, 255, 255), 5)
    cv2.putText(keyboard, "RIGHT", (50 + int(cols/2), 300), font, 5, (255, 255, 255), 5)


def midpoint(p1,p2):
    return int((p1.x+p2.x)/2) , int((p1.y+p2.y)/2)

font = cv2.FONT_HERSHEY_SIMPLEX

def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)

#     hor_line = cv2.line(frame, left_point, right_point,(0,255,0), 2)

    center_top = midpoint(facial_landmarks.part(eye_points[1]),facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]),facial_landmarks.part(eye_points[4]))

#     ver_line = cv2.line(frame, center_top, center_bottom ,(0,255,0), 2)

    #Checking length of horizontal and vertical lines
    ver_line_length = hypot((center_top[0]-center_bottom[0]),(center_top[1]-center_bottom[1]))
    hor_line_length = hypot((left_point[0]-right_point[0]),(left_point[1]-right_point[1]))
        
    ratio = hor_line_length/ver_line_length
    return ratio

def eyes_contour_points(facial_landmarks):
    left_eye = []
    right_eye = []
    for n in range(36, 42):
        x = facial_landmarks.part(n).x
        y = facial_landmarks.part(n).y
        left_eye.append([x, y])
    for n in range(42, 48):
        x = facial_landmarks.part(n).x
        y = facial_landmarks.part(n).y
        right_eye.append([x, y])
    left_eye = np.array(left_eye, np.int32)
    right_eye = np.array(right_eye, np.int32)
    return left_eye, right_eye


def get_gaze_ratio(eye_points, facial_landmarks):
    left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                    (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                    (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                    (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                    (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                    (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)
                                   ],np.int32)
        
#         print(left_eye_region)
#         cv2.polylines(frame, [left_eye_region] , True, (0,0,255),2)

    height, width, _ = frame.shape
    mask = np.zeros((height,width), np.uint8)

    cv2.polylines(mask, [left_eye_region],True, 255,2)
    cv2.fillPoly(mask, [left_eye_region], 255)
    eye = cv2.bitwise_and(gray,gray, mask=mask)        

    min_x = np.min(left_eye_region[:,0])
    max_x = np.max(left_eye_region[:,0])
    min_y = np.min(left_eye_region[:,1])
    max_y = np.max(left_eye_region[:,1])

#         eye = frame[min_y:max_y,min_x:max_x]
#         gray_eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
#         _,threshold_eye = cv2.threshold(gray_eye, 40, 255, cv2.THRESH_BINARY)
    gray_eye = eye[min_y:max_y,min_x:max_x]
    _,threshold_eye = cv2.threshold(gray_eye, 40, 255, cv2.THRESH_BINARY)
    threshold_height, threshold_width = threshold_eye.shape
    left_side_threshold = threshold_eye[0:threshold_height, 0:int(threshold_width/2)]
    left_side_white = cv2.countNonZero(left_side_threshold)

    right_side_threshold = threshold_eye[0:threshold_height, int(threshold_width/2):threshold_width]
    right_side_white = cv2.countNonZero(right_side_threshold)
    
    if left_side_white==0:
        gaze_ratio = 1
    elif right_side_white==0:
        gaze_ratio = 5
    else:
        gaze_ratio = left_side_white/right_side_white
    
    return gaze_ratio

#Frames Counter - Since videos work on frames
frames = 0
letter_index = 0
blinking_frames = 0
frames_to_blink = 7
frames_active_letter = 9
blinked = False

#Text and keyboard settigns
text=""
keyboard_selected = "left"
last_keyboard_selected = "left"
select_keyboard_menu = True
keyboard_selection_frames = 0


while(True):
    _ , frame = cap.read()
    # Resize the frame to fit into the combined image
    frame = cv2.resize(frame, (480, 360))
    rows, cols, _ = frame.shape
    keyboard[:] = (26,26,26)
    frames += 1
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     active_letter = keys_set_1[letter_index]
    
    # Draw a white space for loading bar
    frame[rows - 50: rows, 0: cols] = (150, 150, 150)
    
    if select_keyboard_menu is True:
        draw_menu()
        
    # Keyboard selected
    if keyboard_selected == "left":
        keys_set = keys_set
    else:
        keys_set = keys_set
    active_letter = keys_set[letter_index]
    
    #Face detection
    faces = detector(gray)
    for face in faces:
#         face detection
#         x,y = face.left(),face.top()
#         x1,y1 = face.right(),face.bottom()
#         cv2.rectangle(frame, (x,y),(x1,y1),(0,255,0),2)

        landmarks = predictor(gray,face)
#         print(landmarks.part(36)) # Position of point 36 of face landmarks
#         x = landmarks.part(36).x
#         y = landmarks.part(36).y
#         cv2.circle(frame, (x,y),2,(0,0,255),2)

        left_eye, right_eye = eyes_contour_points(landmarks)
#         Detect Blinking
        left_eye_ratio = get_blinking_ratio([36,37,38,39,40,41], landmarks)
        right_eye_ratio = get_blinking_ratio([42,43,44,45,46,47], landmarks)
        
        blinking_ratio = (left_eye_ratio + right_eye_ratio)/2
        
        # Eyes color
        cv2.polylines(frame, [left_eye], True, (0, 0, 255), 2)
        cv2.polylines(frame, [right_eye], True, (0, 0, 255), 2)
        
        
         
          
        if select_keyboard_menu is True:
            #Three directions covered - Left side, Right side and Center
            #Gaze Detection
            gaze_ratio_left_eye = get_gaze_ratio([36,37,38,39,40,41], landmarks)
            gaze_ratio_right_eye = get_gaze_ratio([42,43,44,45,46,47], landmarks)

            gaze_ratio = (gaze_ratio_left_eye+gaze_ratio_right_eye)/2
    #         print(gaze_ratio)

    #         cv2.putText(frame, str(left_side_white), (50,100), font, 2, (0,0,255), 3)
    #         cv2.putText(frame, str(right_side_white), (50,150), font, 2, (0,0,255), 3)

            if(blinked == True):
                time.sleep(4)
            blinked=False  
            if gaze_ratio<=0.9:
    #             cv2.putText(frame, "RIGHT", (50,150), font, 2, (0,0,255), 3)
    #             new_frame[:] = (0,0,255)
                keyboard_selected = "right"
                keyboard_selection_frames += 1
                # If Kept gaze on one side more than 15 frames, move to keyboard
                if keyboard_selection_frames == 15:
                    select_keyboard_menu = False
                    right_sound.play()
                    # Set frames count to 0 when keyboard selected
                    frames = 0
                    keyboard_selection_frames = 0
                    
                if keyboard_selected != last_keyboard_selected:
                    last_keyboard_selected = keyboard_selected
                    keyboard_selection_frames = 0
            elif 0.49< gaze_ratio < 1.1:
                print("",end="")
            else:
    #             cv2.putText(frame, "LEFT", (50,150), font, 2, (0,0,255), 3)
    #             new_frame[:] = (255,0,0)
                keyboard_selected = "left"
                keyboard_selection_frames += 1
                # If Kept gaze on one side more than 15 frames, move to keyboard
                if keyboard_selection_frames == 15:
                    select_keyboard_menu = False
                    left_sound.play()
                    # Set frames count to 0 when keyboard selected
                    frames = 0
                if keyboard_selected != last_keyboard_selected:
                    last_keyboard_selected = keyboard_selected
                    keyboard_selection_frames = 0
        else:
            # Detect the blinking to select the key that is lighting up
            
            if blinking_ratio > 4.25:
                # cv2.putText(frame, "BLINKING", (50, 150), font, 4, (255, 0, 0), thickness=3)
                blinking_frames += 1
                frames -= 1

                # Show green eyes when closed
                cv2.polylines(frame, [left_eye], True, (0, 255, 0), 2)
                cv2.polylines(frame, [right_eye], True, (0, 255, 0), 2)

                # Typing letter
                if blinking_frames == frames_to_blink:
                                    if active_letter != "<":
                                        text += active_letter
                                        sound.play()
                                        select_keyboard_menu = True
                                        blinked=True
                                    else:
                                        blinking_frames = 0

            

            
            
#         Showing detection
        
        
#         threshold_eye = cv2.resize(threshold_eye, None , fx=5, fy=5)
#         eye = cv2.resize(gray_eye, None , fx=5, fy=5)
# #         cv2.imshow("Eye",eye)
#         cv2.imshow("Threshold", threshold_eye)
# #         cv2.imshow("Mask",mask)
# #         cv2.imshow("Left eye",left_eye)
#         cv2.imshow("left",left_side_threshold)
#         cv2.imshow("right",right_side_threshold)
        
    # Display letters on the keyboard
    if select_keyboard_menu is False:
        if frames == frames_active_letter:
            letter_index += 1
            frames = 0
        if letter_index == 15:
            letter_index = 0
        for i in range(15):
            if i == letter_index:
                light = True
            else:
                light = False
            letter(i, keys_set[i], light)
        
    cv2.putText(board,text,(10,100),font,4,0,3)
    
    # Blinking loading bar
    percentage_blinking = blinking_frames / frames_to_blink
    loading_x = int(cols * percentage_blinking)
    cv2.rectangle(frame, (0, rows - 50), (loading_x, rows), (51, 51, 51), -1)


    # # primary window is true then display thw windows inside the primary window
    # cv2.moveWindow("Frame", 10, 30)  # Move the "Frame" window to (10, 30) position within the primary window
    # cv2.imshow("Frame", frame)

    # cv2.moveWindow("Virtual Keyboard", 10, 300)  # Move the "Virtual Keyboard" window to (10, 300) position
    # cv2.imshow("Virtual Keyboard", keyboard)

    # cv2.moveWindow("Board", 800, 30)  # Move the "Board" window to (800, 30) position
    # cv2.imshow("Board", board)

    # Create a combined image with all sub-windows
    combined_image = np.zeros((720, 1280, 3), dtype=np.uint8)

    # Calculate the position to center the video frame
    video_x = int((1280 - 480) / 2)  # Calculate the x-coordinate to center the video frame
    video_y = 10 # Calculate the y-coordinate to center the video frame

    combined_image[video_y:video_y+360, video_x:video_x+480] = frame  # Centered Frame
    keyboard_y = 400  # Set the y-coordinate to move the virtual keyboard up
    combined_image[keyboard_y:keyboard_y+200, :1280] = cv2.resize(keyboard, (1280, 200))  # Virtual Keyboard resized

    # Resize the board to fit the available width in combined_image and convert to 3-channel
    board_resized = cv2.resize(board, (1280, 80))  # Resize board to fit within 1280 width
    board_resized_color = board_resized  # It's already a BGR image, no need to convert

    combined_image[640:720, :1280] = board_resized_color  # Place the board in the combined image

    # Show the combined image in the primary window
    cv2.imshow("Primary Window", combined_image)

    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()






